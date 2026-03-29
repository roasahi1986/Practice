import sys
import os

# Get the absolute path of this file's directory and append ../utils to sys.path
_current_dir = os.getcwd()
sys.path.append(os.path.join(_current_dir, '..', 'utils'))

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# Databricks display function - fallback to show() for local execution
try:
    from databricks.sdk.runtime import display
except ImportError:
    display = lambda df: df.show()

from utils.data_validation import (
    load_and_prepare_raw_data,
)
from utils.model_training import (
    extract_top_k_publishers_old,
    prepare_experiment_data_old,
    create_training_dataset,
    prepare_training_features,
)
from utils.feature_engineering import (
    process_top_k_publisher
)

# QPS Caps to iterate over
QPS_CAPS = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

TRAINING_INPUT_DATA_PATH = "s3://exchange/machine_learning/data/42"
TRAINING_PATHS = [
    "s3://exchange/machine_learning/data/42/dt=2025-11-20",
    "s3://exchange/machine_learning/data/42/dt=2025-11-21",
    "s3://exchange/machine_learning/data/42/dt=2025-11-22",
    "s3://exchange/machine_learning/data/42/dt=2025-11-23",
    "s3://exchange/machine_learning/data/42/dt=2025-11-24",
    "s3://exchange/machine_learning/data/42/dt=2025-11-25",
    "s3://exchange/machine_learning/data/42/dt=2025-11-26"
]
TESTING_INPUT_DATA_PATH = "s3://exchange/machine_learning/data/42"
TEST_PATHS = [
    "s3://exchange/machine_learning/data/42/dt=2025-11-27",
    "s3://exchange/machine_learning/data/42/dt=2025-11-28",
    "s3://exchange/machine_learning/data/42/dt=2025-11-29"
]

RENAME_COLUMNS = {"rtb_connection_id": "rtb_id"}
EXP_BUCKET_CONTROL = "project_v1_control"
EXP_BUCKET_TEST = ["48", "49"]
MULTIPLIER = 3
INCREASE_RATIO_MIN = 1e-3

params = {
     "feature_selected": [
        "rtb_id",
        "supply_name",
        "pub_app_object_id",
        "geo",
        "placement_type",
        "platform",
        "do_not_track",
        "major_os_version"
    ],
    "weight": "explore.original_bid_requests",
    "y": "unr_uplift_per_original_br",
    "top_k_pub_size": 5000
}

# --- 1. Load and Prepare Training Data ---
raw_df = load_and_prepare_raw_data(
    TRAINING_INPUT_DATA_PATH,
    TRAINING_PATHS,
    RENAME_COLUMNS
)
top_k_publishers = extract_top_k_publishers_old(
    raw_df,
    params["top_k_pub_size"],
    EXP_BUCKET_CONTROL
)

raw_df = load_and_prepare_raw_data(
    TESTING_INPUT_DATA_PATH,
    TEST_PATHS,
    RENAME_COLUMNS
)
processed_df = process_top_k_publisher(raw_df, top_k_publishers)

numerical_features = []
categorical_features = params["feature_selected"] + ["rtb_account_id"]
all_features = list(set(numerical_features + categorical_features))

experiment_df = prepare_experiment_data_old(
    processed_df,
    all_features,
    EXP_BUCKET_CONTROL,
    EXP_BUCKET_TEST
)
training_df = create_training_dataset(
    experiment_df, 
    all_features,
)
final_training_df = prepare_training_features(
    training_df,
    all_features,
    params["y"],
    params["weight"]
)

# --- 2. Prepare Global Objects (Common across all caps) ---
all_rtb_ids = final_training_df.select("rtb_account_id").distinct()

latest_path = (
    spark.read.format("binaryFile")
    .load("/FileStore/smart_qps/qps_cap_by_dsp_*.csv")
    .orderBy(F.col("modificationTime").desc())
    .limit(1)
    .select("path")
    .first()["path"]
)
print(f"✅ Latest file path for QPS cap data: {latest_path}")

pred_df_grouped = final_training_df.withColumnRenamed("y", "y_pred")
w_rtb = Window.partitionBy("rtb_account_id")
w_cum = Window.partitionBy("rtb_account_id").orderBy(F.desc("y_pred"))

pred_df_cum = (
    pred_df_grouped
      .withColumn("rtb_sum_weight", F.sum("weight").over(w_rtb))
      .withColumn("cumulative_sum_weight", F.sum("weight").over(w_cum))
      .withColumn("percentile", F.col("cumulative_sum_weight") / F.col("rtb_sum_weight"))
)
pred_df_cum.persist()

# --- 3. Prepare Test Data (Joined Ground Truth) ---
print("Preparing Test Data...")
raw_test_df = spark.read.option("basePath", TESTING_INPUT_DATA_PATH).parquet(*TEST_PATHS)
raw_test_df = raw_test_df.withColumnsRenamed(RENAME_COLUMNS)
raw_test_df = process_top_k_publisher(raw_test_df, top_k_publishers)

control_df = raw_test_df.filter(F.col("exp_bucket") == EXP_BUCKET_CONTROL)
explore_df = raw_test_df.filter(F.col("exp_bucket").isin(EXP_BUCKET_TEST))

group_cols = list(set(params["feature_selected"])) + ["rtb_account_id"]

control_df = (
    control_df.groupBy(group_cols)
    .agg(
        F.sum("bid_requests").alias("bid_requests"),
        F.sum("multiplied_bid_requests").alias("multiplied_bid_requests"),
        F.sum("unr").alias("unr"),
    )
    .withColumn(
        "original_bid_requests",
        F.col("bid_requests") - F.col("multiplied_bid_requests"),
    )
    .filter(F.col("original_bid_requests") > 0)
    .withColumn(
        "unr_per_original_br",
        F.col("unr") * 1_000_000_000 / F.col("original_bid_requests"),
    )
)

explore_df = (
    explore_df.groupBy(group_cols)
    .agg(
        F.sum("bid_requests").alias("bid_requests"),
        F.sum("multiplied_bid_requests").alias("multiplied_bid_requests"),
        F.sum("unr").alias("unr"),
    )
    .withColumn(
        "original_bid_requests",
        F.col("bid_requests") - F.col("multiplied_bid_requests"),
    )
    .filter(F.col("original_bid_requests") > 0)
    .withColumn(
        "unr_per_original_br",
        F.col("unr") * 1_000_000_000 / F.col("original_bid_requests"),
    )
)

control_renamed = control_df.select(
    *[F.col(c) for c in group_cols],
    F.col("unr").alias("control_unr"),
    F.col("original_bid_requests").alias("control_br"),
    F.col("unr_per_original_br").alias("control_unr_rate")
)

explore_renamed = explore_df.select(
    *[F.col(c) for c in group_cols],
    F.col("unr").alias("explore_unr"),
    F.col("original_bid_requests").alias("explore_br"),
    F.col("unr_per_original_br").alias("explore_unr_rate")
)

joined_test_df = control_renamed.join(
    explore_renamed,
    on=group_cols,
    how="inner"
)

joined_test_df = joined_test_df.withColumn(
    "unr_uplift",
    F.col("explore_unr_rate") - F.col("control_unr_rate")
)
joined_test_df.persist()
print("Test Data Prepared.")


# --- 4. Loop over Caps ---

for qps_cap in QPS_CAPS:
    print(f"Processing QPS Cap: {qps_cap}")
    
    # Calculate caps_sdf
    if qps_cap >= 1.0:
        caps_sdf = (
            all_rtb_ids
            .withColumn("cap", F.lit(qps_cap))
            .withColumn("increase_ratio", F.col("cap") - 1)
            .filter(F.col("increase_ratio") >= INCREASE_RATIO_MIN)
            .withColumn("target_pct", F.col("increase_ratio") / F.lit(MULTIPLIER - 1))
            .select("rtb_account_id", "target_pct")
        )
    else:
        CAP_MAX = 3.0
        CAP_MIN = 1.0

        csv_caps = (
            spark.read.option("header", True)
            .csv(latest_path)
            .select("rtb_account_id", F.col("cap").cast("float"))
        )

        caps_sdf = (
            all_rtb_ids.join(csv_caps, "rtb_account_id", "left")
                .fillna(1.0, subset=["cap"])
                .withColumn("cap", F.when(F.col("cap") > CAP_MAX, CAP_MAX)
                                    .when(F.col("cap") < CAP_MIN, CAP_MIN)
                                    .otherwise(F.col("cap")))
                .withColumn("increase_ratio", F.col("cap") - 1)
                .filter(F.col("increase_ratio") >= INCREASE_RATIO_MIN)
                .withColumn("target_pct", F.col("increase_ratio") / F.lit(MULTIPLIER - 1))
                .select("rtb_account_id", "target_pct")
        )
    
    # Calculate threshold_df
    threshold_df = (
        pred_df_cum
          .join(F.broadcast(caps_sdf), "rtb_account_id")
          .filter(F.col("percentile") <= F.col("target_pct"))
          .groupBy("rtb_account_id")
          .agg(F.min("y_pred").alias("threshold"))
    )
    
    # Evaluation Logic
    joined_with_thresh = joined_test_df.join(
        threshold_df,
        on="rtb_account_id",
        how="inner"
    )

    # Important: Cache joined_with_thresh to ensure consistent evaluation within this iteration
    joined_with_thresh.cache()
    
    # Add total_br column for weighting
    joined_with_thresh = joined_with_thresh.withColumn(
        "total_br", F.col("control_br") + F.col("explore_br")
    )

    # Aggregate for ALL rows (denominator for qps_uplift and weighted_control_rate)
    all_rows_agg = joined_with_thresh.groupBy("rtb_account_id").agg(
        F.sum("total_br").alias("total_br_all"),
        F.sum(F.col("control_unr_rate") * F.col("total_br")).alias("weighted_control_rate")
    )

    # Aggregate for treated rows (unr_uplift >= threshold)
    treated_rows_agg = (
        joined_with_thresh
        .filter(F.col("unr_uplift") >= F.col("threshold"))
        .groupBy("rtb_account_id")
        .agg(
            F.sum("total_br").alias("total_br_treated"),
            F.sum(F.col("unr_uplift") * F.col("total_br")).alias("weighted_unr_uplift")
        )
    )

    stats_df = all_rows_agg.join(
        treated_rows_agg, "rtb_account_id", "inner"
    ).fillna(0)

    # qps_uplift = (treated traffic / all traffic) * (MULTIPLIER - 1)
    stats_df = stats_df.withColumn(
        "qps_uplift",
        F.when(
            F.col("total_br_all") > 0,
            (F.col("total_br_treated") / F.col("total_br_all")) * (MULTIPLIER - 1)
        ).otherwise(F.lit(0))
    ).withColumn(
        # unr_uplift = weighted_unr_uplift / weighted_control_rate (for treated rows)
        "unr_uplift",
        F.when(
            F.col("weighted_control_rate") > 0,
            F.col("weighted_unr_uplift") / F.col("weighted_control_rate")
        ).otherwise(F.lit(0))
    )
    
    print(f"Stats for QPS Cap {qps_cap}:")
    display(
        stats_df.select(
            "rtb_account_id",
            "qps_uplift",
            "unr_uplift",
            "total_br_all",
            "total_br_treated"
        )
    )
    
    output_path = f"s3://exchange-dev/luxu/cap_navigation_gt/{qps_cap}"
    print(f"Saving to {output_path}")
    stats_df.write.mode("overwrite").format("parquet").save(output_path)
    
    joined_with_thresh.unpersist()

# Cleanup
pred_df_cum.unpersist()
joined_test_df.unpersist()
