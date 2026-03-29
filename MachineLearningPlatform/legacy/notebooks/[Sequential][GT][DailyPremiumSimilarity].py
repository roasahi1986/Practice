import sys
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# Databricks display function - fallback to show() for local execution
try:
    from databricks.sdk.runtime import display
except ImportError:
    display = lambda df: df.show()

# COMMOND -------------------------------------------------------------
# Get the absolute path of this file's directory and append ../utils to sys.path
_current_dir = os.getcwd()
sys.path.append(os.path.join(_current_dir, '..', 'utils'))

from utils.data_validation import load_and_prepare_raw_data
from utils.model_training import (
    extract_top_k_publishers_old,
    prepare_experiment_data_old,
    create_training_dataset,
    prepare_training_features,
)
from utils.feature_engineering import process_top_k_publisher

# COMMOND -------------------------------------------------------------
# Constants and Configuration
TRAINING_INPUT_DATA_PATH = "s3://exchange/machine_learning/data/42"
# Note: Paths contain the date information: dt=YYYY-MM-DD
TRAINING_PATHS = [
    "s3://exchange/machine_learning/data/42/dt=2025-11-20",
    "s3://exchange/machine_learning/data/42/dt=2025-11-21",
    "s3://exchange/machine_learning/data/42/dt=2025-11-22",
    "s3://exchange/machine_learning/data/42/dt=2025-11-23",
    "s3://exchange/machine_learning/data/42/dt=2025-11-24",
    "s3://exchange/machine_learning/data/42/dt=2025-11-25",
    "s3://exchange/machine_learning/data/42/dt=2025-11-26"
]

RENAME_COLUMNS = {"rtb_connection_id": "rtb_id"}
EXP_BUCKET_CONTROL = "project_v1_control"
EXP_BUCKET_EXPLORE = ["48"] 
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

# COMMOND -------------------------------------------------------------
# 1. Load Common Resources (Cap, Top K Publishers)
print("Loading Reference Data (Top K)...")
# Load full range to get consistent Top K
raw_df_all = load_and_prepare_raw_data(
    TRAINING_INPUT_DATA_PATH,
    TRAINING_PATHS,
    RENAME_COLUMNS
)
top_k_publishers = extract_top_k_publishers_old(
    raw_df_all,
    params["top_k_pub_size"],
    EXP_BUCKET_CONTROL
)

# Load Cap Data (Latest)
latest_path = (
    spark.read.format("binaryFile")
    .load("/FileStore/smart_qps/qps_cap_by_dsp_*.csv")
    .orderBy(F.col("modificationTime").desc())
    .limit(1)
    .select("path")
    .first()["path"]
)
print(f"✅ Latest file path for QPS cap data: {latest_path}")

caps_sdf = (
    spark.read.option("header", True).csv(latest_path)
    .fillna(1.0) 
    .withColumn("cap", F.col("cap").cast("double"))
    .withColumn("increase_ratio", F.col("cap") - 1)
    .filter(F.col("increase_ratio") >= INCREASE_RATIO_MIN)
    .withColumn("target_pct", F.col("increase_ratio") / F.lit(MULTIPLIER))
    .select("rtb_account_id", "target_pct")
)

# COMMOND -------------------------------------------------------------
# Function to get Premium Segments for a specific date path
segment_features = [f for f in params["feature_selected"] if f != "rtb_id"]
def get_premium_segments_for_date(date_path):
    print(f"Processing date: {date_path}...")
    
    # 1. Load Daily Data
    raw_df = load_and_prepare_raw_data(
        TRAINING_INPUT_DATA_PATH,
        [date_path], # Single day
        RENAME_COLUMNS
    )
    
    # 2. Process Features & Uplift
    processed_df = process_top_k_publisher(raw_df, top_k_publishers)
    
    numerical_features = []
    categorical_features = params["feature_selected"] + ["rtb_account_id"]
    all_features = list(set(numerical_features + categorical_features))
    
    experiment_df = prepare_experiment_data_old(
        processed_df,
        all_features,
        EXP_BUCKET_CONTROL,
        EXP_BUCKET_EXPLORE
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
    
    # Rename to y_pred for consistency
    pred_df = final_training_df.withColumnRenamed("y", "y_pred")
    
    # 3. Calculate Thresholds & Identify Premium
    pred_df_with_target = pred_df.join(F.broadcast(caps_sdf), "rtb_account_id", "inner")
    
    w_rtb_id = Window.partitionBy("rtb_account_id")
    w_cum_rtb_id = Window.partitionBy("rtb_account_id").orderBy(F.desc("y_pred"))
    
    pred_df_cum = (
        pred_df_with_target
        .withColumn("rtb_sum_weight", F.sum("weight").over(w_rtb_id))
        .withColumn("cumulative_sum_weight", F.sum("weight").over(w_cum_rtb_id))
        .withColumn("percentile", F.col("cumulative_sum_weight") / F.col("rtb_sum_weight"))
    )
    
    premium_segments_df = (
        pred_df_cum
        .filter(F.col("percentile") <= F.col("target_pct"))
        .select("rtb_id", "rtb_account_id", *segment_features)
    )
    
    # Collect as Set of Strings per RTB ID
    # We return a DataFrame: rtb_id, segments_set (Array<String>)
    premium_segments_agg = (
        premium_segments_df
        .select("rtb_id", "rtb_account_id", F.struct(segment_features).alias("segment_struct"))
        .withColumn("segment_str", F.col("segment_struct").cast("string"))
        .groupBy("rtb_id", "rtb_account_id")
        .agg(F.collect_set("segment_str").alias("segments"))
    )
    
    return premium_segments_agg, premium_segments_df

# COMMOND -------------------------------------------------------------
# Main Loop: Day-over-Day Similarity

previous_premium_df = None
previous_date = None

# List to collect all daily comparison results (DataFrame)
all_daily_similarities = []
all_feature_counts = []

for i, date_path in enumerate(TRAINING_PATHS):
    current_date = date_path.split("dt=")[-1]
    
    # Get current day's premium segments
    current_premium_df, current_premium_raw = get_premium_segments_for_date(date_path)
    
    # Calculate Feature Distribution efficiently using stack (unpivot)
    # This avoids iterating through features and creating multiple branches in the DAG
    stack_expr_parts = []
    for f in segment_features:
        stack_expr_parts.append(f"'{f}'")
        stack_expr_parts.append(f"CAST({f} AS STRING)")
    
    stack_string = ", ".join(stack_expr_parts)
    stack_expr = f"stack({len(segment_features)}, {stack_string}) as (feature, token)"
    
    cnt_df = (
        current_premium_raw
        .select(F.expr(stack_expr))
        .groupBy("feature", "token")
        .count()
        .withColumn("date", F.lit(current_date))
    )
    all_feature_counts.append(cnt_df)
    
    if i == 0:
        # First day, just store and continue
        previous_premium_df = current_premium_df
        previous_date = current_date
        continue
        
    print(f"Comparing {current_date} vs {previous_date}...")
    
    # Calculate Similarity per RTB ID
    comparison_df = (
        previous_premium_df.withColumnRenamed("segments", "segments_prev")
        .join(
            current_premium_df.withColumnRenamed("segments", "segments_curr"),
            ["rtb_id", "rtb_account_id"],
            "inner"
        )
    )
    
    @F.udf("double")
    def jaccard_similarity(list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return float(intersection) / float(union) if union > 0 else 0.0

    similarity_df = comparison_df.withColumn(
        "similarity", 
        jaccard_similarity(F.col("segments_prev"), F.col("segments_curr"))
    ).select("rtb_id", "rtb_account_id", F.lit(current_date).alias("date"), "similarity")
    
    # Append to list
    all_daily_similarities.append(similarity_df)
    
    # Update previous for next iteration
    previous_premium_df = current_premium_df
    previous_date = current_date

# COMMOND -------------------------------------------------------------
# Union All and Pivot for Matrix View

if all_daily_similarities:
    print("\n=== Generating Day-over-Day Similarity Matrix ===")
    
    # Union all daily comparisons into one DataFrame
    full_similarity_df = all_daily_similarities[0]
    for df in all_daily_similarities[1:]:
        full_similarity_df = full_similarity_df.union(df)
        
    # Pivot: Rows = rtb_id, Cols = date, Values = similarity
    # Use 'avg' aggregator as pivot requires one, but per (rtb_id, date) it's unique anyway
    matrix_df = (
        full_similarity_df
        .groupBy("rtb_id", "rtb_account_id")
        .pivot("date")
        .agg(F.first("similarity")) # or avg, max
        .fillna(-1.0) # Fill missing dates (if any) with 0 or null
    )
    
    # Calculate average stability for sorting
    similarity_cols = [c for c in matrix_df.columns if c != "rtb_id" and c != "rtb_account_id"]
    # Dynamic expression for average across all date columns
    avg_expr = sum(F.col(c) for c in similarity_cols) / len(similarity_cols)
    
    matrix_sorted_df = matrix_df.withColumn("avg_stability", avg_expr).orderBy(F.asc("avg_stability"))
    
    print("Displaying Stability Matrix (Sorted by Lowest Stability):")
    display(matrix_sorted_df.drop("avg_stability"))
else:
    print("No similarity data generated.")

if all_feature_counts:
    print("\n=== Generating Feature Distribution Shift ===")
    
    # Union all counts
    full_counts_df = all_feature_counts[0]
    for df in all_feature_counts[1:]:
        full_counts_df = full_counts_df.union(df)
        
    # Pivot: Rows = (feature, token), Cols = date, Values = count
    dist_matrix = (
        full_counts_df
        .groupBy("feature", "token")
        .pivot("date")
        .sum("count")
        .fillna(0)
    )
    
    print("Displaying Feature Distribution Shift:")
    display(dist_matrix.orderBy("feature", "token"))
