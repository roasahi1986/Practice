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
# Main Loop: Overlap Coefficient Calculation

processed_df = process_top_k_publisher(raw_df_all, top_k_publishers)
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

pred_df = final_training_df.withColumnRenamed("y", "y_pred")
pred_df_with_target = pred_df.join(F.broadcast(caps_sdf), "rtb_account_id", "inner")

w_rtb_id = Window.partitionBy("rtb_account_id")
w_cum_rtb_id = Window.partitionBy("rtb_account_id").orderBy(F.desc("y_pred"))

pred_df_cum = (
    pred_df_with_target
    .withColumn("rtb_sum_weight", F.sum("weight").over(w_rtb_id))
    .withColumn("cumulative_sum_weight", F.sum("weight").over(w_cum_rtb_id))
    .withColumn("percentile", F.col("cumulative_sum_weight") / F.col("rtb_sum_weight"))
)

segment_features = [f for f in params["feature_selected"] if f != "rtb_id"]

premium_segments_df = (
    pred_df_cum
    .filter(F.col("percentile") <= F.col("target_pct"))
    .select("rtb_id", F.struct(segment_features).alias("segment_struct"))
)

premium_segments_agg = (
    premium_segments_df
    .withColumn("segment_str", F.col("segment_struct").cast("string"))
    .groupBy("rtb_id")
    .agg(F.collect_set("segment_str").alias("segments"))
)

# Filter empty
premium_segments_agg = premium_segments_agg.filter(F.size(F.col("segments")) > 0)

# 2. Collect to Driver
print("Collecting premium segments to driver for Overlap Calculation...")
rtb_data = premium_segments_agg.collect()
rtb_map = {row.rtb_id: set(row.segments) for row in rtb_data}
rtb_ids = sorted(rtb_map.keys())
n = len(rtb_ids)

print(f"Calculating Overlap Coefficient for {n} RTB IDs ({n*n} pairs)...")

# 3. Compute Overlap Matrix
# Overlap(A, B) = |A n B| / min(|A|, |B|)
# This matrix is symmetric.

overlap_results = []

# Pre-calculate sizes
sizes = {rid: len(s) for rid, s in rtb_map.items()}

for i in range(n):
    id_a = rtb_ids[i]
    set_a = rtb_map[id_a]
    size_a = sizes[id_a]
    
    for j in range(i + 1, n): # Upper triangle
        id_b = rtb_ids[j]
        set_b = rtb_map[id_b]
        size_b = sizes[id_b]
        
        intersection = len(set_a.intersection(set_b))
        min_len = min(size_a, size_b)
        
        overlap = intersection / min_len if min_len > 0 else 0.0
        
        if overlap > 0:
            overlap_results.append((id_a, id_b, overlap))
            overlap_results.append((id_b, id_a, overlap)) # Symmetric

# Self-similarity is 1.0
for rid in rtb_ids:
    overlap_results.append((rid, rid, 1.0))

# 4. Create DataFrame and Display
print("Creating Result DataFrame...")
similarity_df = spark.createDataFrame(overlap_results, ["rtb_id_A", "rtb_id_B", "overlap_coefficient"])

print("Displaying Top Overlap Pairs:")
display(
    similarity_df
    .filter(F.col("rtb_id_A") != F.col("rtb_id_B"))
    .orderBy(F.desc("overlap_coefficient"))
)

# Optional: Save for visualization
# similarity_df.write.mode("overwrite").option("header", True).csv("data/rtb_overlap_similarity.csv")
