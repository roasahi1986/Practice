import sys
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import CountVectorizer, MinHashLSH

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
EXP_BUCKET_EXPLORE = ["48"] # Keep for compatibility, though we might focus on rtb_id logic
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
# 1. Load Data
print("Loading Data...")
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
processed_df = process_top_k_publisher(raw_df, top_k_publishers)

numerical_features = []
# Include rtb_account_id to link with cap file, but analysis is on rtb_id
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

# COMMOND -------------------------------------------------------------
# 2. Get Default Cap from Latest Path
latest_path = (
    spark.read.format("binaryFile")
    .load("/FileStore/smart_qps/qps_cap_by_dsp_*.csv")
    .orderBy(F.col("modificationTime").desc())
    .limit(1)
    .select("path")
    .first()["path"]
)
print(f"✅ Latest file path for QPS cap data: {latest_path}")

# Read caps, but instead of user input, we use the 'cap' column directly.
# We need to map caps to rtb_id.
# Assuming the file has rtb_account_id, we map it back to rtb_id via the training data relationship.
# However, the logic requests: "use default cap from latest_path"
# And then "compute premium traffic segments for each rtb_id"
# We'll calculate target_pct per rtb_account_id first.

caps_sdf = (
    spark.read.option("header", True).csv(latest_path)
    .fillna(1.0) # Fill NA with default cap 1.0
    .withColumn("cap", F.col("cap").cast("double"))
    .withColumn("increase_ratio", F.col("cap") - 1)
    .filter(F.col("increase_ratio") >= INCREASE_RATIO_MIN)
    .withColumn("target_pct", F.col("increase_ratio") / F.lit(MULTIPLIER))
    .select("rtb_account_id", "target_pct")
)

# COMMOND -------------------------------------------------------------
# 3. Predict & Determine Premium Segments per rtb_id
# Note: The original script calculates thresholds per rtb_account_id.
# The user asked: "compute premium traffic segments for each rtb_id (not rtb_account_id)".
# This implies the thresholding logic should probably be applied per rtb_id?
# BUT the caps are per rtb_account_id.
# Strategy: Join caps to data on rtb_account_id, but calculate percentile/threshold GROUPED BY rtb_id.
# This treats each rtb_id as an independent entity for thresholding, sharing the account's target_pct.

pred_df = final_training_df.withColumnRenamed("y", "y_pred")

# Join target_pct to pred_df
# Need to ensure rtb_account_id is in pred_df (it should be in all_features)
pred_df_with_target = pred_df.join(F.broadcast(caps_sdf), "rtb_account_id", "inner")

# Calculate Percentile per rtb_id
w_rtb_id = Window.partitionBy("rtb_id")
w_cum_rtb_id = Window.partitionBy("rtb_id").orderBy(F.desc("y_pred"))

pred_df_cum = (
    pred_df_with_target
    .withColumn("rtb_sum_weight", F.sum("weight").over(w_rtb_id))
    .withColumn("cumulative_sum_weight", F.sum("weight").over(w_cum_rtb_id))
    .withColumn("percentile", F.col("cumulative_sum_weight") / F.col("rtb_sum_weight"))
)

# Identify Premium Segments
# Logic: Premium if percentile <= target_pct
# We need to extract the "Traffic Segment" (Feature Combination)
# The segments are defined by params["feature_selected"] minus rtb_id itself (since we group by rtb_id)
segment_features = [f for f in params["feature_selected"] if f != "rtb_id"]

premium_segments_df = (
    pred_df_cum
    .filter(F.col("percentile") <= F.col("target_pct"))
    .select("rtb_id", F.struct(segment_features).alias("segment_struct"))
)

# COMMOND -------------------------------------------------------------
# 4. Calculate Jaccard Similarity using MinHashLSH (Distributed)

# A. Convert Segment Strings to Array of Strings per rtb_id
premium_segments_agg = (
    premium_segments_df
    .withColumn("segment_str", F.col("segment_struct").cast("string"))
    .groupBy("rtb_id")
    .agg(F.collect_set("segment_str").alias("segments"))
)

# LOG: Count RTB IDs with premium segments
total_rtb_with_premium = premium_segments_agg.count()
print(f"🔍 [LOG] RTB IDs with at least one premium segment: {total_rtb_with_premium}")

# Filter out rtb_ids with empty segments to prevent MinHashLSH error
# "Must have at least 1 non zero entry"
premium_segments_agg = premium_segments_agg.filter(F.size(F.col("segments")) > 0)

# LOG: Count after empty filter
count_after_filter = premium_segments_agg.count()
print(f"🔍 [LOG] RTB IDs after filtering empty segments: {count_after_filter}")

# B. Vectorize the sets using CountVectorizer
# This converts the array of strings into a sparse vector
cv = CountVectorizer(inputCol="segments", outputCol="features", minDF=1.0) # Keep all segments
model_cv = cv.fit(premium_segments_agg)
vectorized_df = model_cv.transform(premium_segments_agg).select("rtb_id", "features")

# LOG: Vocabulary size
vocab_size = len(model_cv.vocabulary)
print(f"🔍 [LOG] Total unique premium segments (Vocabulary Size): {vocab_size}")

# Define a UDF to check for non-zero vectors
# MinHashLSH requires at least 1 non-zero entry
@F.udf("boolean")
def has_non_zero(v):
    return v.numNonzeros() > 0

# Filter out empty vectors
vectorized_df = vectorized_df.filter(has_non_zero(F.col("features")))

# Force materialization to ensure filter happens BEFORE MinHashLSH
# Using localCheckpoint to cut lineage and force computation
vectorized_df = vectorized_df.localCheckpoint()

# Cache to avoid recomputing
vectorized_df.cache()
n_rtbs = vectorized_df.count()
print(f"🔍 [LOG] RTB IDs with non-zero feature vectors (Ready for LSH): {n_rtbs}")

if n_rtbs > 600:
    print("⚠️ [WARNING] RTB count > 600. Matrix might be larger than expected.")

# C. Fit MinHashLSH
# NumHashTables trades off accuracy vs speed. 5-10 is usually good for visualization.
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=10)
model_mh = mh.fit(vectorized_df)

# D. Calculate All-Pairs Similarity
# approxSimilarityJoin returns pairs with distance < threshold.
# Jaccard Distance = 1 - Jaccard Similarity.
# To get full matrix, threshold=1.0 (allow all).
# Note: This produces N*N/2 rows. For N=100 -> 5000 rows (tiny). For N=1000 -> 500k rows (manageable).
similarity_df = model_mh.approxSimilarityJoin(
    vectorized_df, vectorized_df, threshold=1.0, distCol="jaccard_distance"
)

# LOG: Result size
sim_count = similarity_df.count()
print(f"🔍 [LOG] Total similarity pairs calculated (similarity > 0): {sim_count}")
expected_max = n_rtbs * n_rtbs
print(f"🔍 [LOG] Max possible pairs (N*N): {expected_max}")

# E. Collect only the Similarity Matrix (N x N floats), which is tiny.
# Select only necessary columns
# edges = (
#     similarity_df
#     .select(
#         F.col("datasetA.rtb_id").alias("id_i"),
#         F.col("datasetB.rtb_id").alias("id_j"),
#         (1.0 - F.col("jaccard_distance")).alias("similarity")
#     )
#     .collect()
# )

# Build Matrix on Driver
# Map rtb_id to index
# rtb_ids_list = sorted(list(set([row.id_i for row in edges] + [row.id_j for row in edges])))
# id_to_idx = {rid: i for i, rid in enumerate(rtb_ids_list)}
# n_matrix = len(rtb_ids_list)

# similarity_matrix = np.zeros((n_matrix, n_matrix))

# for row in edges:
#     i = id_to_idx[row.id_i]
#     j = id_to_idx[row.id_j]
#     similarity_matrix[i, j] = row.similarity
#     similarity_matrix[j, i] = row.similarity # Symmetric

# rtb_ids = rtb_ids_list
# n_rtbs = n_matrix

# Unpersist
vectorized_df.unpersist()

# Display Similarity DataFrame directly (Top matches)
print("Displaying Top Similar Pairs:")
display(
    similarity_df
    .select(
        F.col("datasetA.rtb_id").alias("rtb_id_A"),
        F.col("datasetB.rtb_id").alias("rtb_id_B"),
        (1.0 - F.col("jaccard_distance")).alias("similarity")
    )
    .filter(F.col("rtb_id_A") != F.col("rtb_id_B")) # Remove self-similarity for display
    .orderBy(F.desc("similarity"))
)

