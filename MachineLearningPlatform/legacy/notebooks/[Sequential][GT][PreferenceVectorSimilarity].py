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

# COMMOND -------------------------------------------------------------
# Main Loop: Preference Vector Similarity Calculation

# 1. Calculate Preference Distribution (UNR Share) per RTB
print("Calculating Feature Preference Vectors...")

processed_df = process_top_k_publisher(raw_df_all, top_k_publishers)

# We want to calculate the distribution of UNR across key feature dimensions for each RTB.
# Features to profile: geo, platform, placement_type (high-level strategy indicators)
# We can concat them or treat them as a single vocab.
# "geo=US", "platform=ios", etc.

# Explode or Stack features is hard.
# Better: Concat all features into a single string column "feature_value" (e.g. "geo:US", "platform:ios")
# But raw data has them in separate columns.
# We need to pivot or aggregate.

# Strategy:
# For each feature column F in [geo, platform, placement_type, supply_name]:
#   Group by (rtb_id, F) -> Sum(UNR)
#   Format as "F:value" -> score
#   Union all these into a long table (rtb_id, feature_token, score)

profile_features = ["geo", "platform", "placement_type", "supply_name", "major_os_version"]
long_profile_dfs = []

for feat in profile_features:
    # Aggregation
    feat_df = (
        processed_df
        .groupBy("rtb_id", feat)
        .agg(F.sum("unr").alias("unr_val"))
        .filter(F.col("unr_val") > 0)
        .withColumn("feature_token", F.concat(F.lit(f"{feat}:"), F.col(feat).cast("string")))
        .select("rtb_id", "feature_token", "unr_val")
    )
    long_profile_dfs.append(feat_df)

# Union all
full_profile_df = long_profile_dfs[0]
for df in long_profile_dfs[1:]:
    full_profile_df = full_profile_df.union(df)

# Normalize per RTB to get Probability Distribution (Sum = 1 or L2 norm)
# L2 Norm is better for Cosine Similarity.
# But standard "Preference Vector" usually implies probability (Sum=1).
# Cosine Similarity formula handles normalization, so raw UNR or Probability works.
# However, raw UNR magnitude varies by RTB size. We should normalize to remove size effect.
# P(feature | rtb) = UNR(rtb, feature) / Total_UNR(rtb)

w_rtb = Window.partitionBy("rtb_id")
normalized_profile_df = (
    full_profile_df
    .withColumn("total_rtb_unr", F.sum("unr_val").over(w_rtb))
    .withColumn("score", F.col("unr_val") / F.col("total_rtb_unr"))
    .select("rtb_id", "feature_token", "score")
)

# 2. Distributed Cosine Similarity Calculation
print("Calculating Distributed Cosine Similarity...")

# Cosine(A, B) = Dot(A, B) / (Norm(A) * Norm(B))
# Norm(A) = Sqrt(Sum(score^2))

# A. Calculate Norms per RTB
rtb_norms = (
    normalized_profile_df
    .groupBy("rtb_id")
    .agg(F.sqrt(F.sum(F.pow("score", 2))).alias("norm"))
)

# B. Prepare for Self-Join
# We only need (rtb_id, feature_token, score)
# Filter out tiny scores to reduce join size (optional optimization)
# normalized_profile_df = normalized_profile_df.filter(F.col("score") > 1e-5)

df_vector = normalized_profile_df.select("rtb_id", "feature_token", "score")

# C. Calculate Dot Product via Self-Join on Feature
# This computes Sum(score_A * score_B) for each pair (A, B) that shares a feature
dot_products = (
    df_vector.alias("v1")
    .join(df_vector.alias("v2"), on="feature_token", how="inner")
    .filter(F.col("v1.rtb_id") <= F.col("v2.rtb_id")) # Compute upper triangle + diagonal only
    .groupBy(F.col("v1.rtb_id").alias("rtb_id_A"), F.col("v2.rtb_id").alias("rtb_id_B"))
    .agg(F.sum(F.col("v1.score") * F.col("v2.score")).alias("dot_product"))
)

# D. Join Norms to Calculate Cosine
similarity_df = (
    dot_products
    .join(rtb_norms.alias("n1"), F.col("rtb_id_A") == F.col("n1.rtb_id"))
    .join(rtb_norms.alias("n2"), F.col("rtb_id_B") == F.col("n2.rtb_id"))
    .select(
        "rtb_id_A",
        "rtb_id_B",
        (F.col("dot_product") / (F.col("n1.norm") * F.col("n2.norm"))).alias("cosine_similarity")
    )
)

# Mirror to lower triangle if needed, but for display/search usually upper is enough.
# If we want full matrix for heatmap, we can union the inverse.
similarity_full_df = (
    similarity_df
    .union(
        similarity_df
        .filter(F.col("rtb_id_A") != F.col("rtb_id_B"))
        .select(F.col("rtb_id_B").alias("rtb_id_A"), F.col("rtb_id_A").alias("rtb_id_B"), "cosine_similarity")
    )
)

# 5. Display
print("Displaying Top Strategy-Similar Pairs (Based on Feature Preference):")
display(
    similarity_full_df
    .filter(F.col("rtb_id_A") != F.col("rtb_id_B"))
    .orderBy(F.desc("cosine_similarity"))
)

# Optional: Save
# similarity_df.write.mode("overwrite").option("header", True).csv("data/rtb_preference_similarity.csv")
