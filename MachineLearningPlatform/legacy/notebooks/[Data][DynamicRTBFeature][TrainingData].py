"""
Training Data with Dynamic RTB Features

This script decorates the training dataset (from project_report) with dynamic RTB features
from S3 tables. The approach is: aggregate first, then decorate with RTB features.
This is more efficient as we join on a smaller, pre-aggregated dataset.

Dynamic RTB Features:
1. sum_win_price_1h, sum_win_count_1h - by country, impression_type, os, supply_name, pub_app_store_id

Total: 2 features * 5 dimensions = 10 dynamic RTB features
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType
import pycountry
from datetime import datetime, timedelta

spark = SparkSession.builder.getOrCreate()

# Databricks display function - fallback to show() for local execution
try:
    from databricks.sdk.runtime import display
except ImportError:
    display = lambda df: df.show()

# =============================================================================
# Configuration
# =============================================================================
# Date range for training data
START_DATE = "2025-11-26"
# END_DATE is automatically set to the next day of START_DATE (exclusive end)
END_DATE = (datetime.strptime(START_DATE, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

# S3 paths for dynamic RTB features
S3_RTB_FEATURES_PATH = "s3://exchange-dev/luxu/realtime_rtb_features/rolling_features"
BY_COUNTRY_PATH = f"{S3_RTB_FEATURES_PATH}/by_country"
BY_IMPRESSION_TYPE_PATH = f"{S3_RTB_FEATURES_PATH}/by_impression_type"
BY_OS_PATH = f"{S3_RTB_FEATURES_PATH}/by_os"
BY_SUPPLY_NAME_PATH = f"{S3_RTB_FEATURES_PATH}/by_supply_name"
BY_APP_STORE_ID_PATH = f"{S3_RTB_FEATURES_PATH}/by_pub_app_store_id"

# Output path
OUTPUT_PATH = f"s3://exchange-dev/luxu/realtime_rtb_features/training_data/{START_DATE}"

# =============================================================================
# Constants
# =============================================================================
EXCLUDED_RTB_ACCOUNT_IDS = [
    "5bc0e10e25c7d7796ebe8fc0",
    "5e8e56f8fc8d4200150b3c16",
    "5f6413c9612b1a0015099993",
    "5f9b336ea3b86126bbdb3aa1",
    "60bdc983a982ad0017edb6f7",
    "63924801f7a526001bc728e2",
    "66e46b7a0f05bd0011e180e6",
    "67055dc40f05bd0011e279ae",
]

EXP_BUCKET_CONTROL = [
    "project_v1_control",
    "55",
]

EXP_BUCKET_EXPLORE = [
    "48",
    "49",
]

SEQ_FALLBACK_THRESHOLD = 1500

# Mapping from placement_type to req_impression_type
PLACEMENT_TO_IMPRESSION_TYPE = {
    "banner": "banner",
    "mrec": "banner",
    "native": "native",
    "appopen": "playable",
    "in_line": "playable",
    "interstitial": "playable",
    "rewarded": "playable",
    "video": "playable",
}

# Build ISO 3166-1 alpha-3 to alpha-2 country code mapping from pycountry
# This maps 3-char country codes (from req_country) to 2-char codes (project_report.country_code)
ALPHA3_TO_ALPHA2 = {
    country.alpha_3: country.alpha_2 
    for country in pycountry.countries
}

# Create a broadcast-friendly UDF for country code mapping
@F.udf(StringType())
def map_alpha3_to_alpha2(alpha3_code):
    """Map ISO 3166-1 alpha-3 to alpha-2 country code."""
    if alpha3_code is None:
        return None
    return ALPHA3_TO_ALPHA2.get(alpha3_code.upper(), alpha3_code)

# =============================================================================
# STEP 1: Load and Aggregate Training Data from Project Report
# =============================================================================
print("=" * 80)
print("STEP 1: Loading and aggregating training data from project_report...")
print("=" * 80)

# Load and aggregate at the final granularity upfront (aggregate first approach)
# This reduces the data volume before joining with RTB features
query = f"""
    SELECT
        DATE_FORMAT(event_time, 'yyyy-MM-dd') AS date,
        DATE_FORMAT(event_time, 'HH') AS hour,
        CASE
            WHEN project_exp_bucket = '55' THEN 'project_v1_control'
            ELSE project_exp_bucket
        END AS exp_bucket,
        rtb_account_id,
        rtb_connection_id,
        pub_app_object_id,
        placement_type,
        platform,
        country_code AS geo,
        supply_name,
        SUM(COALESCE(bid_requests, 0)) AS bid_requests,
        SUM(COALESCE(duplicated_bid_requests, 0)) AS multiplied_bid_requests,
        SUM(COALESCE(unified_net_rev, 0)) AS unr
    FROM hive_prod.lena.project_report
    WHERE event_time >= DATE '{START_DATE}'
      AND event_time <  DATE '{END_DATE}'
      AND rtb_account_id IS NOT NULL
      AND rtb_account_id NOT IN ('{"', '".join(EXCLUDED_RTB_ACCOUNT_IDS)}')
      AND platform IN ('android', 'iOS')
      AND (
          project_exp_bucket IN ('{("', '".join(EXP_BUCKET_EXPLORE))}')
          OR (
              project_exp_bucket IN ('{("', '".join(EXP_BUCKET_CONTROL))}')
              AND auction_timeout_bucket >= {SEQ_FALLBACK_THRESHOLD}
          )
      )
    GROUP BY
        DATE_FORMAT(event_time, 'yyyy-MM-dd'),
        DATE_FORMAT(event_time, 'HH'),
        CASE
            WHEN project_exp_bucket = '55' THEN 'project_v1_control'
            ELSE project_exp_bucket
        END,
        rtb_account_id,
        rtb_connection_id,
        pub_app_object_id,
        placement_type,
        platform,
        country_code,
        supply_name
    HAVING SUM(COALESCE(bid_requests, 0)) > 0
"""
print(f"Query to be executed:\n{query}")

aggregated_df = spark.sql(query)

# Map placement_type to req_impression_type (needed for impression_type feature join)
placement_mapping_expr = F.when(
    F.col("placement_type") == "banner", "banner"
).when(
    F.col("placement_type") == "mrec", "banner"
).when(
    F.col("placement_type") == "native", "native"
).otherwise("playable")  # appopen, in_line, interstitial, rewarded, video

aggregated_df = (
    aggregated_df
    .withColumn("req_impression_type", placement_mapping_expr)
    .withColumn("platform_lower", F.lower(F.col("platform")))
)

# Repartition and cache aggregated data for efficient joins
aggregated_df = (
    aggregated_df
    .repartition(200, "rtb_connection_id", "date", "hour")
    .persist()
)
aggregated_count = aggregated_df.count()
print(f"✅ Training data loaded and aggregated. Row count: {aggregated_count}")

# =============================================================================
# STEP 2: Load Dynamic RTB Features from S3
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: Loading dynamic RTB features from S3...")
print("=" * 80)

# Filter RTB features to only the date we need (reduces data volume significantly)
date_filter = F.col("date") == START_DATE

# Load by_country features
print(f"Loading features from {BY_COUNTRY_PATH}...")
country_features_df = (
    spark.read.parquet(BY_COUNTRY_PATH)
    .filter(date_filter)
    # Convert req_country (alpha-3) to 2-char country code (alpha-2) using UDF
    .withColumn("req_country_2char", map_alpha3_to_alpha2(F.col("req_country")))
    .select(
        "date", "hour", "rtb_id", "req_country_2char",
        F.col("sum_win_count_1h").alias("sum_win_count_1h_country"),
        F.col("sum_win_price_1h").alias("sum_win_price_1h_country")
    )
)

# Load by_impression_type features
print(f"Loading features from {BY_IMPRESSION_TYPE_PATH}...")
impression_type_features_df = (
    spark.read.parquet(BY_IMPRESSION_TYPE_PATH)
    .filter(date_filter)
    .select(
        "date", "hour", "rtb_id", "req_impression_type",
        F.col("sum_win_count_1h").alias("sum_win_count_1h_impression_type"),
        F.col("sum_win_price_1h").alias("sum_win_price_1h_impression_type")
    )
)

# Load by_os features
print(f"Loading features from {BY_OS_PATH}...")
os_features_df = (
    spark.read.parquet(BY_OS_PATH)
    .filter(date_filter)
    # Normalize req_os to lowercase for joining
    .withColumn("req_os_lower", F.lower(F.col("req_os")))
    .select(
        "date", "hour", "rtb_id", "req_os_lower",
        F.col("sum_win_count_1h").alias("sum_win_count_1h_os"),
        F.col("sum_win_price_1h").alias("sum_win_price_1h_os")
    )
)

# Load by_supply_name features
print(f"Loading features from {BY_SUPPLY_NAME_PATH}...")
supply_features_df = (
    spark.read.parquet(BY_SUPPLY_NAME_PATH)
    .filter(date_filter)
    .select(
        "date", "hour", "rtb_id", "supply_name",
        F.col("sum_win_price_1h").alias("sum_win_price_1h_supply"),
        F.col("sum_win_count_1h").alias("sum_win_count_1h_supply")
    )
)

# Load by_app_store_id features
print(f"Loading features from {BY_APP_STORE_ID_PATH}...")
app_store_features_df = (
    spark.read.parquet(BY_APP_STORE_ID_PATH)
    .filter(date_filter)
    .select(
        "date", "hour", "rtb_id", "req_pub_app_store_id",
        F.col("sum_win_price_1h").alias("sum_win_price_1h_app_store"),
        F.col("sum_win_count_1h").alias("sum_win_count_1h_app_store")
    )
)

print("✅ Dynamic RTB features loaded!")

# =============================================================================
# STEP 3: Prepare Feature Tables for Joining
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: Preparing feature tables...")
print("=" * 80)

def remove_duplicates_strict(df, keys, feature_name):
    """
    Check for duplicates on given keys. If found, report count and remove ALL 
    rows associated with those duplicate keys (strict removal).
    """
    dup_counts = df.groupBy(keys).count().filter("count > 1")
    # Trigger action to check count
    dup_count_val = dup_counts.count()
    
    if dup_count_val > 0:
        print(f"⚠️ Found {dup_count_val} duplicate keys in {feature_name} features! Removing ALL rows with these keys...")
        # left_anti join removes rows that match keys in dup_counts
        return df.join(dup_counts, on=keys, how="left_anti")
    
    return df

# Deduplicate feature tables by taking max value per (date, hour, rtb_id, dimension)

print("Preparing country features...")
# Check for duplicates and remove strictly
country_features_clean = remove_duplicates_strict(
    country_features_df, 
    ["date", "hour", "rtb_id", "req_country_2char"], 
    "country"
)

country_features_join = (
    country_features_clean
    .select(
        F.col("date").alias("c_date"),
        F.col("hour").alias("c_hour"),
        F.col("rtb_id").alias("c_rtb_id"),
        F.col("req_country_2char").alias("c_geo"),
        "sum_win_count_1h_country",
        "sum_win_price_1h_country"
    )
    .cache()
)

print("Preparing impression_type features...")
# Check for duplicates and remove strictly
impression_type_features_clean = remove_duplicates_strict(
    impression_type_features_df,
    ["date", "hour", "rtb_id", "req_impression_type"],
    "impression_type"
)

impression_type_features_join = (
    impression_type_features_clean
    .select(
        F.col("date").alias("i_date"),
        F.col("hour").alias("i_hour"),
        F.col("rtb_id").alias("i_rtb_id"),
        F.col("req_impression_type").alias("i_impression_type"),
        "sum_win_count_1h_impression_type",
        "sum_win_price_1h_impression_type"
    )
    .cache()
)

print("Preparing os features...")
# Check for duplicates and remove strictly
os_features_clean = remove_duplicates_strict(
    os_features_df,
    ["date", "hour", "rtb_id", "req_os_lower"],
    "os"
)

os_features_join = (
    os_features_clean
    .select(
        F.col("date").alias("o_date"),
        F.col("hour").alias("o_hour"),
        F.col("rtb_id").alias("o_rtb_id"),
        F.col("req_os_lower").alias("o_platform_lower"),
        "sum_win_count_1h_os",
        "sum_win_price_1h_os"
    )
    .cache()
)

print("Preparing supply_name features...")
# Check for duplicates and remove strictly
supply_features_clean = remove_duplicates_strict(
    supply_features_df,
    ["date", "hour", "rtb_id", "supply_name"],
    "supply_name"
)

supply_features_join = (
    supply_features_clean
    .select(
        F.col("date").alias("s_date"),
        F.col("hour").alias("s_hour"),
        F.col("rtb_id").alias("s_rtb_id"),
        F.col("supply_name").alias("s_supply_name"),
        "sum_win_price_1h_supply",
        "sum_win_count_1h_supply"
    )
    .cache()
)

print("Preparing app_store features...")
# Check for duplicates and remove strictly
app_store_features_clean = remove_duplicates_strict(
    app_store_features_df,
    ["date", "hour", "rtb_id", "req_pub_app_store_id"],
    "app_store"
)

app_store_features_join = (
    app_store_features_clean
    .select(
        F.col("date").alias("a_date"),
        F.col("hour").alias("a_hour"),
        F.col("rtb_id").alias("a_rtb_id"),
        F.col("req_pub_app_store_id").alias("a_app_store_id"),
        "sum_win_price_1h_app_store",
        "sum_win_count_1h_app_store"
    )
    .cache()
)

print("✅ Feature tables prepared!")

# =============================================================================
# STEP 4: Join Aggregated Data with Dynamic RTB Features
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: Joining aggregated data with dynamic RTB features...")
print("=" * 80)

# Use broadcast hints since feature tables are checkpointed and smaller
# Joining on pre-aggregated data is more efficient than decorate-then-aggregate

print("Joining features to aggregated training data...")

# Join with country features
final_df = (
    aggregated_df
    .join(
        F.broadcast(country_features_join),
        (F.col("date") == F.col("c_date")) &
        (F.col("hour") == F.col("c_hour")) &
        (F.col("rtb_connection_id") == F.col("c_rtb_id")) &
        (F.col("geo") == F.col("c_geo")),
        how="left"
    )
    .drop("c_date", "c_hour", "c_rtb_id", "c_geo")
)

# Join with impression_type features
final_df = (
    final_df
    .join(
        F.broadcast(impression_type_features_join),
        (F.col("date") == F.col("i_date")) &
        (F.col("hour") == F.col("i_hour")) &
        (F.col("rtb_connection_id") == F.col("i_rtb_id")) &
        (F.col("req_impression_type") == F.col("i_impression_type")),
        how="left"
    )
    .drop("i_date", "i_hour", "i_rtb_id", "i_impression_type")
)

# Join with os features
final_df = (
    final_df
    .join(
        F.broadcast(os_features_join),
        (F.col("date") == F.col("o_date")) &
        (F.col("hour") == F.col("o_hour")) &
        (F.col("rtb_connection_id") == F.col("o_rtb_id")) &
        (F.col("platform_lower") == F.col("o_platform_lower")),
        how="left"
    )
    .drop("o_date", "o_hour", "o_rtb_id", "o_platform_lower")
)

# Join with supply_name features
final_df = (
    final_df
    .join(
        F.broadcast(supply_features_join),
        (F.col("date") == F.col("s_date")) &
        (F.col("hour") == F.col("s_hour")) &
        (F.col("rtb_connection_id") == F.col("s_rtb_id")) &
        (F.col("supply_name") == F.col("s_supply_name")),
        how="left"
    )
    .drop("s_date", "s_hour", "s_rtb_id", "s_supply_name")
)

# Join with app_store features
final_df = (
    final_df
    .join(
        F.broadcast(app_store_features_join),
        (F.col("date") == F.col("a_date")) &
        (F.col("hour") == F.col("a_hour")) &
        (F.col("rtb_connection_id") == F.col("a_rtb_id")) &
        (F.col("pub_app_object_id") == F.col("a_app_store_id")),
        how="left"
    )
    .drop("a_date", "a_hour", "a_rtb_id", "a_app_store_id")
)

# Fill null feature values with 0 (for cases where no RTB feature data exists)
rtb_feature_cols = [
    "sum_win_price_1h_country",
    "sum_win_count_1h_country",
    "sum_win_price_1h_impression_type",
    "sum_win_count_1h_impression_type",
    "sum_win_price_1h_os",
    "sum_win_count_1h_os",
    "sum_win_price_1h_supply",
    "sum_win_count_1h_supply",
    "sum_win_price_1h_app_store",
    "sum_win_count_1h_app_store",
]
for col_name in rtb_feature_cols:
    final_df = final_df.withColumn(
        col_name, 
        F.coalesce(F.col(col_name), F.lit(0.0))
    )

# Drop intermediate columns not needed in final output
final_df = final_df.drop("req_impression_type", "platform_lower")

# Unpersist aggregated_df as we no longer need it
aggregated_df.unpersist()

print("✅ Training data decorated with dynamic RTB features!")

# =============================================================================
# STEP 5: Save to S3
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: Saving training data to S3...")
print("=" * 80)

# Coalesce to reasonable number of output files to avoid small file problem
# Repartition by key columns to optimize write and downstream reads
final_df.coalesce(100).write.mode("overwrite").parquet(OUTPUT_PATH)
print(f"✅ Training data saved to {OUTPUT_PATH}")

# =============================================================================
# Display Sample Results
# =============================================================================
print("\n" + "=" * 80)
print("Sample Results")
print("=" * 80)

sample_df = spark.read.parquet(OUTPUT_PATH).limit(20)
display(sample_df)

# Show feature statistics
print("\n" + "=" * 80)
print("Dynamic RTB Feature Statistics")
print("=" * 80)

output_df = spark.read.parquet(OUTPUT_PATH)
for col_name in rtb_feature_cols:
    print(f"\n{col_name}:")
    output_df.select(
        F.min(col_name).alias("min"),
        F.max(col_name).alias("max"),
        F.avg(col_name).alias("avg"),
        F.stddev(col_name).alias("stddev")
    ).show()

print("\n" + "=" * 80)
print("✅ Training data generation complete!")
print("=" * 80)
print(f"\nOutput path: {OUTPUT_PATH}")
print(f"\nDynamic RTB Features added (aggregate-first approach):")
print("  - sum_win_price_1h_country: sum win price (1h) by country")
print("  - sum_win_count_1h_country: sum win count (1h) by country")
print("  - sum_win_price_1h_impression_type: sum win price (1h) by impression type")
print("  - sum_win_count_1h_impression_type: sum win count (1h) by impression type")
print("  - sum_win_price_1h_os: sum win price (1h) by OS")
print("  - sum_win_count_1h_os: sum win count (1h) by OS")
print("  - sum_win_price_1h_supply: sum win price (1h) by supply name")
print("  - sum_win_count_1h_supply: sum win count (1h) by supply name")
print("  - sum_win_price_1h_app_store: sum win price (1h) by app store id")
print("  - sum_win_count_1h_app_store: sum win count (1h) by app store id")
