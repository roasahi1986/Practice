"""
Validation script: Compare legacy vs new pipeline hourly aggregates.

This script performs a left join between the new pipeline output and the legacy
notebook output to verify consistency of hourly_win_price_sum and hourly_win_count.
"""

from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.getOrCreate()

# Databricks display function - fallback to show() for local execution
try:
    from databricks.sdk.runtime import display
except ImportError:
    display = lambda df: df.show(truncate=False)

# =============================================================================
# Configuration
# =============================================================================
# New pipeline output (from aggregate_hourly_win_price task)
NEW_PATH = "s3://exchange-dev/luxu/project/etl/aggregate_hourly_win_price"

# Legacy notebook output
LEGACY_PATH = "s3://exchange-dev/luxu/project-backup/etl/aggregate_hourly_win_price"

# Date range to validate (adjust as needed)
DATE_FROM = "2025-11-20"
DATE_TO = "2025-12-01"  # exclusive

# Column names differ between pipelines
NEW_TIME_COL = "time_bucket"      # New pipeline uses time_bucket
LEGACY_TIME_COL = "hour_bucket"   # Legacy uses hour_bucket

# Columns used for joining (after renaming time column)
JOIN_KEYS = [
    "hour_bucket",  # Standardized name after rename
    "rtb_id",
    "supply_name",
    "req_pub_app_store_id",
    "req_impression_type",
    "req_country",
    "req_os",
]

# Columns that may have null values - need special handling for joins
NULLABLE_JOIN_KEYS = ["supply_name"]

# Metrics to compare
METRICS = ["hourly_win_price_sum", "hourly_win_count"]

# Placeholder for null values in join keys (to make null == null work)
NULL_PLACEHOLDER = "__NULL__"

# =============================================================================
# Load Data
# =============================================================================
print("=" * 80)
print("Loading datasets...")
print("=" * 80)

print(f"\nNew pipeline path: {NEW_PATH}")
new_df = spark.read.parquet(NEW_PATH)
# Rename time_bucket -> hour_bucket for consistency
new_df = new_df.withColumnRenamed(NEW_TIME_COL, "hour_bucket")

# Replace nulls with placeholder for nullable join keys
for col_name in NULLABLE_JOIN_KEYS:
    new_df = new_df.withColumn(
        col_name, 
        F.coalesce(F.col(col_name), F.lit(NULL_PLACEHOLDER))
    )
new_count = new_df.count()
print(f"  Total rows: {new_count:,}")

print(f"\nLegacy path: {LEGACY_PATH}")
legacy_df = spark.read.parquet(LEGACY_PATH)
# Legacy already uses hour_bucket, no rename needed
legacy_df = legacy_df.withColumnRenamed(NEW_TIME_COL, "hour_bucket")

# Replace nulls with placeholder for nullable join keys
for col_name in NULLABLE_JOIN_KEYS:
    legacy_df = legacy_df.withColumn(
        col_name, 
        F.coalesce(F.col(col_name), F.lit(NULL_PLACEHOLDER))
    )
legacy_count = legacy_df.count()
print(f"  Total rows: {legacy_count:,}")

# =============================================================================
# Filter by Date Range
# =============================================================================
print("\n" + "=" * 80)
print(f"Filtering to date range: {DATE_FROM} to {DATE_TO}")
print("=" * 80)

new_df = new_df.filter(
    (F.col("hour_bucket") >= DATE_FROM) & 
    (F.col("hour_bucket") < DATE_TO)
)
new_filtered_count = new_df.count()
print(f"  New pipeline rows in range: {new_filtered_count:,}")

legacy_df = legacy_df.filter(
    (F.col("hour_bucket") >= DATE_FROM) & 
    (F.col("hour_bucket") < DATE_TO)
)
legacy_filtered_count = legacy_df.count()
print(f"  Legacy rows in range: {legacy_filtered_count:,}")

# =============================================================================
# Rename columns for clarity before join
# =============================================================================
for metric in METRICS:
    new_df = new_df.withColumnRenamed(metric, f"new_{metric}")
    legacy_df = legacy_df.withColumnRenamed(metric, f"legacy_{metric}")

# =============================================================================
# Left Join: New -> Legacy
# =============================================================================
print("\n" + "=" * 80)
print("Performing LEFT JOIN (new -> legacy)...")
print("=" * 80)

joined_df = new_df.join(legacy_df, on=JOIN_KEYS, how="inner")

# =============================================================================
# Calculate Differences
# =============================================================================
for metric in METRICS:
    new_col = f"new_{metric}"
    legacy_col = f"legacy_{metric}"
    diff_col = f"diff_{metric}"
    pct_diff_col = f"pct_diff_{metric}"
    
    joined_df = joined_df.withColumn(
        diff_col,
        F.col(new_col) - F.coalesce(F.col(legacy_col), F.lit(0))
    )
    joined_df = joined_df.withColumn(
        pct_diff_col,
        F.when(
            F.coalesce(F.col(legacy_col), F.lit(0)) != 0,
            F.abs(F.col(diff_col)) / F.abs(F.col(legacy_col)) * 100
        ).otherwise(
            F.when(F.col(new_col) != 0, F.lit(100.0)).otherwise(F.lit(0.0))
        )
    )

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "=" * 80)
print("Summary Statistics")
print("=" * 80)

total_rows = joined_df.count()
print(f"\nTotal joined rows: {total_rows:,}")

# Count rows with missing legacy data
missing_legacy = joined_df.filter(F.col("legacy_hourly_win_count").isNull()).count()
print(f"Rows missing in legacy: {missing_legacy:,} ({100*missing_legacy/total_rows:.2f}%)")

# Count exact matches
exact_match_price = joined_df.filter(F.col("diff_hourly_win_price_sum") == 0).count()
exact_match_count = joined_df.filter(F.col("diff_hourly_win_count") == 0).count()
print(f"\nExact matches:")
print(f"  hourly_win_price_sum: {exact_match_price:,} ({100*exact_match_price/total_rows:.2f}%)")
print(f"  hourly_win_count: {exact_match_count:,} ({100*exact_match_count/total_rows:.2f}%)")

# Aggregate stats for differences
print("\nDifference statistics:")
stats_df = joined_df.select(
    F.count("*").alias("count"),
    F.avg("diff_hourly_win_price_sum").alias("avg_diff_price"),
    F.max(F.abs(F.col("diff_hourly_win_price_sum"))).alias("max_abs_diff_price"),
    F.avg("pct_diff_hourly_win_price_sum").alias("avg_pct_diff_price"),
    F.avg("diff_hourly_win_count").alias("avg_diff_count"),
    F.max(F.abs(F.col("diff_hourly_win_count"))).alias("max_abs_diff_count"),
    F.avg("pct_diff_hourly_win_count").alias("avg_pct_diff_count"),
)
display(stats_df)

# =============================================================================
# Show Mismatched Rows (if any)
# =============================================================================
print("\n" + "=" * 80)
print("Rows with differences (top 20)")
print("=" * 80)

# Filter for rows with non-zero differences
mismatch_df = joined_df.filter(
    (F.col("diff_hourly_win_price_sum") != 0) | 
    (F.col("diff_hourly_win_count") != 0)
)

mismatch_count = mismatch_df.count()
print(f"\nTotal mismatched rows: {mismatch_count:,}")

if mismatch_count > 0:
    print("\nSample mismatches:")
    display(
        mismatch_df
        .select(
            *JOIN_KEYS,
            "new_hourly_win_price_sum",
            "legacy_hourly_win_price_sum", 
            "diff_hourly_win_price_sum",
            "new_hourly_win_count",
            "legacy_hourly_win_count",
            "diff_hourly_win_count",
        )
        .orderBy(F.abs(F.col("diff_hourly_win_price_sum")).desc())
        .limit(20)
    )
else:
    print("\n✅ All rows match exactly!")

# =============================================================================
# Check for rows in legacy but not in new
# =============================================================================
print("\n" + "=" * 80)
print("Checking for rows in legacy but missing from new...")
print("=" * 80)

# Reload and filter legacy for right anti-join check
legacy_check = spark.read.parquet(LEGACY_PATH).withColumnRenamed(
    NEW_TIME_COL, "hour_bucket"
).filter(
    (F.col(LEGACY_TIME_COL) >= DATE_FROM) & 
    (F.col(LEGACY_TIME_COL) < DATE_TO)
)
# Replace nulls with placeholder for nullable join keys
for col_name in NULLABLE_JOIN_KEYS:
    legacy_check = legacy_check.withColumn(
        col_name, 
        F.coalesce(F.col(col_name), F.lit(NULL_PLACEHOLDER))
    )

new_check = (
    spark.read.parquet(NEW_PATH)
    .withColumnRenamed(NEW_TIME_COL, "hour_bucket")
    .filter(
        (F.col("hour_bucket") >= DATE_FROM) & 
        (F.col("hour_bucket") < DATE_TO)
    )
)
# Replace nulls with placeholder for nullable join keys
for col_name in NULLABLE_JOIN_KEYS:
    new_check = new_check.withColumn(
        col_name, 
        F.coalesce(F.col(col_name), F.lit(NULL_PLACEHOLDER))
    )

missing_from_new = legacy_check.join(new_check, on=JOIN_KEYS, how="left_anti")
missing_count = missing_from_new.count()

print(f"\nRows in legacy but missing from new: {missing_count:,}")
if missing_count > 0:
    print("\nSample missing rows:")
    display(missing_from_new.limit(10))

print("\n" + "=" * 80)
print("Validation Complete!")
print("=" * 80)

