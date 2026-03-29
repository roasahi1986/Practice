"""
Validation script: Compare legacy vs new pipeline rolling features by publisher.

Compares:
- New: s3://exchange-dev/luxu/project/etl/compute_rolling_win_price_by_publisher
- Legacy: s3://exchange-dev/luxu/realtime_rtb_features/rolling_features/by_pub_app_store_id
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
NEW_PATH = "s3://exchange-dev/luxu/project/etl/compute_rolling_win_price_by_publisher"
LEGACY_PATH = "s3://exchange-dev/luxu/realtime_rtb_features/rolling_features/by_pub_app_store_id"

# Date range to validate
DATE_FROM = "2025-11-20"
DATE_TO = "2025-11-30"  # exclusive

# Column name mappings (new -> legacy)
# New pipeline uses "hourly_win_price_sum_1h", legacy uses "sum_win_price_1h"
METRIC_MAPPINGS = {
    "hourly_win_price_sum_1h": "sum_win_price_1h",
    "hourly_win_count_1h": "sum_win_count_1h",
    "hourly_win_price_sum_3h": "sum_win_price_3h",
    "hourly_win_count_3h": "sum_win_count_3h",
}

# Join keys
JOIN_KEYS = ["date", "hour", "rtb_id", "req_pub_app_store_id"]

# =============================================================================
# Load Data
# =============================================================================
print("=" * 80)
print("Loading datasets...")
print("=" * 80)

print(f"\nNew pipeline path: {NEW_PATH}")
new_df = spark.read.parquet(NEW_PATH)
print("  New pipeline schema:")
new_df.printSchema()
new_count = new_df.count()
print(f"  Total rows: {new_count:,}")

print(f"\nLegacy path: {LEGACY_PATH}")
legacy_df = spark.read.parquet(LEGACY_PATH)
print("  Legacy schema:")
legacy_df.printSchema()
legacy_count = legacy_df.count()
print(f"  Total rows: {legacy_count:,}")

# =============================================================================
# Filter by Date Range
# =============================================================================
print("\n" + "=" * 80)
print(f"Filtering to date range: {DATE_FROM} to {DATE_TO}")
print("=" * 80)

new_df = new_df.filter(
    (F.col("date") >= DATE_FROM) & 
    (F.col("date") < DATE_TO)
)
new_filtered_count = new_df.count()
print(f"  New pipeline rows in range: {new_filtered_count:,}")

legacy_df = legacy_df.filter(
    (F.col("date") >= DATE_FROM) & 
    (F.col("date") < DATE_TO)
)
legacy_filtered_count = legacy_df.count()
print(f"  Legacy rows in range: {legacy_filtered_count:,}")

# =============================================================================
# Rename columns for comparison
# =============================================================================
# Rename new columns to have "new_" prefix
for new_col in METRIC_MAPPINGS.keys():
    new_df = new_df.withColumnRenamed(new_col, f"new_{new_col}")

# Rename legacy columns to have "legacy_" prefix (using new naming convention)
for new_col, legacy_col in METRIC_MAPPINGS.items():
    legacy_df = legacy_df.withColumnRenamed(legacy_col, f"legacy_{new_col}")

# =============================================================================
# Left Join: Legacy -> New
# =============================================================================
print("\n" + "=" * 80)
print("Performing INNER JOIN (legacy -> new)...")
print("=" * 80)

joined_df = legacy_df.join(new_df, on=JOIN_KEYS, how="inner")

# =============================================================================
# Calculate Differences
# =============================================================================
for new_col in METRIC_MAPPINGS.keys():
    new_col_name = f"new_{new_col}"
    legacy_col_name = f"legacy_{new_col}"
    diff_col = f"diff_{new_col}"
    
    joined_df = joined_df.withColumn(
        diff_col,
        F.col(legacy_col_name) - F.coalesce(F.col(new_col_name), F.lit(0))
    )

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "=" * 80)
print("Summary Statistics")
print("=" * 80)

total_rows = joined_df.count()
print(f"\nTotal joined rows: {total_rows:,}")

print("\nExact matches per metric:")
for new_col in METRIC_MAPPINGS.keys():
    diff_col = f"diff_{new_col}"
    exact_match = joined_df.filter(F.col(diff_col) <= 1e-6).count()
    print(f"  {new_col}: {exact_match:,} ({100*exact_match/total_rows:.2f}%)")

# Aggregate stats for differences
print("\nDifference statistics:")
stats_exprs = [F.count("*").alias("count")]
for new_col in METRIC_MAPPINGS.keys():
    diff_col = f"diff_{new_col}"
    stats_exprs.extend([
        F.avg(diff_col).alias(f"avg_{diff_col}"),
        F.max(F.abs(F.col(diff_col))).alias(f"max_abs_{diff_col}"),
    ])

stats_df = joined_df.select(*stats_exprs)
display(stats_df)

# =============================================================================
# Show Mismatched Rows (if any)
# =============================================================================
print("\n" + "=" * 80)
print("Rows with differences (top 20)")
print("=" * 80)

# Filter for rows with non-zero differences in any metric
diff_conditions = [F.col(f"diff_{col}") > 1e-6 for col in METRIC_MAPPINGS.keys()]
combined_condition = diff_conditions[0]
for cond in diff_conditions[1:]:
    combined_condition = combined_condition | cond

mismatch_df = joined_df.filter(combined_condition)
mismatch_count = mismatch_df.count()
print(f"\nTotal mismatched rows: {mismatch_count:,}")

if mismatch_count > 0:
    print("\nSample mismatches:")
    display_cols = JOIN_KEYS.copy()
    for new_col in METRIC_MAPPINGS.keys():
        display_cols.extend([
            f"legacy_{new_col}",
            f"new_{new_col}",
            f"diff_{new_col}",
        ])
    
    display(
        mismatch_df
        .select(display_cols)
        .orderBy(F.abs(F.col("diff_hourly_win_price_sum_1h")).desc())
        .limit(20)
    )
else:
    print("\n✅ All rows match exactly!")
