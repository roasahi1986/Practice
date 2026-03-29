import sys
import os
from pyspark.sql import SparkSession, functions as F

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
    "s3://exchange/machine_learning/data/42/dt=2025-11-26",
    "s3://exchange/machine_learning/data/42/dt=2025-11-27",
    "s3://exchange/machine_learning/data/42/dt=2025-11-28",
    "s3://exchange/machine_learning/data/42/dt=2025-11-29",
    "s3://exchange/machine_learning/data/42/dt=2025-11-30",
    "s3://exchange/machine_learning/data/42/dt=2025-12-01",
    "s3://exchange/machine_learning/data/42/dt=2025-12-02",
    "s3://exchange/machine_learning/data/42/dt=2025-12-03",
    "s3://exchange/machine_learning/data/42/dt=2025-12-04",
    "s3://exchange/machine_learning/data/42/dt=2025-12-05",
    "s3://exchange/machine_learning/data/42/dt=2025-12-06",
    "s3://exchange/machine_learning/data/42/dt=2025-12-07",
    "s3://exchange/machine_learning/data/42/dt=2025-12-08",
    "s3://exchange/machine_learning/data/42/dt=2025-12-09"
]

RENAME_COLUMNS = {"rtb_connection_id": "rtb_id"}
EXP_BUCKET_CONTROL = "project_v1_control"

# COMMOND -------------------------------------------------------------
# 1. Load Data
print("Loading Data from all paths...")
raw_df = load_and_prepare_raw_data(
    TRAINING_INPUT_DATA_PATH,
    TRAINING_PATHS,
    RENAME_COLUMNS
)

# Filter for Control Group directly from raw_df
control_df = raw_df.filter(F.col("exp_bucket") == EXP_BUCKET_CONTROL)

# Ensure 'dt' column exists or extract it.
print("Columns available:", control_df.columns)

# COMMOND -------------------------------------------------------------
# 2. Compute Metrics Grouped by Date and ID
print("Calculating daily metrics...")

def calculate_daily_metrics(df, group_col):
    """Calculates daily metrics for a given grouping column."""
    return (
        df.groupBy("dt", group_col)
        .agg(
            F.sum("bid_requests").alias("raw_requests"),
            F.sum("unr").alias("total_unr")
        )
        .withColumn("total_requests_scaled", F.col("raw_requests") * 10)
        .withColumn(
            "nr_per_billion",
            F.when(F.col("raw_requests") > 0, F.col("total_unr") / F.col("raw_requests") * 1e9).otherwise(0)
        )
        .select("dt", group_col, "total_requests_scaled", "nr_per_billion")
    )

rtb_id_daily_df = calculate_daily_metrics(control_df, "rtb_id")
rtb_account_daily_df = calculate_daily_metrics(control_df, "rtb_account_id")

# COMMOND -------------------------------------------------------------
# 3. Pivot to Wide Format (Date as Columns)

def pivot_metrics(daily_df, group_col):
    """Pivots daily metrics into wide format with formatted column names."""
    # Pivot requests
    pivoted_requests_df = (
        daily_df.groupBy(group_col)
        .pivot("dt")
        .agg(F.sum("total_requests_scaled"))
    )
    # Rename columns to indicate metric
    for col in pivoted_requests_df.columns:
        if col != group_col:
            pivoted_requests_df = pivoted_requests_df.withColumnRenamed(col, f"{col}_requests")

    # Pivot NR per Billion
    pivoted_nr_df = (
        daily_df.groupBy(group_col)
        .pivot("dt")
        .agg(F.sum("nr_per_billion"))
    )
    for col in pivoted_nr_df.columns:
        if col != group_col:
            pivoted_nr_df = pivoted_nr_df.withColumnRenamed(col, f"{col}_nr_per_billion")
            
    # Join both metrics
    return pivoted_requests_df.join(pivoted_nr_df, group_col, "outer").fillna(0)

print("Pivoting metrics...")
rtb_id_trend_df = pivot_metrics(rtb_id_daily_df, "rtb_id")
rtb_account_trend_df = pivot_metrics(rtb_account_daily_df, "rtb_account_id")

# COMMOND -------------------------------------------------------------
# 4. Sort and Display

def display_top_trend(trend_df, name):
    """Sorts by total volume across all days and displays all results."""
    request_cols = [c for c in trend_df.columns if c.endswith("_requests")]
    # Dynamic sum of all request columns for sorting
    trend_sorted_df = trend_df.withColumn("total_all_days", sum(F.col(c) for c in request_cols))
    
    print(f"Displaying {name} Trend (Sorted by Total Volume):")
    display(trend_sorted_df.orderBy(F.desc("total_all_days")).drop("total_all_days"))

display_top_trend(rtb_id_trend_df, "RTB ID")
display_top_trend(rtb_account_trend_df, "RTB Account ID")
