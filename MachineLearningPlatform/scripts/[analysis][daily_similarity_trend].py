"""
Daily Similarity Trend Analysis.

Computes cosine similarity between EACH individual training day and test data.
This validates the temporal decay hypothesis:
- If more recent training days have higher similarity to test data,
  then temporal decay (weighting recent data higher) is meaningful.
- If similarity doesn't increase with recency, decay is pointless.

Usage (in Databricks notebook):
    %run ./scripts/daily_similarity_trend.py

Or copy-paste the code directly into a notebook cell.
"""

from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.getOrCreate()

# Databricks display function - fallback to show() for local execution
try:
    from databricks.sdk.runtime import display
except ImportError:
    display = lambda df: df.show()

# =============================================================================
# Configuration
# =============================================================================
TRAINING_PATHS = [
    "s3://exchange/machine_learning/data/42/dt=2025-11-20",
    "s3://exchange/machine_learning/data/42/dt=2025-11-21",
    "s3://exchange/machine_learning/data/42/dt=2025-11-22",
    "s3://exchange/machine_learning/data/42/dt=2025-11-23",
    "s3://exchange/machine_learning/data/42/dt=2025-11-24",
    "s3://exchange/machine_learning/data/42/dt=2025-11-25",
    "s3://exchange/machine_learning/data/42/dt=2025-11-26"
]

TEST_PATHS = [
    "s3://exchange/machine_learning/data/42/dt=2025-11-27",
    "s3://exchange/machine_learning/data/42/dt=2025-11-28",
    "s3://exchange/machine_learning/data/42/dt=2025-11-29"
]

# Features to group by (aggregation dimensions) - date is used to split, not group
GROUP_BY_FEATURES = [
    "rtb_id",
    "supply_name",
    "pub_app_object_id",
    "geo",
    "placement_type",
    "platform",
]

# Columns to aggregate (sum)
AGG_COLUMNS = ["bid_requests", "multiplied_bid_requests", "unr", "original_bid_requests"]

RENAME_COLUMNS = {"rtb_connection_id": "rtb_id"}

# Top-K publisher selection
TOP_K_PUB_SIZE = 5000
EXP_BUCKET_CONTROL = "project_v1_control"

# Metrics to compute similarity for
SIMILARITY_COLUMNS = ["original_bid_requests", "unr", "unr_per_original_bid_requests"]

# =============================================================================
# Step 1: Load Training Data
# =============================================================================
print("=" * 80)
print("Step 1: Loading training data from S3...")
print("=" * 80)

BASE_PATH = "s3://exchange/machine_learning/data/42"
raw_df = spark.read.option("basePath", BASE_PATH).parquet(*TRAINING_PATHS)

# Rename columns
for old_name, new_name in RENAME_COLUMNS.items():
    if old_name in raw_df.columns:
        raw_df = raw_df.withColumnRenamed(old_name, new_name)

# Extract date from partition column 'dt'
if "dt" in raw_df.columns:
    raw_df = raw_df.withColumn("date", F.to_date(F.col("dt")))
elif "date" not in raw_df.columns:
    raw_df = raw_df.withColumn(
        "date",
        F.to_date(F.regexp_extract(F.input_file_name(), r"dt=(\d{4}-\d{2}-\d{2})", 1))
    )

print(f"Loaded raw data from {len(TRAINING_PATHS)} partitions")

# =============================================================================
# Step 1.5: Extract Top-K Publishers
# =============================================================================
print("\n" + "=" * 80)
print(f"Step 1.5: Extracting top {TOP_K_PUB_SIZE} publishers from control bucket...")
print("=" * 80)

publisher_agg = (
    raw_df.filter(F.col("exp_bucket") == EXP_BUCKET_CONTROL)
    .groupBy("pub_app_object_id")
    .agg(
        F.sum("bid_requests").alias("bid_requests"),
        F.sum(F.coalesce(F.col("multiplied_bid_requests"), F.lit(0))).alias("multiplied_bid_requests"),
        F.sum("unr").alias("unr"),
    )
    .withColumn(
        "original_bid_requests",
        F.col("bid_requests") - F.col("multiplied_bid_requests"),
    )
)

top_k_publishers = (
    publisher_agg
    .orderBy(F.col("unr").desc())
    .limit(TOP_K_PUB_SIZE)
    .select("pub_app_object_id")
    .rdd.flatMap(lambda x: x)
    .collect()
)

print(f"✅ Identified {len(top_k_publishers)} top publishers")

# Add original_bid_requests and filter publishers
raw_df = raw_df.withColumn(
    "original_bid_requests",
    F.col("bid_requests") - F.coalesce(F.col("multiplied_bid_requests"), F.lit(0))
)

raw_df = raw_df.withColumn(
    "pub_app_object_id",
    F.when(
        F.col("pub_app_object_id").isin(top_k_publishers),
        F.col("pub_app_object_id")
    ).otherwise(F.lit("other"))
)

print(f"✅ Mapped non-top-K publishers to 'other'")

# =============================================================================
# Step 2: Load Test Data
# =============================================================================
print("\n" + "=" * 80)
print("Step 2: Loading test data...")
print("=" * 80)

test_raw_df = spark.read.option("basePath", BASE_PATH).parquet(*TEST_PATHS)

for old_name, new_name in RENAME_COLUMNS.items():
    if old_name in test_raw_df.columns:
        test_raw_df = test_raw_df.withColumnRenamed(old_name, new_name)

if "dt" in test_raw_df.columns:
    test_raw_df = test_raw_df.withColumn("date", F.to_date(F.col("dt")))

test_raw_df = test_raw_df.withColumn(
    "original_bid_requests",
    F.col("bid_requests") - F.coalesce(F.col("multiplied_bid_requests"), F.lit(0))
)

test_raw_df = test_raw_df.withColumn(
    "pub_app_object_id",
    F.when(
        F.col("pub_app_object_id").isin(top_k_publishers),
        F.col("pub_app_object_id")
    ).otherwise(F.lit("other"))
)

print(f"Loaded test data from {len(TEST_PATHS)} partitions")

# =============================================================================
# Step 3: Aggregate Test Data (all test days combined)
# =============================================================================
print("\n" + "=" * 80)
print("Step 3: Aggregating test data...")
print("=" * 80)

available_group_cols = [c for c in GROUP_BY_FEATURES if c in test_raw_df.columns]
available_agg_cols = [c for c in AGG_COLUMNS if c in test_raw_df.columns]

test_agg_exprs = [F.sum(F.coalesce(F.col(c), F.lit(0))).alias(c) for c in available_agg_cols]
test_agg_df = test_raw_df.groupBy(available_group_cols).agg(*test_agg_exprs)

# Compute unr_per_original_bid_requests
test_agg_df = test_agg_df.withColumn(
    "unr_per_original_bid_requests",
    F.when(F.col("original_bid_requests") > 0,
           F.col("unr") / F.col("original_bid_requests")
    ).otherwise(F.lit(0.0))
)

test_agg_df = test_agg_df.cache()
test_count = test_agg_df.count()
print(f"Test aggregated rows: {test_count:,}")

# =============================================================================
# Step 4: Define Similarity Functions
# =============================================================================


def add_feature_key(df, key_cols):
    """Create composite feature key from all grouping columns."""
    key_expr = F.concat_ws(
        "||",
        *[F.coalesce(F.col(c).cast("string"), F.lit("__NULL__")) for c in key_cols]
    )
    return df.withColumn("_feature_key", key_expr)


def compute_cosine_similarity(train_df, test_df, metric_col):
    """
    Compute cosine similarity between train and test vectors.
    Each dimension = feature combination, value = metric value.
    """
    joined = train_df.alias("train").join(
        test_df.alias("test"),
        F.col("train._feature_key") == F.col("test._feature_key"),
        "right"
    ).select(
        F.col("test._feature_key").alias("feature_key"),
        F.coalesce(F.col(f"train.{metric_col}"), F.lit(0.0)).alias("train_value"),
        F.col(f"test.{metric_col}").alias("test_value"),
    )
    
    valid_df = joined.filter(
        (F.col("test_value").isNotNull()) &
        ((F.col("train_value") != 0) & (F.col("test_value") != 0))
    )
    
    stats = valid_df.agg(
        F.sum(F.col("train_value") * F.col("test_value")).alias("dot_product"),
        F.sqrt(F.sum(F.col("train_value") ** 2)).alias("train_norm"),
        F.sqrt(F.sum(F.col("test_value") ** 2)).alias("test_norm"),
        F.count("*").alias("matched_count")
    ).collect()[0]
    
    dot_product = stats["dot_product"] or 0
    train_norm = stats["train_norm"] or 0
    test_norm = stats["test_norm"] or 0
    matched_count = stats["matched_count"] or 0
    
    if train_norm > 0 and test_norm > 0:
        cosine_sim = dot_product / (train_norm * test_norm)
    else:
        cosine_sim = 0
    
    return {
        "cosine_similarity": cosine_sim,
        "matched_count": matched_count,
    }


# Add feature key to test data
test_keyed = add_feature_key(test_agg_df, available_group_cols)
test_keyed = test_keyed.cache()

# =============================================================================
# Step 5: Compute Similarity for Each Training Day
# =============================================================================
print("\n" + "=" * 80)
print("Step 5: Computing cosine similarity for each training day vs test...")
print("=" * 80)

# Get distinct training dates
training_dates = sorted([
    row["date"] for row in raw_df.select("date").distinct().collect()
])
print(f"Training dates: {training_dates}")

# Store results for each date and metric
daily_results = []

for train_date in training_dates:
    print(f"\n📅 Processing training date: {train_date}")
    
    # Filter training data for this date
    single_day_df = raw_df.filter(F.col("date") == train_date)
    
    # Aggregate by features (without date)
    single_day_agg = single_day_df.groupBy(available_group_cols).agg(*test_agg_exprs)
    
    # Compute unr_per_original_bid_requests
    single_day_agg = single_day_agg.withColumn(
        "unr_per_original_bid_requests",
        F.when(F.col("original_bid_requests") > 0,
               F.col("unr") / F.col("original_bid_requests")
        ).otherwise(F.lit(0.0))
    )
    
    # Add feature key
    single_day_keyed = add_feature_key(single_day_agg, available_group_cols)
    
    # Compute similarity for each metric
    date_result = {"date": train_date}
    
    for metric in SIMILARITY_COLUMNS:
        result = compute_cosine_similarity(
            single_day_keyed.select("_feature_key", metric),
            test_keyed.select("_feature_key", metric),
            metric
        )
        date_result[f"cos_{metric}"] = result["cosine_similarity"]
        date_result[f"matched_{metric}"] = result["matched_count"]
        print(f"   {metric}: cos_sim = {result['cosine_similarity']:.6f}, matched = {result['matched_count']:,}")
    
    daily_results.append(date_result)

# =============================================================================
# Step 6: Create Results DataFrame and Display
# =============================================================================
print("\n" + "=" * 80)
print("Step 6: Summary Results")
print("=" * 80)

import pandas as pd

results_pdf = pd.DataFrame(daily_results)
results_pdf = results_pdf.sort_values("date")

print("\n📊 DAILY COSINE SIMILARITY TO TEST DATA")
print("=" * 80)
print(results_pdf.to_string(index=False))

# Compute trend statistics
print("\n📈 TREND ANALYSIS")
print("=" * 80)

for metric in SIMILARITY_COLUMNS:
    col = f"cos_{metric}"
    values = results_pdf[col].values
    dates = range(len(values))
    
    # Simple linear regression to check trend
    import numpy as np
    slope, intercept = np.polyfit(dates, values, 1)
    
    # Correlation with day index
    correlation = np.corrcoef(dates, values)[0, 1]
    
    first_day = values[0]
    last_day = values[-1]
    change = last_day - first_day
    change_pct = (change / first_day * 100) if first_day > 0 else 0
    
    print(f"\n{metric}:")
    print(f"   First day similarity: {first_day:.6f}")
    print(f"   Last day similarity:  {last_day:.6f}")
    print(f"   Change: {change:+.6f} ({change_pct:+.2f}%)")
    print(f"   Correlation with recency: {correlation:.4f}")
    print(f"   Trend slope: {slope:.8f} per day")
    
    if correlation > 0.5 and change > 0:
        print(f"   ✅ STRONG POSITIVE TREND - Decay is meaningful!")
    elif correlation > 0 and change > 0:
        print(f"   ⚠️ WEAK POSITIVE TREND - Decay may help slightly")
    elif correlation < -0.5 and change < 0:
        print(f"   ❌ STRONG NEGATIVE TREND - Decay would hurt! Older data is better.")
    elif abs(correlation) < 0.3:
        print(f"   ⚠️ NO CLEAR TREND - Decay may not matter")
    else:
        print(f"   ❌ NEGATIVE TREND - Decay is counterproductive")

# =============================================================================
# Step 7: Visualize with Plotly
# =============================================================================
print("\n" + "=" * 80)
print("Step 7: Generating visualizations...")
print("=" * 80)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplot for each metric
    fig = make_subplots(
        rows=len(SIMILARITY_COLUMNS), cols=1,
        subplot_titles=[f"Cosine Similarity: {m}" for m in SIMILARITY_COLUMNS],
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(SIMILARITY_COLUMNS):
        col = f"cos_{metric}"
        
        # Line plot
        fig.add_trace(
            go.Scatter(
                x=results_pdf["date"].astype(str),
                y=results_pdf[col],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i], width=3),
                marker=dict(size=10),
            ),
            row=i+1, col=1
        )
        
        # Add trend line
        import numpy as np
        dates_numeric = np.arange(len(results_pdf))
        slope, intercept = np.polyfit(dates_numeric, results_pdf[col].values, 1)
        trend_line = intercept + slope * dates_numeric
        
        fig.add_trace(
            go.Scatter(
                x=results_pdf["date"].astype(str),
                y=trend_line,
                mode='lines',
                name=f'{metric} trend',
                line=dict(color=colors[i], width=2, dash='dash'),
                showlegend=False,
            ),
            row=i+1, col=1
        )
        
        fig.update_yaxes(title_text="Cosine Similarity", row=i+1, col=1)
    
    fig.update_xaxes(title_text="Training Date", row=len(SIMILARITY_COLUMNS), col=1)
    
    # Add annotation for the last training day (closest to test)
    last_date_str = results_pdf["date"].astype(str).iloc[-1]
    
    fig.update_layout(
        title="Daily Training Data Similarity to Test Data<br><sub>If similarity increases with recency, decay is meaningful</sub>",
        template='plotly_white',
        height=300 * len(SIMILARITY_COLUMNS),
        showlegend=True,
        annotations=[
            dict(
                x=last_date_str,
                y=1.0,
                xref="x",
                yref="paper",
                text="→ Test Period",
                showarrow=False,
                font=dict(color="red", size=12),
                xanchor="left",
            )
        ]
    )
    
    fig.show()
    print("✅ Displayed: Daily similarity trend chart")
    
    # Create combined chart for comparison
    fig2 = go.Figure()
    
    for i, metric in enumerate(SIMILARITY_COLUMNS):
        col = f"cos_{metric}"
        # Normalize to show relative change from first day
        baseline = results_pdf[col].iloc[0]
        normalized = (results_pdf[col] - baseline) / baseline * 100
        
        fig2.add_trace(
            go.Scatter(
                x=results_pdf["date"].astype(str),
                y=normalized,
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i], width=3),
                marker=dict(size=10),
            )
        )
    
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Day 1 baseline")
    
    fig2.update_layout(
        title="Relative Similarity Change from First Training Day (%)<br><sub>Positive = more similar to test than oldest training day</sub>",
        xaxis_title="Training Date",
        yaxis_title="% Change from Day 1",
        template='plotly_white',
        height=500,
    )
    
    fig2.show()
    print("✅ Displayed: Relative similarity change chart")

except ImportError:
    print("⚠️ Plotly not available. Install with: pip install plotly")

# =============================================================================
# Cleanup
# =============================================================================
test_agg_df.unpersist()
test_keyed.unpersist()

print("\n" + "=" * 80)
print("✅ Daily similarity trend analysis complete!")
print("=" * 80)
print("""
INTERPRETATION:
- If cosine similarity INCREASES as training date approaches test period:
  → Recent data is more predictive → Temporal decay is MEANINGFUL
  
- If cosine similarity is FLAT or RANDOM:
  → All training days are equally predictive → Decay adds no value
  
- If cosine similarity DECREASES toward test period:
  → Older data is more predictive → Decay would HURT performance
""")

