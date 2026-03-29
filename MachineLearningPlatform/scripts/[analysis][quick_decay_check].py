"""
Quick standalone script to check temporal decay on training data.

Reads data from S3, aggregates by selected features, applies temporal decay,
and compares similarity to test data using cosine similarity.

Key features:
- Groups raw data by selected features to reduce data volume
- Computes cosine similarity between training and test vectors
- Each vector dimension = unique feature combination, value = metric

Usage (in Databricks notebook):
    %run ./scripts/quick_decay_check.py

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

# Features to group by (aggregation dimensions)
GROUP_BY_FEATURES = [
    "date",
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

# Decay rates for different metrics
# Using different rates allows unr_per_original_bid_requests to change after decay
DECAY_RATES = {
    "original_bid_requests": 0.9,  # Bid requests decay faster (older = less relevant)
    "unr": 0.95,                   # Revenue decays slower (still valuable signal)
}

# Top-K publisher selection
TOP_K_PUB_SIZE = 5000
EXP_BUCKET_CONTROL = "project_v1_control"

# Columns to apply decay to (after computing original_bid_requests)
DECAY_COLUMNS = ["original_bid_requests", "unr"]

# Columns for cosine similarity comparison (includes derived metrics)
SIMILARITY_COLUMNS = ["original_bid_requests", "unr", "unr_per_original_bid_requests"]

# Number of bins for histogram (more bins = smoother distribution)
NUM_BINS = 100

# =============================================================================
# Step 1: Load Data
# =============================================================================
print("=" * 80)
print("Step 1: Loading data from S3...")
print("=" * 80)

# Use basePath option to correctly handle partitioned data
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
    # Fallback: extract from input_file_name
    raw_df = raw_df.withColumn(
        "date",
        F.to_date(F.regexp_extract(F.input_file_name(), r"dt=(\d{4}-\d{2}-\d{2})", 1))
    )

print(f"Loaded raw data from {len(TRAINING_PATHS)} partitions")
print("\nRaw schema (sample columns):")
for col in raw_df.columns[:15]:
    print(f"  - {col}")
if len(raw_df.columns) > 15:
    print(f"  ... and {len(raw_df.columns) - 15} more columns")

# =============================================================================
# Step 1.5: Extract Top-K Publishers
# =============================================================================
print("\n" + "=" * 80)
print(f"Step 1.5: Extracting top {TOP_K_PUB_SIZE} publishers from control bucket...")
print("=" * 80)

# Aggregate publisher performance from control group
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

# Get top-K publishers ordered by UNR
top_k_publishers = (
    publisher_agg
    .orderBy(F.col("unr").desc())
    .limit(TOP_K_PUB_SIZE)
    .select("pub_app_object_id")
    .rdd.flatMap(lambda x: x)
    .collect()
)

print(f"✅ Identified {len(top_k_publishers)} top publishers")
print(f"   Sample: {top_k_publishers[:5]}...")

raw_df = raw_df.withColumn(
    "original_bid_requests",
    F.col("bid_requests") - F.coalesce(F.col("multiplied_bid_requests"), F.lit(0))
)

# Filter raw_df to only include top-K publishers (map others to "other")
raw_df = raw_df.withColumn(
    "pub_app_object_id",
    F.when(
        F.col("pub_app_object_id").isin(top_k_publishers),
        F.col("pub_app_object_id")
    ).otherwise(F.lit("other"))
)

print(f"✅ Mapped non-top-K publishers to 'other'")

# =============================================================================
# Step 2: Aggregate by Selected Features
# =============================================================================
print("\n" + "=" * 80)
print("Step 2: Aggregating by selected features...")
print("=" * 80)
print(f"Group by: {GROUP_BY_FEATURES}")
print(f"Aggregate (sum): {AGG_COLUMNS}")

# Filter to only columns we need
available_group_cols = [c for c in GROUP_BY_FEATURES if c in raw_df.columns]
available_agg_cols = [c for c in AGG_COLUMNS if c in raw_df.columns]

print(f"\nAvailable group columns: {available_group_cols}")
print(f"Available agg columns: {available_agg_cols}")

# Build aggregation expressions
agg_exprs = [F.sum(F.coalesce(F.col(c), F.lit(0))).alias(c) for c in available_agg_cols]

# Aggregate
agg_df = raw_df.groupBy(available_group_cols).agg(*agg_exprs)

# =============================================================================
# Step 3: Apply Temporal Decay
# =============================================================================
print("\n" + "=" * 80)
print("Step 3: Applying temporal decay...")
print("=" * 80)

# Find the max date (most recent = no decay)
max_date = agg_df.agg(F.max("date")).collect()[0][0]
print(f"Max date (no decay): {max_date}")

# Compute days from max date
agg_df = agg_df.withColumn(
    "days_from_max",
    F.datediff(F.lit(max_date), F.col("date"))
)

# Apply decay: value * (decay_rate ^ days_from_max)
# Each column can have its own decay rate
for col_name in DECAY_COLUMNS:
    if col_name in agg_df.columns:
        decay_rate = DECAY_RATES.get(col_name, 0.9)  # Default to 0.9 if not specified
        decayed_col_name = f"decayed_{col_name}"
        agg_df = agg_df.withColumn(
            decayed_col_name,
            F.col(col_name) * F.pow(F.lit(decay_rate), F.col("days_from_max"))
        )
        print(f"Created: {decayed_col_name} = {col_name} * {decay_rate}^days_from_max")

# Compute derived metrics: unr_per_original_bid_requests
# For original values
agg_df = agg_df.withColumn(
    "unr_per_original_bid_requests",
    F.when(F.col("original_bid_requests") > 0,
           F.col("unr") / F.col("original_bid_requests")
    ).otherwise(F.lit(0.0))
)
print("Created: unr_per_original_bid_requests = unr / original_bid_requests")

# For decayed values
agg_df = agg_df.withColumn(
    "decayed_unr_per_original_bid_requests",
    F.when(F.col("decayed_original_bid_requests") > 0,
           F.col("decayed_unr") / F.col("decayed_original_bid_requests")
    ).otherwise(F.lit(0.0))
)
print("Created: decayed_unr_per_original_bid_requests = decayed_unr / decayed_original_bid_requests")

# Cache after all transformations are done
agg_df = agg_df.cache()
agg_count = agg_df.count()
print(f"\nAggregated rows: {agg_count:,}")

# Show sample with decay applied
print("\nSample data with decay:")
display(
    agg_df.select(
        "date", "days_from_max", 
        *[c for c in agg_df.columns if c in DECAY_COLUMNS or c.startswith("decayed_")]
    ).limit(20)
)

# =============================================================================
# Step 4: Compute Histogram Bins in Spark
# =============================================================================
print("\n" + "=" * 80)
print("Step 4: Computing histogram bins in Spark...")
print("=" * 80)


def compute_histogram_bins_in_spark(df, column_name, num_bins=50, use_log_scale=True):
    """
    Compute histogram bin counts in Spark without pulling raw data.
    
    Args:
        df: Spark DataFrame
        column_name: Column to compute histogram for
        num_bins: Number of bins
        use_log_scale: Whether to use log scale for binning
        
    Returns:
        pandas DataFrame with bin_center and count columns
    """
    # Filter out nulls and non-positive values (for log scale)
    if use_log_scale:
        filtered_df = df.filter((F.col(column_name).isNotNull()) & (F.col(column_name) > 0))
        work_col = f"_log_{column_name}"
        filtered_df = filtered_df.withColumn(work_col, F.log10(F.col(column_name)))
    else:
        filtered_df = df.filter(F.col(column_name).isNotNull())
        work_col = column_name
    
    # Get min and max
    stats = filtered_df.agg(
        F.min(work_col).alias("min_val"),
        F.max(work_col).alias("max_val"),
        F.count(work_col).alias("total_count")
    ).collect()[0]
    
    min_val, max_val, total_count = stats["min_val"], stats["max_val"], stats["total_count"]
    
    if min_val is None or max_val is None or total_count == 0:
        return None
    
    # Compute bin width
    bin_width = (max_val - min_val) / num_bins
    
    if bin_width == 0:
        # All values are the same
        bin_width = 1
    
    # Assign each value to a bin
    binned_df = filtered_df.withColumn(
        "_bin_idx",
        F.floor((F.col(work_col) - F.lit(min_val)) / F.lit(bin_width)).cast("int")
    )
    
    # Clamp to valid range
    binned_df = binned_df.withColumn(
        "_bin_idx",
        F.when(F.col("_bin_idx") >= num_bins, num_bins - 1).otherwise(F.col("_bin_idx"))
    )
    binned_df = binned_df.withColumn(
        "_bin_idx",
        F.when(F.col("_bin_idx") < 0, 0).otherwise(F.col("_bin_idx"))
    )
    
    # Count by bin
    bin_counts = binned_df.groupBy("_bin_idx").agg(
        F.count("*").alias("count")
    ).orderBy("_bin_idx").collect()
    
    # Convert to pandas
    import pandas as pd
    
    result = []
    for row in bin_counts:
        bin_idx = row["_bin_idx"]
        count = row["count"]
        # Compute bin center
        if use_log_scale:
            bin_center = 10 ** (min_val + (bin_idx + 0.5) * bin_width)
        else:
            bin_center = min_val + (bin_idx + 0.5) * bin_width
        result.append({"bin_center": bin_center, "count": count})
    
    pdf = pd.DataFrame(result)
    # Normalize to probability (proportion of total)
    # This ensures values sum to 1 and are always <= 1
    pdf["probability"] = pdf["count"] / pdf["count"].sum()
    
    return pdf


# Compute histograms for original and decayed columns
histogram_data = {}

for col_name in DECAY_COLUMNS:
    if col_name in agg_df.columns:
        print(f"\nComputing histogram for: {col_name}")
        histogram_data[col_name] = compute_histogram_bins_in_spark(
            agg_df, col_name, NUM_BINS, use_log_scale=True
        )
        
        decayed_col = f"decayed_{col_name}"
        if decayed_col in agg_df.columns:
            print(f"Computing histogram for: {decayed_col}")
            histogram_data[decayed_col] = compute_histogram_bins_in_spark(
                agg_df, decayed_col, NUM_BINS, use_log_scale=True
            )

print(f"\nComputed {len(histogram_data)} histograms")

# =============================================================================
# Step 4.5: Load Test Data and Compute Histograms
# =============================================================================
print("\n" + "=" * 80)
print("Step 4.5: Loading test data for comparison...")
print("=" * 80)

# Load test data
test_raw_df = spark.read.option("basePath", BASE_PATH).parquet(*TEST_PATHS)

# Rename columns
for old_name, new_name in RENAME_COLUMNS.items():
    if old_name in test_raw_df.columns:
        test_raw_df = test_raw_df.withColumnRenamed(old_name, new_name)

# Extract date
if "dt" in test_raw_df.columns:
    test_raw_df = test_raw_df.withColumn("date", F.to_date(F.col("dt")))

# Compute original_bid_requests on test_raw_df before aggregation
test_raw_df = test_raw_df.withColumn(
    "original_bid_requests",
    F.col("bid_requests") - F.coalesce(F.col("multiplied_bid_requests"), F.lit(0))
)

# Apply same top-K publisher filtering
test_raw_df = test_raw_df.withColumn(
    "pub_app_object_id",
    F.when(
        F.col("pub_app_object_id").isin(top_k_publishers),
        F.col("pub_app_object_id")
    ).otherwise(F.lit("other"))
)

print(f"Loaded test data from {len(TEST_PATHS)} partitions")

# Aggregate test data by same feature dimensions (excluding date)
test_group_cols = [c for c in available_group_cols if c != "date"]
test_agg_exprs = [F.sum(F.coalesce(F.col(c), F.lit(0))).alias(c) for c in available_agg_cols]

test_agg_df = test_raw_df.groupBy(test_group_cols).agg(*test_agg_exprs)

# Compute unr_per_original_bid_requests for test
test_agg_df = test_agg_df.withColumn(
    "unr_per_original_bid_requests",
    F.when(F.col("original_bid_requests") > 0,
           F.col("unr") / F.col("original_bid_requests")
    ).otherwise(F.lit(0.0))
)

test_agg_df = test_agg_df.cache()
test_count = test_agg_df.count()
print(f"Test aggregated rows: {test_count:,}")

# Also aggregate training without date for fair comparison
# For unr_per_original_bid_requests, we compute weighted average:
#   weighted_avg = SUM(ratio * weight) / SUM(weight), where weight = original_bid_requests
# This simplifies to: SUM(unr) / SUM(original_bid_requests)
train_nodecay_df = agg_df.groupBy(test_group_cols).agg(
    F.sum("original_bid_requests").alias("original_bid_requests"),
    F.sum("unr").alias("unr"),
)
# unr_per_original_bid_requests = weighted average of daily ratios
# = SUM(unr) / SUM(original_bid_requests) when weighted by original_bid_requests
train_nodecay_df = train_nodecay_df.withColumn(
    "unr_per_original_bid_requests",
    F.when(F.col("original_bid_requests") > 0,
           F.col("unr") / F.col("original_bid_requests")
    ).otherwise(F.lit(0.0))
)

train_decayed_df = agg_df.groupBy(test_group_cols).agg(
    F.sum("decayed_original_bid_requests").alias("original_bid_requests"),
    F.sum("decayed_unr").alias("unr"),
)
# Same weighted average for decayed values
train_decayed_df = train_decayed_df.withColumn(
    "unr_per_original_bid_requests",
    F.when(F.col("original_bid_requests") > 0,
           F.col("unr") / F.col("original_bid_requests")
    ).otherwise(F.lit(0.0))
)

train_nodecay_df = train_nodecay_df.cache()
train_decayed_df = train_decayed_df.cache()

print(f"Training (no decay) aggregated rows: {train_nodecay_df.count():,}")
print(f"Training (decayed) aggregated rows: {train_decayed_df.count():,}")

# =============================================================================
# Step 4.6: Compute Cosine Similarity (Vector-based Comparison)
# =============================================================================
print("\n" + "=" * 80)
print("Step 4.6: Computing cosine similarity by feature combinations...")
print("=" * 80)

# Create feature key for joining
FEATURE_KEY_COLS = test_group_cols
print(f"Feature dimensions: {FEATURE_KEY_COLS}")


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
    
    Returns: cosine similarity (0 to 1, higher = more similar)
    """
    # Right join to keep all test rows - we care about test coverage
    joined = train_df.alias("train").join(
        test_df.alias("test"),
        F.col("train._feature_key") == F.col("test._feature_key"),
        "right"  # Keep all test rows
    ).select(
        F.col("test._feature_key").alias("feature_key"),
        F.coalesce(F.col(f"train.{metric_col}"), F.lit(0.0)).alias("train_value"),
        F.col(f"test.{metric_col}").alias("test_value"),
    )
    
    # Filter nulls and rows where BOTH values are 0 (no information)
    # Keep rows where at least one side has a non-zero value
    valid_df = joined.filter(
        (F.col("test_value").isNotNull()) &
        ((F.col("train_value") != 0) & (F.col("test_value") != 0))
    )
    
    # Compute cosine similarity: (A · B) / (||A|| × ||B||)
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


# Add feature keys
train_nodecay_keyed = add_feature_key(train_nodecay_df, FEATURE_KEY_COLS)
train_decayed_keyed = add_feature_key(train_decayed_df, FEATURE_KEY_COLS)
test_keyed = add_feature_key(test_agg_df, FEATURE_KEY_COLS)

# Count unique keys
train_keys = train_nodecay_keyed.select("_feature_key").distinct().count()
test_keys = test_keyed.select("_feature_key").distinct().count()
print(f"\nUnique feature combinations:")
print(f"   Training: {train_keys:,}")
print(f"   Test:     {test_keys:,}")

# Compute cosine similarity for each metric
comparison_results = {}

for col_name in SIMILARITY_COLUMNS:
    print(f"\n📊 Computing cosine similarity for: {col_name}")
    
    # Original training vs Test
    orig_result = compute_cosine_similarity(
        train_nodecay_keyed.select("_feature_key", col_name),
        test_keyed.select("_feature_key", col_name),
        col_name
    )
    
    # Decayed training vs Test
    decay_result = compute_cosine_similarity(
        train_decayed_keyed.select("_feature_key", col_name),
        test_keyed.select("_feature_key", col_name),
        col_name
    )
    
    improvement = decay_result["cosine_similarity"] - orig_result["cosine_similarity"]
    improvement_pct = improvement / orig_result["cosine_similarity"] * 100 if orig_result["cosine_similarity"] > 0 else 0
    
    comparison_results[col_name] = {
        "cos_original_vs_test": orig_result["cosine_similarity"],
        "cos_decayed_vs_test": decay_result["cosine_similarity"],
        "matched_count": orig_result["matched_count"],
        "improvement": improvement,
        "improvement_pct": improvement_pct,
    }
    
    print(f"   Matched feature combinations: {orig_result['matched_count']:,}")
    print(f"   Cosine Similarity (Original → Test): {orig_result['cosine_similarity']:.6f}")
    print(f"   Cosine Similarity (Decayed → Test):  {decay_result['cosine_similarity']:.6f}")
    if decay_result["cosine_similarity"] > orig_result["cosine_similarity"]:
        print(f"   ✅ Decay HELPS! Improvement: {improvement_pct:+.4f}%")
    else:
        print(f"   ❌ Decay does NOT help: {improvement_pct:.4f}%")

# Print summary
print("\n" + "=" * 80)
print("📈 COSINE SIMILARITY SUMMARY")
print("=" * 80)
print(f"\nVector: each dimension = feature combination, value = metric")
if comparison_results:
    first_metric = list(comparison_results.keys())[0]
    print(f"Matched {comparison_results[first_metric]['matched_count']:,} feature combinations\n")
else:
    print("No metrics computed.\n")

print(f"{'Metric':<25} {'Cos(Orig→Test)':<18} {'Cos(Decay→Test)':<18} {'Improvement':<12}")
print("-" * 75)
for col_name, r in comparison_results.items():
    imp_str = f"{r['improvement_pct']:+.4f}%" if r['improvement'] > 0 else f"{r['improvement_pct']:.4f}%"
    print(f"{col_name:<25} {r['cos_original_vs_test']:<18.6f} {r['cos_decayed_vs_test']:<18.6f} {imp_str:<12}")
print("-" * 75)
print("Note: Higher cosine similarity = more similar. Positive improvement = decay helps.")

# =============================================================================
# Step 5: Aggregate by Date for Time Series
# =============================================================================
print("\n" + "=" * 80)
print("Step 5: Aggregating by date for time series...")
print("=" * 80)

# Aggregate by date to see decay effect over time
agg_by_date = agg_df.groupBy("date").agg(
    F.sum("original_bid_requests").alias("total_original_bid_requests"),
    F.sum("decayed_original_bid_requests").alias("total_decayed_original_bid_requests"),
    F.sum("unr").alias("total_unr") if "unr" in agg_df.columns else F.lit(0).alias("total_unr"),
    F.sum("decayed_unr").alias("total_decayed_unr") if "decayed_unr" in agg_df.columns else F.lit(0).alias("total_decayed_unr"),
).orderBy("date")

print("\nAggregated by date:")
display(agg_by_date)

# Small aggregation - safe to collect
agg_by_date_pdf = agg_by_date.toPandas()

# =============================================================================
# Step 6: Visualize with Plotly
# =============================================================================
print("\n" + "=" * 80)
print("Step 6: Generating Plotly visualizations...")
print("=" * 80)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create histogram comparison for original_bid_requests
    # Using step lines instead of bars - works much better with log-scale x-axis
    if "original_bid_requests" in histogram_data and "decayed_original_bid_requests" in histogram_data:
        orig_hist = histogram_data["original_bid_requests"].sort_values("bin_center")
        decay_hist = histogram_data["decayed_original_bid_requests"].sort_values("bin_center")
        
        fig1 = go.Figure()
        
        # Step line for original distribution
        fig1.add_trace(go.Scatter(
            x=orig_hist["bin_center"],
            y=orig_hist["probability"],
            name="Original bid_requests",
            mode='lines',
            line=dict(color='#1f77b4', width=2, shape='hvh'),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.3)',
        ))
        
        # Step line for decayed distribution
        fig1.add_trace(go.Scatter(
            x=decay_hist["bin_center"],
            y=decay_hist["probability"],
            name="Decayed bid_requests",
            mode='lines',
            line=dict(color='#ff7f0e', width=2, shape='hvh'),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.3)',
        ))
        
        fig1.update_layout(
            title=f"Original Bid Requests: Original vs Decayed (decay_rate={DECAY_RATES['original_bid_requests']})",
            xaxis_title="Value (log scale)",
            yaxis_title="Probability",
            template='plotly_white',
            xaxis_type="log",
            height=500,
            hovermode='x unified',
        )
        
        fig1.show()
        print("✅ Displayed: Original Bid Requests distribution (computed in Spark)")
    
    # Create histogram comparison for UNR if available
    if "unr" in histogram_data and "decayed_unr" in histogram_data:
        orig_hist = histogram_data["unr"].sort_values("bin_center")
        decay_hist = histogram_data["decayed_unr"].sort_values("bin_center")
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=orig_hist["bin_center"],
            y=orig_hist["probability"],
            name="Original UNR",
            mode='lines',
            line=dict(color='#2ca02c', width=2, shape='hvh'),
            fill='tozeroy',
            fillcolor='rgba(44, 160, 44, 0.3)',
        ))
        
        fig2.add_trace(go.Scatter(
            x=decay_hist["bin_center"],
            y=decay_hist["probability"],
            name="Decayed UNR",
            mode='lines',
            line=dict(color='#d62728', width=2, shape='hvh'),
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.3)',
        ))
        
        fig2.update_layout(
            title=f"UNR: Original vs Decayed (decay_rate={DECAY_RATES['unr']})",
            xaxis_title="Value (log scale)",
            yaxis_title="Probability",
            template='plotly_white',
            xaxis_type="log",
            height=500,
            hovermode='x unified',
        )
        
        fig2.show()
        print("✅ Displayed: UNR distribution (computed in Spark)")
    
    # Create line chart showing decay effect by date
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=["Bid Requests by Date", "UNR by Date"])
    
    fig3.add_trace(
        go.Scatter(x=agg_by_date_pdf["date"], y=agg_by_date_pdf["total_original_bid_requests"], 
                   name="Original", mode='lines+markers'),
        row=1, col=1
    )
    fig3.add_trace(
        go.Scatter(x=agg_by_date_pdf["date"], y=agg_by_date_pdf["total_decayed_original_bid_requests"], 
                   name="Decayed", mode='lines+markers'),
        row=1, col=1
    )
    
    if "total_unr" in agg_by_date_pdf.columns:
        fig3.add_trace(
            go.Scatter(x=agg_by_date_pdf["date"], y=agg_by_date_pdf["total_unr"], 
                       name="Original UNR", mode='lines+markers', showlegend=False),
            row=1, col=2
        )
        fig3.add_trace(
            go.Scatter(x=agg_by_date_pdf["date"], y=agg_by_date_pdf["total_decayed_unr"], 
                       name="Decayed UNR", mode='lines+markers', showlegend=False),
            row=1, col=2
        )
    
    decay_rates_str = ", ".join([f"{k}={v}" for k, v in DECAY_RATES.items()])
    fig3.update_layout(
        title=f"Decay Effect Over Time ({decay_rates_str})",
        template='plotly_white',
        height=400,
    )
    
    fig3.show()
    print("✅ Displayed: Time series chart")
    
    # Create bar chart comparing cosine similarities
    metrics = list(comparison_results.keys())
    cos_orig = [comparison_results[m]["cos_original_vs_test"] for m in metrics]
    cos_decay = [comparison_results[m]["cos_decayed_vs_test"] for m in metrics]
    
    fig_sim = go.Figure()
    
    fig_sim.add_trace(go.Bar(
        name='Training (original) → Test',
        x=metrics,
        y=cos_orig,
        marker_color='#1f77b4',
        text=[f'{v:.6f}' for v in cos_orig],
        textposition='outside',
    ))
    
    fig_sim.add_trace(go.Bar(
        name='Training (decayed) → Test',
        x=metrics,
        y=cos_decay,
        marker_color='#ff7f0e',
        text=[f'{v:.6f}' for v in cos_decay],
        textposition='outside',
    ))
    
    # Compute y-axis range safely
    all_cos_values = cos_orig + cos_decay
    y_min = min(all_cos_values) * 0.99 if all_cos_values else 0
    y_max = 1.0
    
    fig_sim.update_layout(
        title=f"Cosine Similarity: Training vs Test<br><sub>Decay rates: {decay_rates_str} | Higher = more similar</sub>",
        xaxis_title="Metric",
        yaxis_title="Cosine Similarity",
        barmode='group',
        template='plotly_white',
        height=500,
        yaxis=dict(range=[y_min, y_max]),
    )
    
    fig_sim.show()
    print("✅ Displayed: Cosine similarity comparison")

except ImportError:
    print("⚠️ Plotly not available. Install with: pip install plotly")
    print("Falling back to summary statistics...")
    
    # Show summary statistics instead
    print("\nSummary statistics:")
    agg_df.select(
        "original_bid_requests", "decayed_original_bid_requests",
        *([c for c in ["unr", "decayed_unr"] if c in agg_df.columns])
    ).describe().show()

# Cleanup
agg_df.unpersist()
test_agg_df.unpersist()
train_nodecay_df.unpersist()
train_decayed_df.unpersist()

print("\n" + "=" * 80)
print("✅ Quick decay check complete!")
print("=" * 80)
