"""
Day-to-Day Similarity Analysis.

Computes cosine similarity between consecutive days in the training set.
This helps understand how quickly the data distribution changes over time.

Key insights:
- If day-to-day similarity is HIGH and STABLE: distribution changes slowly
- If day-to-day similarity is LOW or DECLINING: distribution changes rapidly
- Faster change = more aggressive decay is needed

Usage (in Databricks notebook):
    %run ./scripts/day_to_day_similarity.py

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
    "s3://exchange/machine_learning/data/42/dt=2025-11-26",
    # Include test days to see full continuity
    "s3://exchange/machine_learning/data/42/dt=2025-11-27",
    "s3://exchange/machine_learning/data/42/dt=2025-11-28",
    "s3://exchange/machine_learning/data/42/dt=2025-11-29"
]

# Features to group by (aggregation dimensions)
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
# Step 1: Load All Data
# =============================================================================
print("=" * 80)
print("Step 1: Loading all data from S3...")
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
# Step 2: Aggregate Each Day Separately
# =============================================================================
print("\n" + "=" * 80)
print("Step 2: Aggregating each day separately...")
print("=" * 80)

available_group_cols = [c for c in GROUP_BY_FEATURES if c in raw_df.columns]
available_agg_cols = [c for c in AGG_COLUMNS if c in raw_df.columns]

# Get distinct dates
all_dates = sorted([
    row["date"] for row in raw_df.select("date").distinct().collect()
])
print(f"All dates: {all_dates}")

# Pre-aggregate each day and cache
daily_aggregates = {}

for date in all_dates:
    print(f"   Aggregating {date}...")
    
    single_day_df = raw_df.filter(F.col("date") == date)
    
    agg_exprs = [F.sum(F.coalesce(F.col(c), F.lit(0))).alias(c) for c in available_agg_cols]
    single_day_agg = single_day_df.groupBy(available_group_cols).agg(*agg_exprs)
    
    # Compute unr_per_original_bid_requests
    single_day_agg = single_day_agg.withColumn(
        "unr_per_original_bid_requests",
        F.when(F.col("original_bid_requests") > 0,
               F.col("unr") / F.col("original_bid_requests")
        ).otherwise(F.lit(0.0))
    )
    
    # Add feature key
    key_expr = F.concat_ws(
        "||",
        *[F.coalesce(F.col(c).cast("string"), F.lit("__NULL__")) for c in available_group_cols]
    )
    single_day_agg = single_day_agg.withColumn("_feature_key", key_expr)
    
    single_day_agg = single_day_agg.cache()
    single_day_agg.count()  # Force cache
    
    daily_aggregates[date] = single_day_agg

print(f"✅ Cached {len(daily_aggregates)} daily aggregates")

# =============================================================================
# Step 3: Compute Day-to-Day Similarity
# =============================================================================
print("\n" + "=" * 80)
print("Step 3: Computing day-to-day cosine similarity...")
print("=" * 80)


def compute_cosine_similarity(df1, df2, metric_col):
    """Compute cosine similarity between two day's data."""
    joined = df1.alias("day1").join(
        df2.alias("day2"),
        F.col("day1._feature_key") == F.col("day2._feature_key"),
        "inner"  # Only compare matching feature combinations
    ).select(
        F.col("day1._feature_key").alias("feature_key"),
        F.col(f"day1.{metric_col}").alias("val1"),
        F.col(f"day2.{metric_col}").alias("val2"),
    )
    
    # Filter rows where both values are non-zero
    valid_df = joined.filter(
        (F.col("val1").isNotNull()) & (F.col("val2").isNotNull()) &
        (F.col("val1") != 0) & (F.col("val2") != 0)
    )
    
    stats = valid_df.agg(
        F.sum(F.col("val1") * F.col("val2")).alias("dot_product"),
        F.sqrt(F.sum(F.col("val1") ** 2)).alias("norm1"),
        F.sqrt(F.sum(F.col("val2") ** 2)).alias("norm2"),
        F.count("*").alias("matched_count")
    ).collect()[0]
    
    dot_product = stats["dot_product"] or 0
    norm1 = stats["norm1"] or 0
    norm2 = stats["norm2"] or 0
    matched_count = stats["matched_count"] or 0
    
    if norm1 > 0 and norm2 > 0:
        cosine_sim = dot_product / (norm1 * norm2)
    else:
        cosine_sim = 0
    
    return {
        "cosine_similarity": cosine_sim,
        "matched_count": matched_count,
    }


# Compute consecutive day similarities
consecutive_results = []

for i in range(len(all_dates) - 1):
    date1 = all_dates[i]
    date2 = all_dates[i + 1]
    
    print(f"\n📅 {date1} → {date2}")
    
    result = {"date1": date1, "date2": date2, "pair": f"{date1} → {date2}"}
    
    for metric in SIMILARITY_COLUMNS:
        sim_result = compute_cosine_similarity(
            daily_aggregates[date1],
            daily_aggregates[date2],
            metric
        )
        result[f"cos_{metric}"] = sim_result["cosine_similarity"]
        result[f"matched_{metric}"] = sim_result["matched_count"]
        print(f"   {metric}: {sim_result['cosine_similarity']:.6f}")
    
    consecutive_results.append(result)

# =============================================================================
# Step 4: Compute N-Day Gap Similarities
# =============================================================================
print("\n" + "=" * 80)
print("Step 4: Computing similarity across different day gaps...")
print("=" * 80)

gap_results = []

# For each gap size (1 day, 2 days, etc.)
max_gap = min(5, len(all_dates) - 1)

for gap in range(1, max_gap + 1):
    print(f"\n📏 Gap = {gap} day(s)")
    
    similarities = {metric: [] for metric in SIMILARITY_COLUMNS}
    
    for i in range(len(all_dates) - gap):
        date1 = all_dates[i]
        date2 = all_dates[i + gap]
        
        for metric in SIMILARITY_COLUMNS:
            sim_result = compute_cosine_similarity(
                daily_aggregates[date1],
                daily_aggregates[date2],
                metric
            )
            similarities[metric].append(sim_result["cosine_similarity"])
    
    # Average similarity for this gap
    result = {"gap_days": gap}
    for metric in SIMILARITY_COLUMNS:
        avg_sim = sum(similarities[metric]) / len(similarities[metric]) if similarities[metric] else 0
        result[f"avg_cos_{metric}"] = avg_sim
        print(f"   Avg {metric}: {avg_sim:.6f}")
    
    gap_results.append(result)

# =============================================================================
# Step 5: Display Results
# =============================================================================
print("\n" + "=" * 80)
print("Step 5: Summary Results")
print("=" * 80)

import pandas as pd
import numpy as np

# Consecutive day results
consec_pdf = pd.DataFrame(consecutive_results)
print("\n📊 CONSECUTIVE DAY SIMILARITY")
print("=" * 80)
print(consec_pdf[["pair"] + [f"cos_{m}" for m in SIMILARITY_COLUMNS]].to_string(index=False))

# Gap analysis results
gap_pdf = pd.DataFrame(gap_results)
print("\n📊 SIMILARITY BY DAY GAP (averaged across all pairs)")
print("=" * 80)
print(gap_pdf.to_string(index=False))

# Compute decay rate suggestions
print("\n📈 DECAY RATE SUGGESTIONS")
print("=" * 80)

for metric in SIMILARITY_COLUMNS:
    col = f"avg_cos_{metric}"
    if col in gap_pdf.columns and len(gap_pdf) >= 2:
        # Fit exponential decay: similarity = base * decay_rate^gap
        # log(similarity) = log(base) + gap * log(decay_rate)
        gaps = gap_pdf["gap_days"].values
        sims = gap_pdf[col].values
        
        # Filter out zeros
        valid_mask = sims > 0
        if valid_mask.sum() >= 2:
            log_sims = np.log(sims[valid_mask])
            valid_gaps = gaps[valid_mask]
            
            # Linear regression on log scale
            slope, intercept = np.polyfit(valid_gaps, log_sims, 1)
            implied_decay_rate = np.exp(slope)
            
            print(f"\n{metric}:")
            print(f"   1-day similarity: {gap_pdf[col].iloc[0]:.6f}")
            if len(gap_pdf) >= 2:
                print(f"   2-day similarity: {gap_pdf[col].iloc[1]:.6f}")
            print(f"   Implied decay rate: {implied_decay_rate:.4f}")
            print(f"   Suggested decay parameter: {implied_decay_rate:.2f}")

# =============================================================================
# Step 6: Visualizations
# =============================================================================
print("\n" + "=" * 80)
print("Step 6: Generating visualizations...")
print("=" * 80)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Chart 1: Consecutive day similarity
    fig1 = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(SIMILARITY_COLUMNS):
        col = f"cos_{metric}"
        fig1.add_trace(
            go.Scatter(
                x=consec_pdf["pair"],
                y=consec_pdf[col],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i], width=3),
                marker=dict(size=10),
            )
        )
    
    fig1.update_layout(
        title="Consecutive Day Similarity<br><sub>How similar is each day to the next?</sub>",
        xaxis_title="Day Pair",
        yaxis_title="Cosine Similarity",
        template='plotly_white',
        height=500,
        yaxis=dict(range=[min(0.9, consec_pdf[[f"cos_{m}" for m in SIMILARITY_COLUMNS]].min().min() * 0.99), 1.0]),
    )
    
    fig1.show()
    print("✅ Displayed: Consecutive day similarity chart")
    
    # Chart 2: Similarity decay by gap
    fig2 = go.Figure()
    
    for i, metric in enumerate(SIMILARITY_COLUMNS):
        col = f"avg_cos_{metric}"
        fig2.add_trace(
            go.Scatter(
                x=gap_pdf["gap_days"],
                y=gap_pdf[col],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i], width=3),
                marker=dict(size=12),
            )
        )
    
    fig2.update_layout(
        title="Similarity Decay by Day Gap<br><sub>How quickly does similarity decay with time distance?</sub>",
        xaxis_title="Days Apart",
        yaxis_title="Average Cosine Similarity",
        template='plotly_white',
        height=500,
        xaxis=dict(dtick=1),
        yaxis=dict(range=[min(0.85, gap_pdf[[f"avg_cos_{m}" for m in SIMILARITY_COLUMNS]].min().min() * 0.99), 1.0]),
    )
    
    fig2.show()
    print("✅ Displayed: Similarity decay by gap chart")
    
    # Chart 3: Heatmap of all pairwise similarities for primary metric
    primary_metric = "unr_per_original_bid_requests"
    
    # Compute full pairwise similarity matrix
    n_days = len(all_dates)
    similarity_matrix = np.zeros((n_days, n_days))
    
    print(f"\nComputing full pairwise similarity matrix for {primary_metric}...")
    for i, date1 in enumerate(all_dates):
        for j, date2 in enumerate(all_dates):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                sim_result = compute_cosine_similarity(
                    daily_aggregates[date1],
                    daily_aggregates[date2],
                    primary_metric
                )
                similarity_matrix[i, j] = sim_result["cosine_similarity"]
                similarity_matrix[j, i] = sim_result["cosine_similarity"]
    
    date_labels = [str(d) for d in all_dates]
    
    fig3 = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=date_labels,
        y=date_labels,
        colorscale='RdYlGn',
        zmin=similarity_matrix[similarity_matrix < 1].min() * 0.99 if (similarity_matrix < 1).any() else 0.9,
        zmax=1.0,
        text=np.round(similarity_matrix, 4),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig3.update_layout(
        title=f"Pairwise Day Similarity Matrix ({primary_metric})<br><sub>Diagonal = 1.0, off-diagonal shows cross-day similarity</sub>",
        xaxis_title="Date",
        yaxis_title="Date",
        template='plotly_white',
        height=600,
        width=800,
    )
    
    fig3.show()
    print("✅ Displayed: Pairwise similarity heatmap")

except ImportError:
    print("⚠️ Plotly not available. Install with: pip install plotly")

# =============================================================================
# Cleanup
# =============================================================================
for df in daily_aggregates.values():
    df.unpersist()

print("\n" + "=" * 80)
print("✅ Day-to-day similarity analysis complete!")
print("=" * 80)
print("""
INTERPRETATION:

1. CONSECUTIVE DAY SIMILARITY:
   - High (~0.95+): Distribution is stable day-to-day
   - Low (~0.90-): Distribution changes significantly each day

2. SIMILARITY DECAY BY GAP:
   - Steep decline: Data becomes stale quickly → use aggressive decay
   - Gradual decline: Data stays relevant longer → use mild decay
   - Flat: Time doesn't matter much → decay may not help

3. IMPLIED DECAY RATE:
   - Suggests what decay factor to use based on observed similarity patterns
   - E.g., if 1-day similarity is 0.95, implied decay ≈ 0.95
""")

