"""
Hour-to-Hour Similarity Analysis.

Computes cosine similarity between consecutive hours to understand
intra-day distribution changes.

Key insights:
- If hour-to-hour similarity is HIGH: distribution is stable within a day
- If hour-to-hour similarity varies by time: there may be time-of-day effects
- Comparison with day-to-day similarity shows relative importance of recency

Usage (in Databricks notebook):
    %run ./scripts/hour_to_hour_similarity.py

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
# Use fewer days for hourly analysis (more granular = more data)
TRAINING_PATHS = [
    "s3://exchange/machine_learning/data/42/dt=2025-11-24",
    "s3://exchange/machine_learning/data/42/dt=2025-11-25",
    "s3://exchange/machine_learning/data/42/dt=2025-11-26",
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

# Hour column name in the data (adjust if different)
HOUR_COLUMN = "hour"  # or "hr" depending on your schema

# =============================================================================
# Step 1: Load Data
# =============================================================================
print("=" * 80)
print("Step 1: Loading data from S3...")
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

# Check for hour column
print("\nChecking for hour column...")
if HOUR_COLUMN not in raw_df.columns:
    # Try alternative names
    hour_alternatives = ["hr", "hour", "Hour", "HR"]
    found_hour_col = None
    for alt in hour_alternatives:
        if alt in raw_df.columns:
            found_hour_col = alt
            break
    
    if found_hour_col:
        HOUR_COLUMN = found_hour_col
        print(f"Found hour column: {HOUR_COLUMN}")
    else:
        print(f"⚠️ No hour column found. Available columns: {raw_df.columns[:20]}")
        print("Will try to extract from timestamp if available...")
        
        # Try to extract from timestamp column
        ts_columns = [c for c in raw_df.columns if 'time' in c.lower() or 'ts' in c.lower()]
        if ts_columns:
            ts_col = ts_columns[0]
            raw_df = raw_df.withColumn("hour", F.hour(F.col(ts_col)))
            HOUR_COLUMN = "hour"
            print(f"Extracted hour from {ts_col}")
        else:
            raise ValueError("Cannot find or derive hour column!")
else:
    print(f"Using hour column: {HOUR_COLUMN}")

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
# Step 2: Create DateTime Key and Aggregate by Hour
# =============================================================================
print("\n" + "=" * 80)
print("Step 2: Aggregating by date-hour...")
print("=" * 80)

# Create a combined date-hour key
raw_df = raw_df.withColumn(
    "date_hour",
    F.concat(F.col("date").cast("string"), F.lit(" "), F.lpad(F.col(HOUR_COLUMN).cast("string"), 2, "0"))
)

available_group_cols = [c for c in GROUP_BY_FEATURES if c in raw_df.columns]
available_agg_cols = [c for c in AGG_COLUMNS if c in raw_df.columns]

# Get distinct date-hours
all_date_hours = sorted([
    row["date_hour"] for row in raw_df.select("date_hour").distinct().collect()
])
print(f"Total date-hours: {len(all_date_hours)}")
print(f"First: {all_date_hours[0]}, Last: {all_date_hours[-1]}")

# Pre-aggregate each hour and cache
hourly_aggregates = {}

print("\nAggregating each hour...")
for i, date_hour in enumerate(all_date_hours):
    if i % 12 == 0:  # Print progress every 12 hours
        print(f"   Processing {date_hour}... ({i+1}/{len(all_date_hours)})")
    
    single_hour_df = raw_df.filter(F.col("date_hour") == date_hour)
    
    agg_exprs = [F.sum(F.coalesce(F.col(c), F.lit(0))).alias(c) for c in available_agg_cols]
    single_hour_agg = single_hour_df.groupBy(available_group_cols).agg(*agg_exprs)
    
    # Compute unr_per_original_bid_requests
    single_hour_agg = single_hour_agg.withColumn(
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
    single_hour_agg = single_hour_agg.withColumn("_feature_key", key_expr)
    
    single_hour_agg = single_hour_agg.cache()
    single_hour_agg.count()  # Force cache
    
    hourly_aggregates[date_hour] = single_hour_agg

print(f"✅ Cached {len(hourly_aggregates)} hourly aggregates")

# =============================================================================
# Step 3: Compute Hour-to-Hour Similarity
# =============================================================================
print("\n" + "=" * 80)
print("Step 3: Computing consecutive hour similarity...")
print("=" * 80)


def compute_cosine_similarity(df1, df2, metric_col):
    """Compute cosine similarity between two hour's data."""
    joined = df1.alias("h1").join(
        df2.alias("h2"),
        F.col("h1._feature_key") == F.col("h2._feature_key"),
        "right"
    ).select(
        F.col("h1._feature_key").alias("feature_key"),
        F.col(f"h1.{metric_col}").alias("val1"),
        F.col(f"h2.{metric_col}").alias("val2"),
    )
    
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


# Compute consecutive hour similarities
consecutive_results = []

for i in range(len(all_date_hours) - 1):
    dh1 = all_date_hours[i]
    dh2 = all_date_hours[i + 1]
    
    if i % 12 == 0:
        print(f"   Computing {dh1} → {dh2}... ({i+1}/{len(all_date_hours)-1})")
    
    result = {
        "date_hour_1": dh1,
        "date_hour_2": dh2,
        "hour_1": int(dh1.split(" ")[1]),
        "hour_2": int(dh2.split(" ")[1]),
        "date_1": dh1.split(" ")[0],
    }
    
    # Check if this is a day boundary (hour 23 → hour 00)
    result["is_day_boundary"] = result["hour_1"] == 23 and result["hour_2"] == 0
    
    for metric in SIMILARITY_COLUMNS:
        sim_result = compute_cosine_similarity(
            hourly_aggregates[dh1],
            hourly_aggregates[dh2],
            metric
        )
        result[f"cos_{metric}"] = sim_result["cosine_similarity"]
        result[f"matched_{metric}"] = sim_result["matched_count"]
    
    consecutive_results.append(result)

print(f"✅ Computed {len(consecutive_results)} consecutive hour similarities")

# =============================================================================
# Step 4: Compute Hourly Gap Analysis
# =============================================================================
print("\n" + "=" * 80)
print("Step 4: Computing similarity by hour gap...")
print("=" * 80)

gap_results = []

# For each gap size (1 hour, 2 hours, ... up to 48 hours)
max_gap = min(48, len(all_date_hours) - 1)

for gap in [1, 2, 3, 6, 12, 24, 48]:
    if gap >= len(all_date_hours):
        break
    
    print(f"   Gap = {gap} hour(s)...")
    
    similarities = {metric: [] for metric in SIMILARITY_COLUMNS}
    
    for i in range(len(all_date_hours) - gap):
        dh1 = all_date_hours[i]
        dh2 = all_date_hours[i + gap]
        
        for metric in SIMILARITY_COLUMNS:
            sim_result = compute_cosine_similarity(
                hourly_aggregates[dh1],
                hourly_aggregates[dh2],
                metric
            )
            similarities[metric].append(sim_result["cosine_similarity"])
    
    result = {"gap_hours": gap}
    for metric in SIMILARITY_COLUMNS:
        avg_sim = sum(similarities[metric]) / len(similarities[metric]) if similarities[metric] else 0
        result[f"avg_cos_{metric}"] = avg_sim
        print(f"      {metric}: {avg_sim:.6f}")
    
    gap_results.append(result)

# =============================================================================
# Step 5: Analyze by Hour of Day
# =============================================================================
print("\n" + "=" * 80)
print("Step 5: Analyzing similarity by hour of day...")
print("=" * 80)

import pandas as pd
import numpy as np

consec_pdf = pd.DataFrame(consecutive_results)

# Group by starting hour to see if certain hours are more/less stable
hourly_stability = consec_pdf.groupby("hour_1").agg({
    f"cos_{m}": ["mean", "std"] for m in SIMILARITY_COLUMNS
}).reset_index()

# Flatten column names
hourly_stability.columns = ["hour"] + [
    f"{col[0]}_{col[1]}" for col in hourly_stability.columns[1:]
]

print("\n📊 SIMILARITY BY HOUR OF DAY (when transition starts)")
print("=" * 80)
print(hourly_stability.to_string(index=False))

# =============================================================================
# Step 6: Compare Day Boundaries vs Within-Day
# =============================================================================
print("\n" + "=" * 80)
print("Step 6: Day boundary vs within-day comparison...")
print("=" * 80)

day_boundary = consec_pdf[consec_pdf["is_day_boundary"]]
within_day = consec_pdf[~consec_pdf["is_day_boundary"]]

print("\n📊 DAY BOUNDARY TRANSITIONS (hour 23 → hour 00)")
for metric in SIMILARITY_COLUMNS:
    col = f"cos_{metric}"
    boundary_avg = day_boundary[col].mean() if len(day_boundary) > 0 else 0
    within_avg = within_day[col].mean() if len(within_day) > 0 else 0
    diff = boundary_avg - within_avg
    
    print(f"\n{metric}:")
    print(f"   Day boundary avg: {boundary_avg:.6f}")
    print(f"   Within-day avg:   {within_avg:.6f}")
    print(f"   Difference:       {diff:+.6f}")
    if diff < -0.01:
        print(f"   ⚠️ Day boundaries show LOWER similarity (distribution shifts at midnight)")
    elif diff > 0.01:
        print(f"   ✅ Day boundaries show HIGHER similarity (smooth transitions)")
    else:
        print(f"   ≈ No significant difference")

# =============================================================================
# Step 7: Summary Results
# =============================================================================
print("\n" + "=" * 80)
print("Step 7: Summary Results")
print("=" * 80)

gap_pdf = pd.DataFrame(gap_results)

print("\n📊 SIMILARITY BY HOUR GAP")
print("=" * 80)
print(gap_pdf.to_string(index=False))

# Compare 1-hour vs 24-hour gap
print("\n📈 HOURLY VS DAILY DECAY")
print("=" * 80)
for metric in SIMILARITY_COLUMNS:
    col = f"avg_cos_{metric}"
    if col in gap_pdf.columns:
        hour_1 = gap_pdf[gap_pdf["gap_hours"] == 1][col].values[0] if 1 in gap_pdf["gap_hours"].values else None
        hour_24 = gap_pdf[gap_pdf["gap_hours"] == 24][col].values[0] if 24 in gap_pdf["gap_hours"].values else None
        
        if hour_1 is not None and hour_24 is not None:
            decay_per_day = hour_1 - hour_24
            decay_per_hour = decay_per_day / 24
            
            print(f"\n{metric}:")
            print(f"   1-hour gap similarity:  {hour_1:.6f}")
            print(f"   24-hour gap similarity: {hour_24:.6f}")
            print(f"   Decay over 24 hours:    {decay_per_day:+.6f}")
            print(f"   Decay per hour:         {decay_per_hour:+.8f}")

# =============================================================================
# Step 8: Visualizations
# =============================================================================
print("\n" + "=" * 80)
print("Step 8: Generating visualizations...")
print("=" * 80)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Chart 1: Consecutive hour similarity over time
    fig1 = go.Figure()
    
    for i, metric in enumerate(SIMILARITY_COLUMNS):
        col = f"cos_{metric}"
        fig1.add_trace(
            go.Scatter(
                x=list(range(len(consec_pdf))),
                y=consec_pdf[col],
                mode='lines',
                name=metric,
                line=dict(color=colors[i], width=1),
                opacity=0.7,
            )
        )
    
    # Add day boundary markers
    boundary_indices = consec_pdf[consec_pdf["is_day_boundary"]].index.tolist()
    for idx in boundary_indices:
        fig1.add_vline(x=idx, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig1.update_layout(
        title="Hour-to-Hour Similarity Over Time<br><sub>Vertical lines = day boundaries (23:00 → 00:00)</sub>",
        xaxis_title="Hour Index",
        yaxis_title="Cosine Similarity",
        template='plotly_white',
        height=500,
    )
    
    fig1.show()
    print("✅ Displayed: Hour-to-hour similarity timeline")
    
    # Chart 2: Average similarity by hour of day
    fig2 = go.Figure()
    
    for i, metric in enumerate(SIMILARITY_COLUMNS):
        col = f"cos_{metric}_mean"
        fig2.add_trace(
            go.Scatter(
                x=hourly_stability["hour"],
                y=hourly_stability[col],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i], width=3),
                marker=dict(size=8),
            )
        )
    
    fig2.update_layout(
        title="Average Similarity by Hour of Day<br><sub>Does stability vary by time of day?</sub>",
        xaxis_title="Hour of Day (start of transition)",
        yaxis_title="Average Cosine Similarity",
        template='plotly_white',
        height=500,
        xaxis=dict(dtick=2, range=[-0.5, 23.5]),
    )
    
    fig2.show()
    print("✅ Displayed: Similarity by hour of day")
    
    # Chart 3: Similarity decay by gap
    fig3 = go.Figure()
    
    for i, metric in enumerate(SIMILARITY_COLUMNS):
        col = f"avg_cos_{metric}"
        fig3.add_trace(
            go.Scatter(
                x=gap_pdf["gap_hours"],
                y=gap_pdf[col],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i], width=3),
                marker=dict(size=12),
            )
        )
    
    # Add reference lines for key gaps
    fig3.add_vline(x=24, line_dash="dot", line_color="red", 
                   annotation_text="1 day", annotation_position="top")
    
    fig3.update_layout(
        title="Similarity Decay by Hour Gap<br><sub>How quickly does similarity drop over hours?</sub>",
        xaxis_title="Hours Apart",
        yaxis_title="Average Cosine Similarity",
        template='plotly_white',
        height=500,
    )
    
    fig3.show()
    print("✅ Displayed: Similarity decay by hour gap")
    
    # Chart 4: Day boundary vs within-day comparison
    fig4 = go.Figure()
    
    x_labels = SIMILARITY_COLUMNS
    boundary_vals = [day_boundary[f"cos_{m}"].mean() for m in SIMILARITY_COLUMNS]
    within_vals = [within_day[f"cos_{m}"].mean() for m in SIMILARITY_COLUMNS]
    
    fig4.add_trace(go.Bar(
        name='Day Boundary (23→00)',
        x=x_labels,
        y=boundary_vals,
        marker_color='#e74c3c',
        text=[f'{v:.4f}' for v in boundary_vals],
        textposition='outside',
    ))
    
    fig4.add_trace(go.Bar(
        name='Within Day',
        x=x_labels,
        y=within_vals,
        marker_color='#3498db',
        text=[f'{v:.4f}' for v in within_vals],
        textposition='outside',
    ))
    
    # Set y-axis range
    all_vals = boundary_vals + within_vals
    y_min = min(all_vals) * 0.99 if all_vals else 0.9
    
    fig4.update_layout(
        title="Day Boundary vs Within-Day Similarity<br><sub>Is there a discontinuity at midnight?</sub>",
        xaxis_title="Metric",
        yaxis_title="Average Cosine Similarity",
        barmode='group',
        template='plotly_white',
        height=500,
        yaxis=dict(range=[y_min, 1.0]),
    )
    
    fig4.show()
    print("✅ Displayed: Day boundary comparison")

except ImportError:
    print("⚠️ Plotly not available. Install with: pip install plotly")

# =============================================================================
# Cleanup
# =============================================================================
for df in hourly_aggregates.values():
    df.unpersist()

print("\n" + "=" * 80)
print("✅ Hour-to-hour similarity analysis complete!")
print("=" * 80)
print("""
INTERPRETATION:

1. HOUR-TO-HOUR SIMILARITY:
   - Very high (~0.98+): Distribution barely changes hour to hour
   - Moderate (~0.95): Noticeable hourly variation
   - Lower (~0.90): Significant hourly fluctuation

2. TIME-OF-DAY PATTERNS:
   - If certain hours show lower similarity: time-of-day effects exist
   - Consider adding hour as a feature or using hour-specific models

3. DAY BOUNDARY EFFECT:
   - Lower similarity at day boundaries: daily patterns reset at midnight
   - Similar to within-day: smooth continuous distribution

4. HOURLY VS DAILY DECAY:
   - Compare 1-hour gap to 24-hour gap to understand decay rate
   - Steep drop from 1h to 24h: fast decay needed
   - Gradual drop: slower decay is fine
""")

