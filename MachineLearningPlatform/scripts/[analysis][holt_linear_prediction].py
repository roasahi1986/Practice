"""
Holt's Linear Exponential Smoothing for Training Data Prediction.

This script implements Holt's method to predict the next value in a short time series.
Unlike temporal decay (which just weights and averages), Holt's method explicitly
models both LEVEL and TREND, then extrapolates.

Key advantages over decay:
- Adapts to increasing, decreasing, or stable patterns
- Extrapolates trend (doesn't just average)
- Works with very short series (7 days)

Formula:
    Level:  L_t = α * y_t + (1 - α) * (L_{t-1} + T_{t-1})
    Trend:  T_t = β * (L_t - L_{t-1}) + (1 - β) * T_{t-1}
    Forecast: F_{t+1} = L_t + T_t

Parameters:
    α (alpha): Level smoothing (0-1). Higher = more responsive to recent values
    β (beta):  Trend smoothing (0-1). Higher = faster trend adaptation

Usage (in Databricks notebook):
    %run ./scripts/holt_linear_prediction.py
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, StringType, StructType, StructField
import numpy as np

spark = SparkSession.builder.getOrCreate()

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

GROUP_BY_FEATURES = [
    "rtb_id",
    "supply_name",
    "pub_app_object_id",
    "geo",
    "placement_type",
    "platform",
]

AGG_COLUMNS = ["bid_requests", "multiplied_bid_requests", "unr", "original_bid_requests"]

RENAME_COLUMNS = {"rtb_connection_id": "rtb_id"}

TOP_K_PUB_SIZE = 5000
EXP_BUCKET_CONTROL = "project_v1_control"

# Holt's smoothing parameters to test
ALPHA_VALUES = [0.2, 0.3, 0.5, 0.7]  # Level smoothing
BETA_VALUES = [0.1, 0.2, 0.3]        # Trend smoothing

# Decay rates to compare against
DECAY_RATES = [0.85, 0.90, 0.92, 0.95]

# Metrics to predict
PREDICTION_METRICS = ["unr_per_original_bid_requests", "unr", "original_bid_requests"]

# =============================================================================
# Prediction Methods
# =============================================================================

def holt_linear_predict(values: np.ndarray, alpha: float = 0.3, beta: float = 0.1) -> float:
    """
    Holt's Linear Exponential Smoothing.
    
    Captures level and trend with exponential smoothing, then extrapolates.
    
    Args:
        values: Time series values (oldest first)
        alpha: Smoothing factor for level (0-1). Higher = more weight on recent.
        beta: Smoothing factor for trend (0-1). Higher = faster trend adaptation.
    
    Returns:
        Predicted next value
    """
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]
    
    # Initialize level and trend
    # Level: start with first value
    # Trend: start with first difference
    level = values[0]
    trend = values[1] - values[0]
    
    # Update through the series
    for i in range(1, n):
        prev_level = level
        # Level update: blend current observation with predicted value
        level = alpha * values[i] + (1 - alpha) * (prev_level + trend)
        # Trend update: blend current level change with previous trend
        trend = beta * (level - prev_level) + (1 - beta) * trend
    
    # Forecast 1 step ahead: level + trend
    return level + trend


def decay_weighted_mean(values: np.ndarray, decay_rate: float = 0.92) -> float:
    """
    Exponential decay weighted mean (current approach for comparison).
    
    Most recent value has weight 1, each day back has weight * decay_rate.
    """
    n = len(values)
    if n == 0:
        return 0.0
    
    weights = np.array([decay_rate ** (n - 1 - i) for i in range(n)])
    return np.sum(values * weights) / np.sum(weights)


def last_value_only(values: np.ndarray) -> float:
    """Use only the most recent value."""
    if len(values) == 0:
        return 0.0
    return values[-1]


def mean_value(values: np.ndarray) -> float:
    """Simple mean of all values."""
    if len(values) == 0:
        return 0.0
    return np.mean(values)


def linear_trend_predict(values: np.ndarray) -> float:
    """Fit linear trend and extrapolate."""
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]
    
    t = np.arange(n)
    mean_t = np.mean(t)
    mean_y = np.mean(values)
    
    cov_ty = np.sum((t - mean_t) * (values - mean_y))
    var_t = np.sum((t - mean_t) ** 2)
    
    if var_t == 0:
        return mean_y
    
    slope = cov_ty / var_t
    intercept = mean_y - slope * mean_t
    
    return intercept + slope * n


# =============================================================================
# Step 1: Load Data
# =============================================================================
print("=" * 80)
print("Step 1: Loading data...")
print("=" * 80)

BASE_PATH = "s3://exchange/machine_learning/data/42"
raw_df = spark.read.option("basePath", BASE_PATH).parquet(*TRAINING_PATHS)

for old_name, new_name in RENAME_COLUMNS.items():
    if old_name in raw_df.columns:
        raw_df = raw_df.withColumnRenamed(old_name, new_name)

if "dt" in raw_df.columns:
    raw_df = raw_df.withColumn("date", F.to_date(F.col("dt")))

print(f"Loaded {len(TRAINING_PATHS)} training partitions")

# Extract top-K publishers
publisher_agg = (
    raw_df.filter(F.col("exp_bucket") == EXP_BUCKET_CONTROL)
    .groupBy("pub_app_object_id")
    .agg(F.sum("unr").alias("unr"))
)

top_k_publishers = (
    publisher_agg.orderBy(F.col("unr").desc())
    .limit(TOP_K_PUB_SIZE)
    .select("pub_app_object_id")
    .rdd.flatMap(lambda x: x)
    .collect()
)

print(f"✅ Identified {len(top_k_publishers)} top publishers")

# Prepare training data
raw_df = raw_df.withColumn(
    "original_bid_requests",
    F.col("bid_requests") - F.coalesce(F.col("multiplied_bid_requests"), F.lit(0))
)

raw_df = raw_df.withColumn(
    "pub_app_object_id",
    F.when(F.col("pub_app_object_id").isin(top_k_publishers), F.col("pub_app_object_id"))
    .otherwise(F.lit("other"))
)

# Load test data
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
    F.when(F.col("pub_app_object_id").isin(top_k_publishers), F.col("pub_app_object_id"))
    .otherwise(F.lit("other"))
)

print(f"Loaded {len(TEST_PATHS)} test partitions")

# =============================================================================
# Step 2: Aggregate by Feature + Date
# =============================================================================
print("\n" + "=" * 80)
print("Step 2: Aggregating by feature combination and date...")
print("=" * 80)

available_group_cols = [c for c in GROUP_BY_FEATURES if c in raw_df.columns]
available_agg_cols = [c for c in AGG_COLUMNS if c in raw_df.columns]

# Create feature key
feature_key_expr = F.concat_ws(
    "||",
    *[F.coalesce(F.col(c).cast("string"), F.lit("__NULL__")) for c in available_group_cols]
)

# Aggregate training by feature + date
train_agg_exprs = [F.sum(F.coalesce(F.col(c), F.lit(0))).alias(c) for c in available_agg_cols]

train_daily_df = (
    raw_df
    .withColumn("_feature_key", feature_key_expr)
    .groupBy("_feature_key", "date")
    .agg(*train_agg_exprs)
    .withColumn(
        "unr_per_original_bid_requests",
        F.when(F.col("original_bid_requests") > 0,
               F.col("unr") / F.col("original_bid_requests")
        ).otherwise(F.lit(0.0))
    )
)

# Aggregate test by feature (all test days combined)
test_agg_df = (
    test_raw_df
    .withColumn("_feature_key", feature_key_expr)
    .groupBy("_feature_key")
    .agg(*train_agg_exprs)
    .withColumn(
        "unr_per_original_bid_requests",
        F.when(F.col("original_bid_requests") > 0,
               F.col("unr") / F.col("original_bid_requests")
        ).otherwise(F.lit(0.0))
    )
)

# Collect training data as time series per feature
print("Collecting training time series per feature...")
train_ts_df = train_daily_df.groupBy("_feature_key").agg(
    F.collect_list(F.struct("date", *PREDICTION_METRICS)).alias("time_series")
)

train_ts_pdf = train_ts_df.toPandas()
test_pdf = test_agg_df.select("_feature_key", *PREDICTION_METRICS).toPandas()

print(f"Collected {len(train_ts_pdf)} feature combinations from training")
print(f"Test has {len(test_pdf)} feature combinations")

# =============================================================================
# Step 3: Build All Prediction Methods
# =============================================================================
print("\n" + "=" * 80)
print("Step 3: Building prediction methods...")
print("=" * 80)

# Methods to compare
methods = {}

# Baselines
methods["mean"] = mean_value
methods["last_value"] = last_value_only
methods["linear_trend"] = linear_trend_predict

# Decay methods
for decay_rate in DECAY_RATES:
    methods[f"decay_{decay_rate}"] = lambda v, d=decay_rate: decay_weighted_mean(v, d)

# Holt's methods with different parameters
for alpha in ALPHA_VALUES:
    for beta in BETA_VALUES:
        methods[f"holt_a{alpha}_b{beta}"] = lambda v, a=alpha, b=beta: holt_linear_predict(v, a, b)

print(f"Total methods to compare: {len(methods)}")

# =============================================================================
# Step 4: Generate Predictions and Compute Errors
# =============================================================================
print("\n" + "=" * 80)
print("Step 4: Generating predictions and computing errors...")
print("=" * 80)

# Test lookup
test_lookup = {row["_feature_key"]: row for _, row in test_pdf.iterrows()}

# Store errors per metric per method
errors = {metric: {method: [] for method in methods} for metric in PREDICTION_METRICS}

matched_count = 0
skipped_count = 0

for idx, row in train_ts_pdf.iterrows():
    feature_key = row["_feature_key"]
    ts_data = row["time_series"]
    
    if feature_key not in test_lookup:
        skipped_count += 1
        continue
    
    test_row = test_lookup[feature_key]
    
    # Sort by date
    ts_sorted = sorted(ts_data, key=lambda x: x["date"])
    
    for metric in PREDICTION_METRICS:
        values = np.array([x[metric] for x in ts_sorted if x[metric] is not None])
        actual = test_row[metric]
        
        if len(values) == 0 or actual is None or actual == 0 or np.isnan(actual):
            continue
        
        matched_count += 1
        
        for method_name, method_func in methods.items():
            try:
                pred = method_func(values)
                rel_error = abs(pred - actual) / abs(actual)
                errors[metric][method_name].append(rel_error)
            except Exception:
                pass

print(f"Matched: {matched_count} (feature, metric) combinations")
print(f"Skipped: {skipped_count} feature combinations (not in test)")

# =============================================================================
# Step 5: Results Summary
# =============================================================================
print("\n" + "=" * 80)
print("Step 5: Results Summary")
print("=" * 80)

import pandas as pd

for metric in PREDICTION_METRICS:
    print(f"\n{'=' * 80}")
    print(f"📊 RESULTS FOR: {metric}")
    print("=" * 80)
    
    results = []
    for method_name, errs in errors[metric].items():
        if len(errs) > 0:
            mape = np.mean(errs) * 100
            median_ape = np.median(errs) * 100
            results.append({
                "method": method_name,
                "MAPE (%)": mape,
                "Median APE (%)": median_ape,
                "n": len(errs),
            })
    
    results_df = pd.DataFrame(results).sort_values("MAPE (%)")
    
    print("\nTop 10 Methods (by MAPE):")
    print("-" * 70)
    print(results_df.head(10).to_string(index=False))
    
    # Best method
    best = results_df.iloc[0]
    print(f"\n✅ Best method: {best['method']} (MAPE: {best['MAPE (%)']:.2f}%)")
    
    # Compare Holt's vs Decay
    holt_methods = results_df[results_df["method"].str.startswith("holt_")]
    decay_methods = results_df[results_df["method"].str.startswith("decay_")]
    
    if len(holt_methods) > 0 and len(decay_methods) > 0:
        best_holt = holt_methods.iloc[0]
        best_decay = decay_methods.iloc[0]
        
        improvement = best_decay["MAPE (%)"] - best_holt["MAPE (%)"]
        improvement_pct = improvement / best_decay["MAPE (%)"] * 100
        
        print(f"\n📈 Holt's vs Decay Comparison:")
        print(f"   Best Holt's:  {best_holt['method']} → MAPE {best_holt['MAPE (%)']:.2f}%")
        print(f"   Best Decay:   {best_decay['method']} → MAPE {best_decay['MAPE (%)']:.2f}%")
        
        if improvement > 0:
            print(f"   ✅ Holt's WINS by {improvement:.2f} percentage points ({improvement_pct:.1f}% relative improvement)")
        else:
            print(f"   ❌ Decay wins by {-improvement:.2f} percentage points")

# =============================================================================
# Step 6: Best Holt's Parameters Summary
# =============================================================================
print("\n" + "=" * 80)
print("Step 6: Best Holt's Parameters")
print("=" * 80)

for metric in PREDICTION_METRICS:
    results = []
    for method_name, errs in errors[metric].items():
        if method_name.startswith("holt_") and len(errs) > 0:
            # Parse parameters
            parts = method_name.replace("holt_a", "").replace("_b", " ").split()
            alpha = float(parts[0])
            beta = float(parts[1])
            mape = np.mean(errs) * 100
            results.append({"alpha": alpha, "beta": beta, "MAPE": mape})
    
    if results:
        results_df = pd.DataFrame(results).sort_values("MAPE")
        best = results_df.iloc[0]
        print(f"\n{metric}:")
        print(f"   Best α (alpha): {best['alpha']} - controls level responsiveness")
        print(f"   Best β (beta):  {best['beta']} - controls trend responsiveness")
        print(f"   MAPE: {best['MAPE']:.2f}%")

# =============================================================================
# Step 7: Visualization
# =============================================================================
print("\n" + "=" * 80)
print("Step 7: Generating visualizations...")
print("=" * 80)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Focus on the main metric
    metric = "unr_per_original_bid_requests"
    
    # Get results for this metric
    results = []
    for method_name, errs in errors[metric].items():
        if len(errs) > 0:
            results.append({
                "method": method_name,
                "MAPE": np.mean(errs) * 100,
                "type": "holt" if method_name.startswith("holt_") else 
                        "decay" if method_name.startswith("decay_") else "baseline"
            })
    
    results_df = pd.DataFrame(results).sort_values("MAPE")
    
    # Top 15 methods bar chart
    top_n = results_df.head(15)
    
    colors = {
        "holt": "#2ecc71",   # Green for Holt's
        "decay": "#e74c3c",  # Red for decay
        "baseline": "#3498db"  # Blue for baselines
    }
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_n["method"],
        y=top_n["MAPE"],
        marker_color=[colors[t] for t in top_n["type"]],
        text=[f'{v:.1f}%' for v in top_n["MAPE"]],
        textposition='outside',
    ))
    
    fig.update_layout(
        title=f"Prediction Error Comparison for {metric}<br><sub>🟢 Holt's | 🔴 Decay | 🔵 Baseline</sub>",
        xaxis_title="Method",
        yaxis_title="MAPE (%)",
        template='plotly_white',
        height=600,
        xaxis_tickangle=-45,
    )
    
    fig.show()
    print("✅ Displayed: Method comparison chart")
    
    # Holt's parameter heatmap
    holt_results = []
    for method_name, errs in errors[metric].items():
        if method_name.startswith("holt_") and len(errs) > 0:
            parts = method_name.replace("holt_a", "").replace("_b", " ").split()
            alpha = float(parts[0])
            beta = float(parts[1])
            mape = np.mean(errs) * 100
            holt_results.append({"alpha": alpha, "beta": beta, "MAPE": mape})
    
    if holt_results:
        holt_df = pd.DataFrame(holt_results)
        heatmap_data = holt_df.pivot(index="beta", columns="alpha", values="MAPE")
        
        fig2 = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[str(a) for a in heatmap_data.columns],
            y=[str(b) for b in heatmap_data.index],
            colorscale='RdYlGn_r',  # Red = high error (bad), Green = low error (good)
            text=np.round(heatmap_data.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 14},
            colorbar=dict(title="MAPE (%)"),
        ))
        
        fig2.update_layout(
            title="Holt's Parameter Sensitivity (MAPE %)<br><sub>Lower (green) is better</sub>",
            xaxis_title="α (alpha) - Level Smoothing",
            yaxis_title="β (beta) - Trend Smoothing",
            template='plotly_white',
            height=400,
            width=600,
        )
        
        fig2.show()
        print("✅ Displayed: Holt's parameter heatmap")

except ImportError:
    print("⚠️ Plotly not available")

# =============================================================================
# Step 8: Recommendation
# =============================================================================
print("\n" + "=" * 80)
print("Step 8: Recommendations")
print("=" * 80)

print("""
HOLT'S LINEAR SMOOTHING SUMMARY
================================

How it works:
    Level:  L_t = α * y_t + (1 - α) * (L_{t-1} + T_{t-1})
    Trend:  T_t = β * (L_t - L_{t-1}) + (1 - β) * T_{t-1}
    Forecast: F_{t+1} = L_t + T_t

Parameter interpretation:
    α (alpha): Level responsiveness
        - High α (0.7): Quick response to recent values
        - Low α (0.2): Smooth, stable level estimate
        
    β (beta): Trend responsiveness
        - High β (0.3): Quick trend changes
        - Low β (0.1): Stable trend, less sensitive to noise

When to use:
    ✅ Short time series (7 days is fine)
    ✅ Data with increasing or decreasing trends
    ✅ When you want to extrapolate, not just average
    ✅ No seasonality assumption required

Comparison to decay:
    - Decay: Weighted average → can only predict within historical range
    - Holt's: Extrapolation → can predict beyond historical range
    
    For INCREASING trends: Holt's extrapolates UP (decay underestimates)
    For DECREASING trends: Holt's extrapolates DOWN (decay may overestimate)
    For STABLE patterns: Both similar, Holt's slightly better due to trend=0
""")

print("\n" + "=" * 80)
print("✅ Holt's Linear Prediction analysis complete!")
print("=" * 80)

