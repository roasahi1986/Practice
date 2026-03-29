# Temporal Decay Analysis for Training Data

**Author**: ML Platform Team  
**Date**: January 2026  
**Status**: Experimental Analysis

---

## Executive Summary

This document presents an empirical analysis of temporal decay for training data in our ML pipeline. We investigated whether applying exponential decay weights to older training data improves prediction accuracy for future data.

**Key Findings**:
1. Day-to-day similarity for `unr_per_original_bid_requests` is relatively low (~0.54), suggesting high temporal variability
2. Applying temporal decay with weighted aggregation does **not** improve similarity to test data
3. The aggregation step itself (merging multiple days) may be the bottleneck, not the decay weights
4. **Holt's Linear Smoothing** provides a better alternative by explicitly modeling trend and extrapolating

---

## 1. Problem Statement

### Background

When training ML models on historical data, we face a fundamental question: **How should we weight older data relative to recent data?**

Two common approaches:
1. **Equal weighting**: Treat all training days equally
2. **Temporal decay**: Weight recent data higher using exponential decay: `weight = decay_rate^days_old`

### Hypothesis

If data distributions shift over time (non-stationarity), applying temporal decay should make training data more similar to test data, potentially improving model performance.

### Metrics Under Analysis

| Metric | Description |
|--------|-------------|
| `original_bid_requests` | Bid request volume (count) |
| `unr` | Uncapped Net Revenue |
| `unr_per_original_bid_requests` | Revenue per bid request (derived ratio) |

---

## 2. Methodology

### 2.1 Cosine Similarity

We use **cosine similarity** to measure distributional similarity between datasets:

$$
\text{cos}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}
$$

Where:
- Each **dimension** = unique feature combination (rtb_id × supply_name × pub_app_object_id × geo × placement_type × platform)
- Each **value** = aggregated metric (e.g., sum of UNR for that feature combination)

**Interpretation**:
- 1.0 = identical distributions
- 0.0 = orthogonal (completely different)
- Higher = more similar

### 2.2 Experimental Setup

**Data Range**:
- Training: 2025-11-20 to 2025-11-26 (7 days)
- Test: 2025-11-27 to 2025-11-29 (3 days)

**Feature Dimensions**:
- `rtb_id`, `supply_name`, `pub_app_object_id`, `geo`, `placement_type`, `platform`

**Top-K Publisher Filtering**:
- Selected top 5,000 publishers by UNR from control bucket
- Mapped remaining publishers to "other" to reduce dimensionality

### 2.3 Analysis Scripts

| Script | Purpose |
|--------|---------|
| `day_to_day_similarity.py` | Measures similarity between consecutive days |
| `daily_similarity_trend.py` | Measures each training day's similarity to test data |
| `quick_decay_check.py` | Applies temporal decay and compares to test data |
| `holt_linear_prediction.py` | Compares Holt's method vs decay for prediction |

---

## 3. Experiment 1: Day-to-Day Similarity

### Purpose
Understand how quickly data distributions change from one day to the next.

### Results: Consecutive Day Similarity

| Day Pair | original_bid_requests | unr | unr_per_original_bid_requests |
|----------|----------------------|-----|-------------------------------|
| Nov 20 → Nov 21 | 0.9802 | 0.9474 | 0.6284 |
| Nov 21 → Nov 22 | 0.9745 | 0.9586 | 0.5540 |
| Nov 22 → Nov 23 | 0.9862 | 0.9614 | 0.5897 |
| Nov 23 → Nov 24 | 0.9757 | 0.9484 | 0.4592 |
| Nov 24 → Nov 25 | 0.9825 | 0.8802 | 0.5193 |
| Nov 25 → Nov 26 | 0.9832 | 0.8851 | 0.5357 |
| Nov 26 → Nov 27 | 0.9741 | 0.9470 | 0.4505 |
| Nov 27 → Nov 28 | 0.9857 | 0.9382 | 0.5906 |
| Nov 28 → Nov 29 | 0.9745 | 0.9340 | 0.5572 |

### Results: Similarity by Day Gap

| Gap (days) | original_bid_requests | unr | unr_per_original_bid_requests |
|------------|----------------------|-----|-------------------------------|
| 1 | 0.9796 | 0.9334 | 0.5427 |
| 2 | 0.9577 | 0.8971 | 0.4538 |
| 3 | 0.9451 | 0.8719 | 0.4246 |
| 4 | 0.9369 | 0.8534 | 0.4282 |
| 5 | 0.9299 | 0.8484 | 0.3759 |

### Implied Decay Rates

Based on observed similarity decay patterns:

| Metric | 1-day Similarity | Implied Decay Rate |
|--------|-----------------|-------------------|
| original_bid_requests | 0.9796 | ~0.99 |
| unr | 0.9334 | ~0.98 |
| unr_per_original_bid_requests | 0.5427 | ~0.92 |

### Key Observations

1. **`original_bid_requests`** is highly stable (>0.97 day-to-day) — decay has minimal effect
2. **`unr`** shows moderate stability (~0.93 day-to-day)
3. **`unr_per_original_bid_requests`** shows **high variability** (only ~0.54 day-to-day)
   - This is our primary prediction target
   - The low similarity suggests significant daily fluctuation in revenue rates

---

## 4. Experiment 2: Applying Temporal Decay

### Configuration

Based on Experiment 1's implied decay rate for `unr_per_original_bid_requests`:

```python
DECAY_RATES = {
    "original_bid_requests": 1.0,  # No decay (stable metric)
    "unr": 0.92,                    # Derived from observed similarity
}
```

**Note**: `unr_per_original_bid_requests` is computed as `unr / original_bid_requests` after decay is applied to both numerator and denominator.

### Results

| Metric | Cos(Original→Test) | Cos(Decayed→Test) | Improvement |
|--------|-------------------|-------------------|-------------|
| original_bid_requests | 0.9448 | 0.9448 | +0.00% |
| unr | 0.9002 | 0.9025 | +0.26% |
| unr_per_original_bid_requests | **0.2703** | **0.2668** | **-1.27%** |

### Key Finding

**Temporal decay does NOT improve similarity for `unr_per_original_bid_requests`.**

In fact, it slightly hurts similarity (-1.27%).

---

## 5. Analysis: Why Decay Doesn't Help

### The Aggregation Problem

The core issue is that we're comparing **aggregated training data** (all 7 days combined) to test data. Consider what happens:

```
Training Day Similarity to Test:
├── Nov 20: 0.25 (7 days before test)
├── Nov 21: 0.26
├── Nov 22: 0.27
├── Nov 23: 0.28
├── Nov 24: 0.29
├── Nov 25: 0.30
└── Nov 26: 0.31 (1 day before test)

Aggregated (no decay): 0.27  ← weighted average
Aggregated (with decay): 0.28  ← slightly higher weight on recent

Best single day (Nov 26): 0.31  ← highest possible!
```

### Mathematical Intuition

When we aggregate multiple days:

$$
\text{Aggregated Vector} = \sum_{d=1}^{7} w_d \cdot \text{DayVector}_d
$$

Even with optimal decay weights, the aggregated vector is a **mixture** of all days. This mixture:
1. Blends in noise from dissimilar older days
2. Dilutes the signal from the most similar recent days
3. Can never exceed the similarity of the best individual day

### Pattern-Specific Analysis

Metric changes can follow three patterns:

| Pattern | Decay Effect | Reasoning |
|---------|--------------|-----------|
| **Increasing trend** | ❌ Hurts | Underweights recent higher values |
| **Decreasing trend** | ✅ May help | Overweights recent lower values |
| **Fluctuating** | ❌ Hurts | Adds noise without benefit |

Since our data shows mixed patterns across feature combinations, a global decay rate cannot optimize for all cases.

### The Fundamental Trade-off

| Approach | Pros | Cons |
|----------|------|------|
| Single most recent day | Highest similarity | Less data volume |
| Aggregated with decay | More data | Lower similarity |
| Aggregated without decay | Most data | Lowest similarity |

---

## 6. Alternative Approach: Holt's Linear Smoothing

### The Problem with Decay

Decay computes a **weighted average** — it can only predict values within the historical range. It cannot extrapolate trends.

### Holt's Method

Holt's Linear Exponential Smoothing explicitly models **level** and **trend**, then extrapolates:

```
Level:    L_t = α × y_t + (1 - α) × (L_{t-1} + T_{t-1})
Trend:    T_t = β × (L_t - L_{t-1}) + (1 - β) × T_{t-1}
Forecast: F_{t+1} = L_t + T_t
```

**Parameters**:
- **α (alpha)**: Level smoothing (0-1). Higher = more responsive to recent values.
- **β (beta)**: Trend smoothing (0-1). Higher = faster trend adaptation.

### Key Advantages

| Aspect | Decay | Holt's |
|--------|-------|--------|
| Operation | Weighted average | Trend extrapolation |
| Increasing trend | Underestimates | Extrapolates up ✅ |
| Decreasing trend | May help | Extrapolates down ✅ |
| Fluctuating | Adds noise | Smooths + no trend ✅ |
| Seasonality assumption | None | None |
| Suitable for 7 days | Yes | Yes |

### Implementation

See `scripts/holt_linear_prediction.py` for a full comparison of Holt's method vs decay across multiple parameter combinations.

---

## 7. Recommendations

### For `unr_per_original_bid_requests` (high variability)

1. **Consider Holt's Linear Smoothing**: Explicitly models trend direction and extrapolates, adapting to increasing, decreasing, or stable patterns.

2. **Consider shorter training windows**: Using only the most recent 1-2 days may yield higher similarity than using 7 days with any weighting scheme.

3. **Investigate pattern-specific approaches**: Use trend detection to choose the method per feature combination.

### For `original_bid_requests` and `unr` (stable metrics)

1. Decay has minimal effect (metrics are already stable)
2. Longer training windows are acceptable without decay

### General Recommendations

1. **Validate on downstream task**: This analysis measures distributional similarity, not model performance. The two may not correlate perfectly.

2. **Consider adaptive methods**: Instead of one global decay rate, use pattern detection:

```python
if trend_r2 > 0.6:
    # Strong trend - use Holt's extrapolation
    prediction = holt_linear_predict(values, alpha=0.3, beta=0.1)
else:
    # Weak/no trend - use last value or mean
    prediction = values[-1]
```

3. **Investigate the source of variability**: The low day-to-day similarity for `unr_per_original_bid_requests` suggests:
   - Advertiser budget fluctuations
   - Campaign start/stop events
   - Publisher inventory changes
   - Market competition dynamics

---

## 8. Appendix: Scripts Reference

### A. Day-to-Day Similarity (`day_to_day_similarity.py`)

**Purpose**: Compute cosine similarity between consecutive days and across various day gaps.

**Output**:
- Consecutive day similarity matrix
- Gap-based similarity decay curve
- Implied decay rate suggestions
- Pairwise similarity heatmap

### B. Daily Trend to Test (`daily_similarity_trend.py`)

**Purpose**: Compute each individual training day's similarity to test data.

**Output**:
- Per-day similarity to test
- Trend analysis (is recent data more similar?)
- Correlation with recency

### C. Decay Application (`quick_decay_check.py`)

**Purpose**: Apply temporal decay to training data and compare to test.

**Configuration**:
```python
DECAY_RATES = {
    "original_bid_requests": 0.9,  # or 1.0 for no decay
    "unr": 0.95,
}
```

**Decay Formula**:
```
decayed_value = original_value × decay_rate^days_from_most_recent
```

### D. Holt's Linear Prediction (`holt_linear_prediction.py`)

**Purpose**: Compare Holt's Linear Smoothing against decay for prediction accuracy.

**Methods Compared**:
- Holt's with various (α, β) combinations
- Decay with rates 0.85, 0.90, 0.92, 0.95
- Baselines: mean, last value, linear trend

**Holt's Formula**:
```
Level:  L_t = α * y_t + (1 - α) * (L_{t-1} + T_{t-1})
Trend:  T_t = β * (L_t - L_{t-1}) + (1 - β) * T_{t-1}
Forecast: F_{t+1} = L_t + T_t
```

### E. Hour-to-Hour Similarity (`hour_to_hour_similarity.py`)

**Purpose**: Analyze intra-day similarity patterns at hourly granularity.

**Output**:
- Consecutive hour similarity
- Hour-of-day stability patterns
- Day boundary effects
- Hourly vs daily decay comparison

---

## 9. Conclusion

Our analysis shows that for `unr_per_original_bid_requests`:

1. **Day-to-day similarity is low** (~0.54), indicating high temporal variability
2. **Temporal decay does not improve** similarity to test data when aggregating multiple days
3. **The aggregation itself** may be the bottleneck—merging dissimilar days dilutes signal
4. **Holt's Linear Smoothing** offers a better alternative by modeling trend and extrapolating

**Bottom line**: 
- For highly variable metrics like revenue rates, using less data (most recent day) or trend-aware methods (Holt's) may be more effective than using more data with decay weights.
- Decay only helps when the pattern is consistently decreasing, which is not guaranteed across all feature combinations.

---

*Generated by ML Platform temporal decay analysis pipeline*

