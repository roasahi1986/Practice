from typing import List

from pyspark.sql import DataFrame, functions as F

# Databricks display function - fallback to show() for local execution
try:
    from databricks.sdk.runtime import display
except ImportError:
    display = lambda df: df.show()


def apply_temporal_decay(
    df: DataFrame,
    columns: List[str],
    date_column: str = "dt",
    decay_rate: float = 0.9
) -> DataFrame:
    """
    Apply exponential temporal decay to specified columns based on date.
    
    The most recent day has no decay (factor=1.0), and each preceding day
    is multiplied by decay_rate^(days_from_last).
    
    Args:
        df: Input DataFrame with a date column
        columns: List of column names to apply decay to
        date_column: Name of the date column (default: "dt")
        decay_rate: Decay rate per day (default: 0.9)
    
    Returns:
        DataFrame with decayed column values
    """
    max_date = df.agg(F.max(date_column)).collect()[0][0]
    
    df = df.withColumn(
        "_days_from_last",
        F.datediff(F.to_date(F.lit(max_date)), F.to_date(F.col(date_column)))
    ).withColumn(
        "_decay_factor",
        F.pow(F.lit(decay_rate), F.col("_days_from_last"))
    )
    
    for col_name in columns:
        df = df.withColumn(col_name, F.col(col_name) * F.col("_decay_factor"))
    
    df = df.drop("_days_from_last", "_decay_factor")
    
    return df


def extract_top_k_publishers(
    df: DataFrame, 
    top_k_pub_size: int,
    exp_bucket_control: int = 0
) -> List[str]:
    """Extract top-K publishers based on UNR performance from control group."""
    print("🏆 Identifying top-K publishers...")
    
    # Aggregate publisher performance from control group
    publisher_aggregation = (
        df.filter(F.col("experiment_id") == exp_bucket_control)
        .groupBy(["pub_app_object_id"])
        .agg(
            F.sum("original_bid_requests").alias("original_bid_requests"),
            F.sum("unr").alias("unr"),
        )
        .withColumn(
            "unr_per_original_br",
            F.col("unr") * 1_000_000_000 / F.col("original_bid_requests"),
        )
    )
    
    top_k_publishers = (
        publisher_aggregation
        .orderBy(F.col("unr").desc())
        .limit(top_k_pub_size)
        .select("pub_app_object_id")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    
    print(f"✅ Identified {len(top_k_publishers)} top publishers (K={top_k_pub_size})")
    return top_k_publishers

def extract_top_k_publishers_old(
    df: DataFrame, 
    top_k_pub_size: int,
    exp_bucket_control: str
) -> List[str]:
    """Extract top-K publishers based on UNR performance from control group."""
    print("🏆 Identifying top-K publishers...")
    
    # Aggregate publisher performance from control group
    publisher_aggregation = (
        df.filter(F.col("exp_bucket") == exp_bucket_control)
        .groupBy(["pub_app_object_id"])
        .agg(
            F.sum("bid_requests").alias("bid_requests"),
            F.sum("multiplied_bid_requests").alias("multiplied_bid_requests"),
            F.sum("unr").alias("unr"),
        )
        .withColumn(
            "original_bid_requests",
            F.col("bid_requests") - F.col("multiplied_bid_requests"),
        )
        .withColumn(
            "unr_per_original_br",
            F.col("unr") * 1_000_000_000 / F.col("original_bid_requests"),
        )
    )
    
    top_k_publishers = (
        publisher_aggregation
        .orderBy(F.col("unr").desc())
        .limit(top_k_pub_size)
        .select("pub_app_object_id")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    
    print(f"✅ Identified {len(top_k_publishers)} top publishers (K={top_k_pub_size})")
    return top_k_publishers

def prepare_experiment_data(df: DataFrame, feature_columns: List[str], control_bucket: int = 0, explore_buckets: List[int] = None) -> DataFrame:
    # Filter to only include control and explore experiment IDs
    valid_experiment_ids = [control_bucket] + explore_buckets
    df_filtered = df.filter(F.col("experiment_id").isin(valid_experiment_ids))
    
    # Add a column to distinguish control vs explore groups
    df_with_group = df_filtered.withColumn(
        "is_control", 
        F.when(F.col("experiment_id") == control_bucket, True).otherwise(False)
    )
    
    # Aggregate by selected features and control/explore group
    aggregated_df = (
        df_with_group.groupBy(feature_columns + ["is_control"])
        .agg(
            F.sum("original_bid_requests").alias("original_bid_requests"),
            F.sum("unr").alias("unr"),
        )
        .withColumn(
            "unr_per_original_br",
            F.col("unr") * 1_000_000_000 / F.col("original_bid_requests"),
        )
    )
    
    print(f"✅ Data aggregated by {len(feature_columns)} features with control/explore grouping")
    return aggregated_df

def prepare_experiment_data_old(df: DataFrame, feature_columns: List[str], control_bucket: str, explore_buckets: List[str] = None) -> DataFrame:
    # Filter to only include control and explore experiment IDs
    valid_experiment_ids = [control_bucket] + explore_buckets
    df_filtered = df.filter(F.col("exp_bucket").isin(valid_experiment_ids))
    
    # Add a column to distinguish control vs explore groups
    df_with_group = df_filtered.withColumn(
        "is_control", 
        F.when(F.col("exp_bucket") == control_bucket, True).otherwise(False)
    )
    
    # Aggregate by selected features and control/explore group
    aggregated_df = (
        df_with_group.groupBy(feature_columns + ["is_control"])
        .agg(
            F.sum("bid_requests").alias("bid_requests"),
            F.sum("multiplied_bid_requests").alias("multiplied_bid_requests"),
            F.sum("unr").alias("unr"),
        )
        .withColumn(
            "original_bid_requests",
            F.col("bid_requests") - F.col("multiplied_bid_requests"),
        )
        .filter(F.col("original_bid_requests") > 0)
        .withColumn(
            "unr_per_original_br",
            F.col("unr") * 1_000_000_000 / F.col("original_bid_requests"),
        )
    )
    
    print(f"✅ Data aggregated by {len(feature_columns)} features with control/explore grouping")
    return aggregated_df

def create_training_dataset(df: DataFrame, feature_columns: List[str]) -> DataFrame:
    """Create training dataset by comparing explore vs control groups."""
    print("📈 Creating training dataset with uplift calculations...")
    
    # Separate control and explore groups based on is_control flag
    control_df = df.filter(F.col("is_control") == True)
    explore_df = df.filter(F.col("is_control") == False)
    
    print(f"   Control group records: {control_df.count():,}")
    print(f"   Explore group records: {explore_df.count():,}")
    
    # Join and calculate uplift
    training_df = (
        explore_df.alias("explore")
        .join(
            control_df.alias("control"),
            on=feature_columns,
            how="inner",
        )
        .withColumn(
            "unr_uplift_per_original_br",
            F.col("explore.unr_per_original_br") - F.col("control.unr_per_original_br"),
        )
    )
    
    training_df.persist()
    training_records = training_df.count()
    print(f"✅ Training dataset created with {training_records:,} records")
    
    return training_df


def prepare_training_features(
    df: DataFrame, 
    feature_columns: List[str],
    target_column: str,
    weight_column: str
) -> DataFrame:
    """Prepare final training dataset with proper column naming."""
    print("🔧 Preparing training features...")
    
    # Select required columns
    selected_columns = feature_columns + [target_column, weight_column]
    
    # Prepare final dataset with standardized column names
    training_data = df.select(selected_columns).withColumnsRenamed({
        weight_column.split(".")[1]: "weight",
        target_column: "y",
    })
    
    # Cache for multiple operations
    training_data.persist()
    
    record_count = training_data.count()
    print(f"✅ Training dataset prepared with {record_count:,} records")

    y_summary = training_data.select("y").describe().toPandas()
    print("\n🔎 y summary statistics:")
    print(y_summary)
    
    # Display sample data for validation
    print("\n📋 Sample training data:")
    display(training_data.head(10))
    print("\n")
    
    return training_data


def calculate_unr_metrics(df: DataFrame, group_col: str = "experiment_id") -> DataFrame:
    """Calculate UNR metrics for different experiment groups."""
    return (
        df.groupBy([group_col])
        .agg(
            F.sum("original_bid_requests").alias("total_bid_requests"),
            F.sum("unr").alias("total_unr"),
            F.count("*").alias("record_count"),
        )
        .withColumn(
            "unr_per_bid_request",
            F.col("total_unr") / F.col("total_bid_requests"),
        )
        .orderBy(group_col)
    )


def validate_experiment_buckets(
    df: DataFrame, 
    expected_control: int = 0, 
    expected_explore: List[int] = None
) -> None:
    """Validate that expected experiment buckets exist in the data."""
    if expected_explore is None:
        expected_explore = [17, 3]
    
    unique_buckets = df.select("experiment_id").distinct().rdd.flatMap(lambda x: x).collect()
    unique_buckets = sorted(unique_buckets)
    
    print(f"📊 Found experiment buckets: {unique_buckets}")
    
    if expected_control not in unique_buckets:
        raise ValueError(f"❌ Control bucket {expected_control} not found in data")
    
    missing_explore = [bucket for bucket in expected_explore if bucket not in unique_buckets]
    if missing_explore:
        print(f"⚠️ Warning: Some explore buckets missing: {missing_explore}")
    
    print("✅ Experiment bucket validation completed")

def print_training_log(model) -> None:
    print("📋 RTB Factorization Machine Parameters:")
    print(f"   embedding_dim: {model.embedding_dim}")
    print(f"   include_linear: {model.include_linear}")
    print(f"   epochs: {model.epochs}")
    print(f"   batch_size: {model.batch_size}")
    print(f"   learning_rate: {model.learning_rate}")
    print(f"   weight_decay: {model.weight_decay}")
    print(f"   dropout_rate: {model.dropout_rate}")

    print("\n📈 Training losses")
    for idx, loss in enumerate(model.training_history["train_losses"], start=1):
        if idx > 1:
            prev_loss = model.training_history["train_losses"][idx - 2]
            percent_drop = ((prev_loss - loss) / prev_loss) * 100
            print(f"   epoch {idx}: {loss} ({percent_drop:.2f}% drop from previous epoch)")
        else:
            print(f"   epoch {idx}: {loss}")