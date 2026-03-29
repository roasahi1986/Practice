import datetime
from typing import Dict, Any, List
import json

import boto3
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from databricks.sdk.runtime import dbutils

from machine_learning.util.aws_s3 import ModelFileHelper


def initialize_s3_helper(config: Dict[str, Any]) -> ModelFileHelper:
    """Initialize S3 client and file helper for model operations."""
    s3_client = boto3.client("s3", region_name=config["infra.aws.s3.region"])
    file_helper = ModelFileHelper(s3_client)
    
    print(f"🔗 S3 client initialized for region: {config['infra.aws.s3.region']}")
    return file_helper


def check_model_existence(
    file_helper: ModelFileHelper,
    config: Dict[str, Any],
    task,
    training_data_end_datetime: datetime.datetime,
    date_format: str = "%Y-%m-%d",
) -> None:
    """Check if model already exists to avoid redundant training."""
    training_data_end_date = training_data_end_datetime.strftime(date_format)
    
    model_exists = file_helper.model_file_already_existed(
        config["infra.aws.s3.bucket"],
        f"{task.output_data_path}/{task.project}",
        task.model.name,
        task.model.version,
        training_data_end_date,
    )
    
    if model_exists:
        print(f"⚠️ Model already exists for date: {training_data_end_date}")
        dbutils.notebook.exit(
            f"Model data already generated for {training_data_end_date}. Training skipped."
        )
    else:
        print(f"✅ No existing model found for Task {task.id}. Proceeding with training.")


def validate_input_data_availability(
    file_helper: ModelFileHelper,
    config: Dict[str, Any],
    input_data_path: str,
    training_data_begin_datetime: datetime.datetime,
    training_data_end_datetime: datetime.datetime,
    date_format: str = "%Y-%m-%d",
) -> List[str]:
    """Validate that all required input data is available for the training period."""
    input_bucket = config["infra.aws.s3.bucket"]
    data_paths = []
    
    print("🔍 Validating input data availability...")
    
    # Check each day in the training period
    for dt in pd.date_range(
        training_data_begin_datetime,
        training_data_end_datetime - datetime.timedelta(days=1),
    ):
        path = f"{input_data_path}/dt={dt.strftime(date_format)}"
        s3_path = path.split(input_bucket + "/")[1]
        
        if not file_helper.success_file_exist(input_bucket, s3_path):
            raise ValueError(f"❌ Data not ready at: {path}")
        
        print(f"✅ Data validated: {dt.strftime(date_format)}")
        data_paths.append(path)
    
    print(f"🎯 All {len(data_paths)} data partitions validated successfully")
    return data_paths


def save_model_to_s3(
    model,
    s3_path,
) -> None:
    """Save trained model to S3 storage."""
    print("💾 Saving model to S3...")
    
    model_data = model.to_model_data()

    # Convert the model data to a JSON string and then encode it to bytes
    model_data_json = json.dumps(model_data).encode('utf-8')

    # Initialize a session using Amazon S3
    s3 = boto3.client('s3')

    # Define the S3 bucket and path
    bucket_name = 'exchange-dev'

    print(f"   Bucket: {bucket_name}")
    print(f"   Path: {s3_path}")
    print(f"   Model version: v{model.version}")

    # Upload the JSON string to the specified S3 path
    if s3.put_object(Bucket=bucket_name, Key=s3_path, Body=model_data_json):
        print(f"✅ Model successfully saved")
    else:
        raise RuntimeError(f"❌ Failed to save model")

def load_model_from_s3(
    s3_path,
):
    """Load model from S3 storage."""
    print("💾 Loading model from S3...")

    # Read the model data from S3
    s3 = boto3.client('s3')

    bucket_name = 'exchange-dev'
    response = s3.get_object(Bucket=bucket_name, Key=s3_path)
    model_data_json = response['Body'].read().decode('utf-8')
    model_data = json.loads(model_data_json)
    return model_data


def load_and_prepare_raw_data(
    input_data_path: str, 
    input_data_paths: List[str], 
    rename_columns: Dict[str, str]
) -> DataFrame:
    """Load raw parquet data and apply initial transformations."""
    print("📊 Loading raw training data...")
    
    # Load data from validated paths
    spark = SparkSession.builder.getOrCreate()
    raw_df = spark.read.option("basePath", input_data_path).parquet(*input_data_paths)
    
    # Apply column renaming
    raw_df = raw_df.withColumnsRenamed(rename_columns)
    
    print(f"✅ Raw data loaded with {raw_df.count():,} records")
    return raw_df


def calculate_uplift_ratio(
    df: DataFrame,
    feature_columns: List[str],
    exp_bucket_col: str,
    control_group: str,
    explore_group: List[str],
    metric_col: str,
) -> DataFrame:
    """
    Calculate uplift ratio between explore and control groups for a given metric.

    Uplift ratio = (explore_sum - control_sum) / control_sum

    Args:
        df: Input DataFrame with feature columns, exp_bucket, and metric column
        feature_columns: List of feature column names to group by
        exp_bucket_col: Column name containing experiment bucket names
        control_group: Name of the control group in exp_bucket_col
        explore_group: List of explore group names in exp_bucket_col
        metric_col: Column name for the metric to calculate uplift ratio

    Returns:
        DataFrame with feature columns and {metric_col}_uplift_ratio
    """
    from pyspark.sql import functions as F

    # Filter to only control and explore groups
    df = df.filter(F.col(exp_bucket_col).isin([control_group] + explore_group))

    # Aggregate by feature columns and exp_bucket group
    aggregated_df = df.groupBy(feature_columns + [exp_bucket_col]).agg(
        F.sum(metric_col).alias("_metric_sum"),
    )

    # Pivot to get control and explore values as separate columns
    pivoted_df = aggregated_df.groupBy(feature_columns).pivot(
        exp_bucket_col, [control_group] + explore_group
    ).agg(
        F.first("_metric_sum"),
    )

    # Rename control column and sum all explore group columns
    pivoted_df = pivoted_df.withColumnRenamed(control_group, "_control_sum")
    pivoted_df = pivoted_df.withColumn(
        "_explore_sum",
        sum(F.coalesce(F.col(g), F.lit(0)) for g in explore_group)
    )

    # Calculate uplift ratio: (explore - control) / control
    output_col = f"{metric_col}_uplift_ratio"
    result_df = pivoted_df.withColumn(
        output_col,
        F.when(
            F.col("_control_sum").isNull() | (F.col("_control_sum") == 0),
            F.lit(0.0)
        ).otherwise(
            (F.col("_explore_sum") - F.col("_control_sum")) / F.col("_control_sum")
        )
    )

    # Keep only feature columns and the uplift ratio
    result_df = result_df.select(feature_columns + [output_col])

    return result_df


def validate_dataframe_not_empty(df: DataFrame, df_name: str = "DataFrame") -> None:
    """Validate that a DataFrame is not empty."""
    if df.count() == 0:
        raise ValueError(f"❌ {df_name} is empty")
    print(f"✅ {df_name} validation passed")


def print_dataframe_summary(df: DataFrame, df_name: str = "DataFrame") -> None:
    """Print a summary of DataFrame statistics."""
    count = df.count()
    columns = len(df.columns)
    
    print(f"📊 {df_name} Summary:")
    print(f"   Records: {count:,}")
    print(f"   Columns: {columns}")
    
    if count > 0:
        print(f"   Sample data:")
        df.show(5, truncate=False)

def save_threshold_model_to_s3(model_data_json, s3_path) -> None:
    """Save threshold model to S3 storage."""
    print("💾 Saving threshold model to S3...")

    # Initialize a session using Amazon S3
    s3 = boto3.client('s3')

    # Define the S3 bucket and path
    bucket_name = 'exchange-dev'

    print(f"   Bucket: {bucket_name}")
    print(f"   Path: {s3_path}")

    # Upload the JSON string to the specified S3 path
    if s3.put_object(Bucket=bucket_name, Key=s3_path, Body=model_data_json):
        print(f"✅ Model successfully saved")
    else:
        raise RuntimeError(f"❌ Failed to save model")

def load_threshold_model_from_s3(s3_path) -> str:
    """Load threshold model from S3 storage."""
    print("💾 Loading threshold model from S3...")

    # Read the threshold model data from S3
    s3 = boto3.client('s3')

    bucket_name = 'exchange-dev'
    response = s3.get_object(Bucket=bucket_name, Key=s3_path)
    model_data = response['Body'].read().decode('utf-8')

    return model_data
