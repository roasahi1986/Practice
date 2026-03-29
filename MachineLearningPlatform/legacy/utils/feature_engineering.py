from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    when,
    lower,
    split,
    col,
    minute,
    format_string,
    dayofweek,
)
from typing import List

EMPTY_STR = "##empty##"
OTHER_STR = "others"
EMPTY_OS_VERSION = "-1"
TOP_DEV_MAKE_NAME = [
    "apple",
    "samsung",
    "xiaomi",
    "huawei",
    "oppo",
    "motorola",
    "vivo",
    "lenovo",
    "lge",
    "oneplus",
    "realme",
    "sony",
    "tcl",
    "google",
    "hmd global",
    "asus",
    "zte",
    "tecno mobile limited",
    "tecno",
    "honor",
    "infinix",
    "itel",
    "infinix mobility limited",
    "rockchip",
    "blackview",
    "alps",
    "incar",
]
TOP_COUNTRY_VALUES = [
    "us",
    "in",
    "br",
    "jp",
    "id",
    "ru",
    "gb",
    "fr",
    "de",
    "mx",
]
TOP_DEV_MODEL_NAMES = [
    "iphone14,5",
    "iphone12,1",
    "iphone14,7",
    "iphone13,2",
    "iphone15,4",
    "iphone16,2",
    "iphone15,3",
    "iphone14,3",
    "ipad12,1",
    "iphone16,1",
    "iphone11,8",
    "iphone15,2",
    "iphone13,4",
    "iphone14,2",
    "iphone12,8",
    "iphone13,3",
    "iphone14,8",
    "ipad13,18",
    "iphone12,5",
    "iphone15,5",
    "iphone14,6",
    "iphone12,3",
    "iphone13,1",
    "ipad11,6",
    "iphone14,4",
    "ipad7,5",
    "iphone17,2",
    "ipad7,11",
    "iphone17,1",
    "iphone11,6",
    "sm-a155f",
    "sm-a515f",
    "sm-s918b",
    "sm-a528b",
    "iphone11,2",
    "ipad13,16",
    "sm-a546e",
    "iphone10,5",
    "ipad13,1",
    "sm-a146u",
    "sm-a156u",
    "sm-a127f",
    "sm-s911b",
    "iphone17,3",
    "sm-a546b",
    "sm-s928b",
    "sm-a125f",
    "m2003j15sc",
    "sm-a137f",
    "sm-a135f",
    "redmi note 8 pro",
    "sm-s901u",
    "sm-s918u",
    "sm-a325f",
    "iphone10,6",
    "sm-a536b",
    "m2004j19c",
    "ipad6,11",
    "sm-x200",
    "iphone10,4",
    "m2101k6g",
    "redmi note 9 pro",
    "ipad12,2",
    "sm-s911u",
    "sm-a715f",
    "sm-s928u",
    "sm-a217f",
    "sm-s901b",
    "sm-a155m",
    "sm-s908u",
    "redmi note 8",
    "23108rn04y",
    "iphone10,2",
    "iphone10,1",
    "sm-g991b",
    "sm-a055f",
    "infinix x6525",
    "sm-a346e",
    "sm-a556e",
    "cph2269",
    "m2006c3lg",
    "sm-a145r",
    "sm-a145m",
    "tecno bg6",
    "sm-a346b",
    "moto g54 5g",
    "sm-a226b",
    "sm-g781b",
    "sm-a135m",
    "iphone9,4",
    "23106rn0da",
    "sm-g780g",
    "rmx3834",
    "cph2591",
    "sm-a525f",
    "cph2239",
    "sm-a055m",
    "sm-s921u",
    "pixel 7a",
    "iphone9,3",
]


def process_edsp_count(df: DataFrame, target_col: str) -> DataFrame:
    """Process EDSP count by converting target column values > 0 to 1,
    otherwise 0."""
    df = df.withColumn("edsp_bid", when(col(target_col) > 0, 1).otherwise(0))
    return df


def process_req_connection_type(df: DataFrame) -> DataFrame:
    """Process request connection type by handling empty and null values."""
    df = df.withColumn(
        "req_connection_type", df["req_connection_type"].cast("string")
    )
    df = df.withColumn(
        "req_connection_type",
        when(col("req_connection_type") == "", EMPTY_STR).otherwise(
            col("req_connection_type")
        ),
    )
    df = df.fillna(EMPTY_STR, subset=["req_connection_type"])
    return df


def process_req_sdk_version(df: DataFrame) -> DataFrame:
    """Process request SDK version by handling empty and null values."""
    df = df.withColumn("req_sdk_version", df["req_sdk_version"].cast("string"))
    df = df.withColumn(
        "req_sdk_version",
        when(col("req_sdk_version") == "", EMPTY_STR).otherwise(
            col("req_sdk_version")
        ),
    )
    df = df.fillna(EMPTY_STR, subset=["req_sdk_version"])
    return df


def process_req_ad_type(df: DataFrame) -> DataFrame:
    """Process request ad type by handling empty and null values."""
    df = df.withColumn("req_ad_type", df["req_ad_type"].cast("string"))
    df = df.withColumn(
        "req_ad_type",
        when(col("req_ad_type") == "", EMPTY_STR).otherwise(
            col("req_ad_type")
        ),
    )
    df = df.fillna(EMPTY_STR, subset=["req_ad_type"])
    return df


def process_minute_of_hour_02d(df: DataFrame) -> DataFrame:
    """Extract minute of the hour from auction timestamp with zero padding."""
    df = df.withColumn(
        "minute_of_hour",
        format_string("%02d", minute(col("auction_timestamp"))),
    )
    return df


def process_day_of_week(df: DataFrame) -> DataFrame:
    """Extract day of the week from auction timestamp (0 = Sunday,
    6 = Saturday)."""
    df = df.withColumn(
        "day_of_week", (dayofweek(col("auction_timestamp")) - 1)
    )
    return df


def process_req_country_alpha2(df: DataFrame) -> DataFrame:
    """Process request country alpha2 code by converting to
    lowercase and handling non-top countries."""
    df = df.withColumn("req_country_alpha2", lower(col("req_country_alpha2")))
    df = df.withColumn(
        "req_country_alpha2",
        when(
            col("req_country_alpha2").isin(TOP_COUNTRY_VALUES),
            col("req_country_alpha2"),
        ).otherwise(OTHER_STR),
    )
    return df


def process_req_device_model(df: DataFrame) -> DataFrame:
    """Process request device model by converting to lowercase and
    handling non-top models."""
    df = df.withColumn("req_device_model", lower(col("req_device_model")))
    df = df.withColumn(
        "req_device_model",
        when(
            col("req_device_model").isin(TOP_DEV_MODEL_NAMES),
            col("req_device_model"),
        ).otherwise(OTHER_STR),
    )
    return df


def process_do_not_track(df: DataFrame) -> DataFrame:
    """Process do not track flag by handling null values and
    converting to string."""
    df = df.na.fill({"do_not_track": False})
    df = df.withColumn("do_not_track", df["do_not_track"].cast("int"))
    df = df.withColumn("do_not_track", df["do_not_track"].cast("string"))
    return df


def process_geo(df: DataFrame) -> DataFrame:
    """Process geo information by handling null and empty values."""
    df = df.na.fill({"geo": EMPTY_STR})
    df = df.withColumn(
        "geo", when(col("geo") == "", EMPTY_STR).otherwise(col("geo"))
    )
    return df


def process_top_dev_make(df: DataFrame) -> DataFrame:
    """Process top device make by handling null values and
    categorizing non-top makes."""
    df = df.withColumn("top_dev_make", df["dev_make"])
    df = df.na.fill({"top_dev_make": EMPTY_STR})
    df = df.withColumn(
        "top_dev_make",
        when(col("top_dev_make") == "", EMPTY_STR).otherwise(
            col("top_dev_make")
        ),
    )
    df = df.withColumn("top_dev_make", lower(df["top_dev_make"]))
    df = df.withColumn(
        "top_dev_make",
        when(
            df["top_dev_make"].isin(TOP_DEV_MAKE_NAME), df["top_dev_make"]
        ).otherwise(OTHER_STR),
    )
    return df


def process_major_os_version(df: DataFrame) -> DataFrame:
    """Process major OS version by handling null values and extracting
    major version number."""
    df = df.withColumn("major_os_version", df["platform_version"])
    df = df.na.fill({"major_os_version": EMPTY_OS_VERSION})
    df = df.withColumn(
        "major_os_version",
        when(df["major_os_version"] == "", EMPTY_OS_VERSION).otherwise(
            df["major_os_version"]
        ),
    )
    df = df.withColumn(
        "major_os_version", split(df["major_os_version"], "[. ]")[0]
    )
    return df


def process_aro_country_group(df: DataFrame) -> DataFrame:
    """Process ARO country group by handling null values and
    categorizing countries."""
    df = df.withColumn("country_group", df["country"])
    df = df.na.fill({"country_group": EMPTY_STR})
    df = df.withColumn(
        "country_group",
        when(df["country_group"] == "", EMPTY_STR).otherwise(
            df["country_group"]
        ),
    )
    df = df.withColumn(
        "country_group",
        when(df["country_group"] == "US", "US").otherwise("Other"),
    )
    return df


def process_aro_adtype_group(df: DataFrame) -> DataFrame:
    """Process ARO ad type group by handling null values and
    categorizing ad types."""
    df = df.withColumn("ad_type_group", df["ad_type"])
    df = df.na.fill({"ad_type_group": EMPTY_STR})
    df = df.withColumn(
        "ad_type_group",
        when(df["ad_type_group"] == "", EMPTY_STR).otherwise(
            df["ad_type_group"]
        ),
    )
    df = df.withColumn(
        "ad_type_group",
        when(df["ad_type_group"] == "appopen", "video")
        .when(df["ad_type_group"] == "in_line", "video")
        .when(df["ad_type_group"] == "interstitial", "video")
        .when(df["ad_type_group"] == "rewarded", "video")
        .when(df["ad_type_group"] == "mrec", "banner")
        .otherwise(df["ad_type_group"]),
    )
    return df


def process_is_us(df: DataFrame) -> DataFrame:
    """Process country code to identify US vs other countries."""
    reserved_country = ["US"]
    return df.withColumn(
        "req_country_alpha2",
        when(
            col("req_country_alpha2").isin(reserved_country),
            col("req_country_alpha2"),
        ).otherwise("OTHER"),
    )


def process_minute_of_hour(df: DataFrame) -> DataFrame:
    """Extract minute of the hour from auction timestamp."""
    return df.withColumn("minute_of_hour", minute(col("auction_timestamp")))


def process_avg_bid_price(
    df: DataFrame,
    reserved_avg_bid_price: float,
    truncate_avg_bid_price_to: float,
) -> DataFrame:
    """Process average bid price by adding reserve and truncating to
    maximum value."""
    df = df.withColumn(
        "avg_bid_price", col("avg_bid_price") + reserved_avg_bid_price
    )
    df = df.withColumn(
        "avg_bid_price",
        when(
            col("avg_bid_price") > truncate_avg_bid_price_to,
            truncate_avg_bid_price_to,
        ).otherwise(col("avg_bid_price")),
    )
    return df


def process_device_value_group(df: DataFrame, cut_off: int = 10) -> DataFrame:
    """
    Creates a new column 'device_value_group'.
    If country is 'US' and ad_type is 'video', categorizes into:
    - 'high': avg_device_market_price1d_raw > cut_off
    - 'low': 0 < avg_device_market_price1d_raw <= cut_off
    - 'unknown': all other US video cases
    For all other combinations, assigns 'all'.
    """
    df = df.withColumn(
        "device_value_group",
        when(
            (col("country_group") != "US") | (col("ad_type") != "video"), "all"
        )
        .when((col("avg_device_market_price1d_raw") > cut_off), "high")
        .when((col("avg_device_market_price1d_raw") > 0), "low")
        .otherwise("unknown"),
    )
    return df


def create_bin_expression(
    column: str,
    raw_column: str,
    boundaries: List[float],
    labels: List[str],
    unknown_label: str = EMPTY_STR,
    treat_nonpositive_as_unknown: bool = True,
):
    """
    Create a flexible binning expression using custom boundaries and labels

    Args:
        column: Target column name
        raw_column: Source column name
        boundaries: List of bin boundaries
        labels: List of bin labels (should be len(boundaries) + 1)
        unknown_label: Label for null values
        treat_nonpositive_as_unknown: Whether to treat values <= 0 as unknown
    Returns:
        PySpark Column expression for binning
    """
    if len(labels) != len(boundaries) + 1:
        raise ValueError(
            f"Number of labels must be equal to number of boundaries + 1"
        )

    if treat_nonpositive_as_unknown:
        # Treat NULL and values ≤ 0 as unknown
        expr = when(
            col(raw_column).isNull() | (col(raw_column) <= 0), unknown_label
        )
    else:
        # Only treat NULL as unknown
        expr = when(col(raw_column).isNull(), unknown_label)

    # Add boundary conditions
    for i in range(len(boundaries)):
        expr = expr.when(col(raw_column) <= boundaries[i], labels[i])

    # Add the last condition for values greater than the last boundary
    expr = expr.otherwise(labels[-1])

    return expr


def process_time_diff_hours(df: DataFrame) -> DataFrame:
    """
    Calculate the time difference in hours and bin it
    """
    df = df.withColumn(
        "time_diff_hrs",
        (col("timestamp") - col("device_feature_updated_at")) / 3600,
    )

    df = df.withColumn(
        "time_diff_hrs_bin",
        create_bin_expression(
            "time_diff_hrs_bin",
            "time_diff_hrs",
            [24, 72, 168],
            ["(0,24]", "(24,72]", "(72,168]", ">168"],
            unknown_label=OTHER_STR,
            treat_nonpositive_as_unknown=True,
        ),
    )

    return df


def process_top_k_publisher(df: DataFrame, top_k_pub: List[str]) -> DataFrame:
    """Process the data by mapping non-top-K publishers to "other"."""
    df = df.withColumn(
        "pub_app_object_id",
        when(
            col("pub_app_object_id").isin(top_k_pub), col("pub_app_object_id")
        ).otherwise("other"),
    )
    return df


def process_avg_device_market_price1d_raw(df: DataFrame) -> DataFrame:
    """
    Process and bin the average device market price
    """
    df = df.withColumn(
        "avg_device_market_price_bin",
        create_bin_expression(
            "avg_device_market_price_bin",
            "avg_device_market_price1d_raw",
            [1.0, 2.0, 4.0, 8.0, 16.0],
            ["(0,1]", "(1,2]", "(2,4]", "(4,8]", "(8,16]", ">16"],
            unknown_label=EMPTY_STR,
            treat_nonpositive_as_unknown=True,
        ),
    )

    return df


def process_count_of_device_requests1d_raw(df: DataFrame) -> DataFrame:
    """
    Process and bin the count of device requests
    """
    df = df.withColumn(
        "count_of_device_requests_bin",
        create_bin_expression(
            "count_of_device_requests_bin",
            "count_of_device_requests1d_raw",
            [5, 25, 125, 625, 3125],
            [
                "(0,5]",
                "(5,25]",
                "(25,125]",
                "(125,625]",
                "(625,3125]",
                ">3125",
            ],
            unknown_label=EMPTY_STR,
            treat_nonpositive_as_unknown=True,
        ),
    )

    return df


def process_sdk_supply_name(df: DataFrame) -> DataFrame:
    """
    Extract SDK supply name based on pattern matching in sdk_version.

    Maps sdk_version to supply name categories:
    - 'ironsource' if contains 'ironsource'
    - 'admob' if contains 'admob'
    - 'max' if contains 'max'
    - 'others' for all other values
    """
    df = df.withColumn(
        "sdk_supply_name",
        when(col("sdk_version").like("%ironsource%"), "ironsource")
        .when(col("sdk_version").like("%admob%"), "admob")
        .when(col("sdk_version").like("%max%"), "max")
        .otherwise(OTHER_STR),
    )
    return df


def process_prediction_floor(
    df: DataFrame, decimal_places: int = 3
) -> DataFrame:
    """
    Format the prediction_floor and convert to string.

    Args:
        df: Input DataFrame containing prediction_floor
        decimal_places: Number of decimal

    Returns:
        DataFrame with processed prediction_floor
    """
    format_pattern = f"%.{decimal_places}f"
    return df.withColumn(
        "prediction_floor",
        format_string(format_pattern, col("prediction_floor")).cast("string"),
    )


def process_device_os(df: DataFrame) -> DataFrame:
    """Process device os"""
    df = df.na.fill({"device_os": EMPTY_STR})
    df = df.withColumn(
        "device_os",
        when(df["device_os"] == "", EMPTY_STR).otherwise(df["device_os"]),
    )
    return df


def process_supply_name(df: DataFrame) -> DataFrame:
    """Process supply name"""
    df = df.na.fill({"supply_name": OTHER_STR})
    df = df.withColumn(
        "supply_name",
        when(df["supply_name"] == "", OTHER_STR).otherwise(df["supply_name"]),
    )
    return df


def create_unified_features(df):
    """
    Create unified var_hbedsp_highest_price_7d column 
    based on placement_type
    and remove original placement-specific columns.
    - For known placement types, uses the specific column
    - For null or unknown placement types, uses 0.0
    """
    placement_mapping = {
        "app_open": "var_hbedsp_highest_price_app_open7d",
        "rewarded": "var_hbedsp_highest_price_rewarded7d",
        "interstitial": "var_hbedsp_highest_price_interstitial7d",
        "native": "var_hbedsp_highest_price_native7d",
        "banner": "var_hbedsp_highest_price_banner7d",
        "mrec": "var_hbedsp_highest_price_mrec7d",
        "in_line": "var_hbedsp_highest_price_in_line7d",
    }

    result_df = df
    columns_to_drop = []

    case_expr = None
    for placement_type, original_col in placement_mapping.items():
        if original_col in df.columns:
            if original_col not in columns_to_drop:
                columns_to_drop.append(original_col)

            condition = col("placement_type") == placement_type
            if case_expr is None:
                case_expr = when(condition, col(original_col))
            else:
                case_expr = case_expr.when(condition, col(original_col))

    if case_expr is not None:
        # Use 0.0 as default for null or unknown placement types
        result_df = result_df.withColumn(
            "var_hbedsp_highest_price_7d", case_expr.otherwise(0.0)
        )

    if columns_to_drop:
        result_df = result_df.drop(*columns_to_drop)

    return result_df


def process_var_hbedsp_highest_price_7d(df: DataFrame) -> DataFrame:
    """
    Process and bin the var hbedsp highest price 7d
    """
    df = df.withColumn(
        "var_hbedsp_highest_price_7d_bin",
        create_bin_expression(
            "var_hbedsp_highest_price_7d_bin",
            "var_hbedsp_highest_price_7d",
            [0.001058551445165888, 0.00733071019413941, 0.12203971999280194],
            [
                "(0,0.001058551445165888]",
                "(0.001058551445165888,0.00733071019413941]",
                "(0.00733071019413941,0.12203971999280194]",
                ">0.12203971999280194",
            ],
            unknown_label=EMPTY_STR,
            treat_nonpositive_as_unknown=True,
        ),
    )
    return df


def calculate_weighted_threshold(
    df: DataFrame,
    source_col: str,
    weight_col: str = "bid_requests",
) -> float:
    """
    Calculate weighted average threshold for a numeric column.

    Args:
        df: Input DataFrame
        source_col: Source column name to calculate threshold for
        weight_col: Column to use as weight for computing the weighted average.

    Returns:
        Weighted average threshold value
    """
    from pyspark.sql import functions as F

    stats = df.select(
        F.sum(F.col(source_col) * F.col(weight_col)).alias("weighted_sum"),
        F.sum(F.col(weight_col)).alias("total_weight"),
    ).collect()[0]

    weighted_sum = stats["weighted_sum"] or 0.0
    total_weight = stats["total_weight"] or 1.0
    threshold = weighted_sum / total_weight if total_weight > 0 else 0.0

    return threshold


def bucketize_by_threshold(
    df: DataFrame,
    source_col: str,
    threshold: float,
    output_col: str = None,
) -> DataFrame:
    """
    Bucketize a numeric column into below_threshold and above_threshold bins based on threshold.

    Args:
        df: Input DataFrame
        source_col: Source column name to bucketize
        threshold: Threshold value for binning
        output_col: Output column name. If None, defaults to {source_col}_bin

    Returns:
        DataFrame with new bin column added
    """
    from pyspark.sql import functions as F

    if output_col is None:
        output_col = f"{source_col}_bin"

    # Print summary
    stats = df.select(
        F.min(source_col).alias("min_val"),
        F.max(source_col).alias("max_val"),
        F.avg(source_col).alias("avg_val"),
        F.stddev(source_col).alias("stddev_val"),
        F.count(F.when(F.col(source_col) > 0, 1)).alias("non_zero_count"),
        F.count("*").alias("total_count"),
    ).collect()[0]

    print(f"\n{'=' * 60}")
    print(f"Bucketize Feature: {source_col}")
    print(f"{'=' * 60}")
    print(f"  Min: {stats['min_val']}")
    print(f"  Max: {stats['max_val']}")
    print(f"  Simple Avg: {stats['avg_val']:.6f}")
    print(f"  Stddev: {stats['stddev_val']:.6f}" if stats['stddev_val'] else "  Stddev: N/A")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Non-zero values: {stats['non_zero_count']:,} / {stats['total_count']:,}")
    print(f"  Output Column: {output_col}")
    print(f"{'=' * 60}\n")

    # Bucketize into below_avg and above_avg bins
    df = df.withColumn(
        output_col,
        when(col(source_col).isNull() | (col(source_col) <= 0), EMPTY_STR)
        .when(col(source_col) <= threshold, "below_threshold")
        .otherwise("above_threshold"),
    )

    return df


def calculate_robust_normalization_stats_by_group(
    df: DataFrame,
    source_col: str,
    group_cols: List[str],
) -> DataFrame:
    """
    Calculate robust normalization statistics (median and IQR) for a numeric column,
    grouped by specified columns. Uses log-transform to handle log-normal distributions
    common in spend/revenue metrics.

    The normalization formula is:
        log_x = log(x + 1)
        normalized = (log_x - median(log_x)) / IQR(log_x)

    This approach is robust to:
    - Seasonal volatility: centering by median removes baseline shifts
    - Outliers: log-transform compresses extreme values, IQR is robust to outliers
    - Cross-group comparability: each group normalized to approximately the same scale

    Args:
        df: Input DataFrame
        source_col: Source column name to calculate statistics for
        group_cols: List of column names to group by

    Returns:
        DataFrame with group columns and normalization statistics:
        - {source_col}_log_median: median of log-transformed values
        - {source_col}_log_iqr: IQR (Q3-Q1) of log-transformed values
    """
    from pyspark.sql import functions as F

    # Filter out null and non-positive values
    valid_df = df.filter(
        F.col(source_col).isNotNull() & (F.col(source_col) > 0)
    )

    # Apply log transform: log(x + 1) to handle values close to 0
    log_col = f"_log_{source_col}"
    valid_df = valid_df.withColumn(log_col, F.log1p(F.col(source_col)))

    # Calculate median (Q2), Q1, and Q3 using approx_percentile
    stats_df = valid_df.groupBy(*group_cols).agg(
        F.percentile_approx(
            F.col(log_col),
            [0.25, 0.5, 0.75],
            10000,  # accuracy parameter
        ).alias("_quantiles"),
        F.count("*").alias(f"{source_col}_norm_count"),
    )

    # Extract Q1, median, Q3 and compute IQR
    stats_df = stats_df.withColumn(
        f"{source_col}_log_q1", F.col("_quantiles").getItem(0)
    ).withColumn(
        f"{source_col}_log_median", F.col("_quantiles").getItem(1)
    ).withColumn(
        f"{source_col}_log_q3", F.col("_quantiles").getItem(2)
    ).withColumn(
        f"{source_col}_log_iqr",
        F.col(f"{source_col}_log_q3") - F.col(f"{source_col}_log_q1"),
    )

    # Handle edge case: if IQR is 0 (all values same), use a small epsilon
    # to avoid division by zero, or fallback to stddev-based scaling
    min_iqr = 0.001  # Minimum IQR to prevent division by very small numbers
    stats_df = stats_df.withColumn(
        f"{source_col}_log_iqr",
        F.when(
            F.col(f"{source_col}_log_iqr") < min_iqr, min_iqr
        ).otherwise(F.col(f"{source_col}_log_iqr")),
    )

    # Drop temporary columns
    stats_df = stats_df.drop(
        "_quantiles", f"{source_col}_log_q1", f"{source_col}_log_q3"
    )

    return stats_df


def normalize_by_robust_stats(
    df: DataFrame,
    stats_df: DataFrame,
    source_col: str,
    group_cols: List[str],
    output_col: str = None,
) -> DataFrame:
    """
    Apply robust normalization to a numeric column using pre-computed statistics.

    The normalization formula is:
        log_x = log(x + 1)
        normalized = (log_x - median) / IQR

    This produces values centered around 0, where:
    - 0 means the value equals the group median
    - Positive values are above median
    - Negative values are below median
    - The magnitude is in units of IQR (similar to z-score but robust)

    Args:
        df: Input DataFrame
        stats_df: DataFrame containing normalization statistics from
                  calculate_robust_normalization_stats_by_group
        source_col: Source column name to normalize
        group_cols: List of column names used for joining with stats_df
        output_col: Output column name. If None, defaults to {source_col}_normalized

    Returns:
        DataFrame with new normalized column added (original column preserved)
    """
    from pyspark.sql import functions as F

    if output_col is None:
        output_col = f"{source_col}_normalized"

    median_col = f"{source_col}_log_median"
    iqr_col = f"{source_col}_log_iqr"
    count_col = f"{source_col}_norm_count"

    # Join with statistics DataFrame
    df = df.join(stats_df, on=group_cols, how="left")

    # Apply normalization:
    # 1. For null/non-positive values -> 0 (neutral, no signal)
    # 2. For groups without stats -> 0 (no reference for normalization)
    # 3. Otherwise: (log(x+1) - median) / IQR
    df = df.withColumn(
        output_col,
        when(
            F.col(source_col).isNull() | (F.col(source_col) <= 0), 0.0
        ).when(
            F.col(median_col).isNull(), 0.0
        ).otherwise(
            (F.log1p(F.col(source_col)) - F.col(median_col)) / F.col(iqr_col)
        ),
    )

    # Drop the statistics columns after use
    df = df.drop(median_col, iqr_col, count_col)

    return df


def calculate_quantile_thresholds_by_group(
    df: DataFrame,
    source_col: str,
    group_cols: List[str],
    n_buckets: int = 3,
) -> DataFrame:
    """
    Calculate quantile-based thresholds for a numeric column, grouped by specified columns.

    Uses approx_percentile for efficient single-pass computation of all thresholds.

    Args:
        df: Input DataFrame
        source_col: Source column name to calculate thresholds for
        group_cols: List of column names to group by
        n_buckets: Number of buckets to split the data into (default: 5)

    Returns:
        DataFrame with group columns and threshold columns (threshold_1, threshold_2, ..., threshold_{n-1})
    """
    from pyspark.sql import functions as F

    # Filter out null and non-positive values for quantile calculation
    valid_df = df.filter(
        F.col(source_col).isNotNull() & (F.col(source_col) > 0)
    )

    # For a Log-Normal distribution (or any right-skewed distribution), the 'head' (lower values)
    # typically contains the bulk of the population but spans a small value range, while the 'tail'
    # spans a large value range but contains few data points.
    #
    # Standard linear quantiles (0.2, 0.4...) divide the population equally.
    # To increase resolution at the lower end, we apply a convex power-law transformation:
    #     p_i = (i / n_buckets) ^ k  (where k > 1)
    #
    # With k=2 (Quadratic), the bin sizes (probability masses) grow linearly:
    # For n=5: 0.04, 0.16, 0.36, 0.64 -> Bins: 4%, 12%, 20%, 28%, 36%.
    
    skew_factor = 3.0
    quantile_points = [pow(i / n_buckets, skew_factor) for i in range(1, n_buckets)]

    # Use approx_percentile to calculate all thresholds in a single aggregation
    # This is much more efficient than multiple window operations
    threshold_df = valid_df.groupBy(*group_cols).agg(
        F.percentile_approx(
            F.col(source_col),
            quantile_points,
            10000,  # accuracy parameter
        ).alias("_thresholds")
    )

    # Extract individual threshold columns from the array
    for i in range(1, n_buckets):
        threshold_col = f"{source_col}_threshold_{i}"
        threshold_df = threshold_df.withColumn(
            threshold_col,
            F.col("_thresholds").getItem(i - 1),
        )

    # Drop the temporary array column
    threshold_df = threshold_df.drop("_thresholds")

    return threshold_df


def bucketize_by_quantile_thresholds_df(
    df: DataFrame,
    threshold_df: DataFrame,
    source_col: str,
    group_cols: List[str],
    n_buckets: int = 3,
    output_col: str = None,
) -> DataFrame:
    """
    Bucketize a numeric column into N buckets based on per-group quantile thresholds.

    Args:
        df: Input DataFrame
        threshold_df: DataFrame containing group columns and threshold values
                      (threshold_1, threshold_2, ..., threshold_{n-1})
        source_col: Source column name to bucketize
        group_cols: List of column names used for joining with threshold_df
        n_buckets: Number of buckets (must match the thresholds in threshold_df)
        output_col: Output column name. If None, defaults to {source_col}_bin

    Returns:
        DataFrame with new bin column added (bucket_1, bucket_2, ..., bucket_n)
    """
    from pyspark.sql import functions as F

    if output_col is None:
        output_col = f"{source_col}_bin"

    threshold_cols = [f"{source_col}_threshold_{i}" for i in range(1, n_buckets)]

    # Join with threshold DataFrame
    df = df.join(threshold_df, on=group_cols, how="left")

    # Check if any threshold is missing (group not in threshold_df or sparse data)
    first_threshold = threshold_cols[0]
    has_thresholds = F.col(first_threshold).isNotNull()

    # Build the bucket assignment expression
    # Start with null/non-positive source value check
    bucket_expr = when(col(source_col).isNull() | (col(source_col) <= 0), "bucket_1")

    # Handle case where thresholds are missing (group had no valid data)
    bucket_expr = bucket_expr.when(~has_thresholds, "bucket_1")

    # Add conditions for each bucket
    for i, threshold_col in enumerate(threshold_cols):
        bucket_name = f"bucket_{i + 1}"
        bucket_expr = bucket_expr.when(
            col(source_col) <= col(threshold_col), bucket_name
        )

    # Last bucket for values above all thresholds
    bucket_expr = bucket_expr.otherwise(f"bucket_{n_buckets}")

    df = df.withColumn(output_col, bucket_expr)

    # Drop the threshold columns after use
    for threshold_col in threshold_cols:
        df = df.drop(threshold_col)

    return df


def bucketize_by_fixed_thresholds(
    df: DataFrame,
    source_col: str,
    boundaries: List[float],
    labels: List[str] = None,
    output_col: str = None,
    null_label: str = "unknown",
) -> DataFrame:
    """
    Bucketize a numeric column using fixed, predefined thresholds.
    
    This is ideal for normalized values (e.g., z-scores, IQR-normalized) where
    the thresholds have semantic meaning that should be consistent across
    training and inference.
    
    Example usage for IQR-normalized attention scores:
        boundaries = [-1.5, -0.5, 0.5, 1.5]
        labels = ["very_low", "below_avg", "typical", "above_avg", "very_high"]
    
    Interpretation:
        - very_low: <= -1.5 IQRs (bottom tail, unusually low attention)
        - below_avg: -1.5 to -0.5 IQRs (below median)
        - typical: -0.5 to 0.5 IQRs (around the median)
        - above_avg: 0.5 to 1.5 IQRs (above median)
        - very_high: > 1.5 IQRs (top tail, unusually high attention)
    
    Args:
        df: Input DataFrame
        source_col: Source column name (should be normalized/standardized)
        boundaries: List of threshold values in ascending order
                    Creates len(boundaries) + 1 buckets
        labels: Optional list of bucket labels. If None, uses bucket_1, bucket_2, ...
                Length must equal len(boundaries) + 1
        output_col: Output column name. If None, defaults to {source_col}_bin
        null_label: Label for null/NaN values
    
    Returns:
        DataFrame with new bin column added
    """
    from pyspark.sql import functions as F
    
    if output_col is None:
        output_col = f"{source_col}_bin"
    
    n_buckets = len(boundaries) + 1
    
    # Generate default labels if not provided
    if labels is None:
        labels = [f"bucket_{i+1}" for i in range(n_buckets)]
    
    if len(labels) != n_buckets:
        raise ValueError(
            f"Number of labels ({len(labels)}) must equal "
            f"number of buckets ({n_buckets} = len(boundaries) + 1)"
        )
    
    # Build the bucket assignment expression
    # Handle null values first
    bucket_expr = when(col(source_col).isNull(), null_label)
    
    # Add conditions for each bucket boundary
    for i, boundary in enumerate(boundaries):
        bucket_expr = bucket_expr.when(col(source_col) <= boundary, labels[i])
    
    # Last bucket for values above the highest boundary
    bucket_expr = bucket_expr.otherwise(labels[-1])
    
    df = df.withColumn(output_col, bucket_expr)
    
    return df


def generate_normalized_buckets(
    n_buckets: int,
    min_val: float = -2.5,
    max_val: float = 2.5,
) -> tuple:
    """
    Generate evenly-spaced bucket boundaries for normalized values.
    
    For IQR-normalized values, the default range [-2.5, 2.5] covers ~98% of 
    data for most distributions. Values beyond this go to edge buckets.
    
    Args:
        n_buckets: Number of buckets to create
        min_val: Minimum value of the range (default: -2.5)
        max_val: Maximum value of the range (default: 2.5)
    
    Returns:
        Tuple of (boundaries, labels)
        - boundaries: List of n_buckets - 1 threshold values
        - labels: List of n_buckets labels ["bucket_1", "bucket_2", ...]
    
    Example:
        boundaries, labels = generate_normalized_buckets(10)
        # boundaries = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        # labels = ["bucket_1", ..., "bucket_10"]
    """
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2")
    
    step = (max_val - min_val) / n_buckets
    boundaries = [min_val + step * (i + 1) for i in range(n_buckets - 1)]
    labels = [f"bucket_{i+1}" for i in range(n_buckets)]
    
    return boundaries, labels


def generate_share_buckets(
    n_buckets: int,
    power: float = 0.5,
) -> tuple:
    """
    Generate bucket boundaries for share/ratio values in [0, 1].
    
    Uses power-law spacing for finer granularity at lower values
    (where most data points typically lie in share distributions).
    
    Args:
        n_buckets: Number of buckets to create
        power: Power for spacing (default: 0.5). Lower = more resolution at low end.
    
    Returns:
        Tuple of (boundaries, labels)
    
    Example:
        boundaries, labels = generate_share_buckets(10)
    """
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2")
    
    boundaries = [pow((i + 1) / n_buckets, power) for i in range(n_buckets - 1)]
    labels = [f"bucket_{i+1}" for i in range(n_buckets)]
    
    return boundaries, labels
