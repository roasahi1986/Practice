"""
ETL Task for the ML Platform.

This task handles data transformation and feature generation workflows:
1. Time-based aggregation (hourly, daily, etc.)
2. Rolling window feature calculations
3. Two-stage ETL with intermediate checkpoints

Replaces manual scripts like [Data][DynamicRTBFeature][Gemini].py
with configuration-driven execution.
"""

from typing import List, Literal, Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from ml_platform.tasks.base import BaseTask
from ml_platform.config.models import AggregationColumn, RollingWindowSpec


class ETLTask(BaseTask):
    """
    Task for ETL data transformation pipelines.

    Supports:
    - Configurable time bucketing (minute, hour, day)
    - Multiple aggregation functions with optional conditions
    - Rolling window calculations for time-series features
    - Two-stage processing with intermediate output
    """

    def _process(self, df: DataFrame) -> DataFrame:
        """
        Execute the ETL transformation pipeline.

        Args:
            df: Input DataFrame from source

        Returns:
            Transformed DataFrame with aggregated and rolling features
        """
        params = self.task_config.get_etl_params()

        # Stage 0: Deduplication (if configured)
        # Mode: "none" (skip), "keep_first" (default, fast), or "drop_all" (slow)
        dedupe_mode = params.dedupe_mode or "keep_first"
        if params.dedupe_by and dedupe_mode != "none":
            df = self._remove_duplicates(df, params.dedupe_by, mode=dedupe_mode)
        elif dedupe_mode == "none":
            print("   Skipping deduplication (mode: none)")

        # Stage 1: Time bucketing
        if params.time_bucket:
            df = self._add_time_bucket(df, params.time_column, params.time_bucket)

        # Stage 2: Aggregation (optional - skip if no aggregations defined)
        if params.aggregations:
            agg_df = self._aggregate(df, params.group_by, params.aggregations)
        else:
            agg_df = df

        # Stage 3: Rolling window features (optional)
        if params.rolling_windows:
            agg_df = self._add_rolling_features(agg_df, params.group_by, params.rolling_windows)

        # Stage 4: Filter to output date range
        agg_df = self._filter_to_output_range(agg_df)

        # Stage 5: Add partition columns
        agg_df = self._add_partition_columns(agg_df)

        return agg_df

    def _remove_duplicates(
        self, 
        df: DataFrame, 
        dedupe_col: str, 
        mode: str = "keep_first"
    ) -> DataFrame:
        """
        Remove duplicate rows based on a column.

        Args:
            df: Input DataFrame
            dedupe_col: Column name to check for duplicates
            mode: Deduplication mode:
                - "none": Skip deduplication entirely
                - "keep_first": Keep one copy of each duplicate (fast, uses dropDuplicates)
                - "drop_all": Remove ALL rows where duplicates exist (slow, uses groupBy + join)

        Returns:
            DataFrame with duplicates handled according to mode
        """
        print(f"   Removing duplicates by: {dedupe_col} (mode: {mode})")

        if mode == "keep_first":
            # Fast path: keep one arbitrary row per dedupe_col value
            return df.dropDuplicates([dedupe_col])
        
        elif mode == "drop_all":
            # Slow path: remove ALL rows where dedupe_col has duplicates
            # Uses groupBy + join instead of window functions for better scalability
            unique_keys = (
                df.groupBy(dedupe_col)
                .count()
                .filter("count = 1")
                .select(dedupe_col)
            )
            return df.join(unique_keys, on=dedupe_col, how="inner")
        
        else:
            raise ValueError(f"Unknown dedupe_mode: {mode}. Use 'none', 'keep_first', or 'drop_all'")

    def _add_time_bucket(
        self,
        df: DataFrame,
        time_col: str,
        granularity: Literal["minute", "hour", "day"],
    ) -> DataFrame:
        """
        Add time bucket column by truncating timestamp.

        Args:
            df: Input DataFrame
            time_col: Name of timestamp column
            granularity: Truncation level (minute, hour, day)

        Returns:
            DataFrame with 'time_bucket' column added, or original df if granularity is None
        """
        return (
            df
            .withColumn("_ts", F.to_timestamp(time_col))
            .withColumn("time_bucket", F.date_trunc(granularity, "_ts"))
            .drop("_ts")
        )

    def _aggregate(
        self,
        df: DataFrame,
        group_by: List[str],
        aggregations: List[AggregationColumn],
    ) -> DataFrame:
        """
        Perform aggregations based on configuration.

        Args:
            df: Input DataFrame
            group_by: Columns to group by
            aggregations: List of aggregation specifications

        Returns:
            Aggregated DataFrame
        """
        agg_exprs = []

        for agg in aggregations:
            expr = self._build_agg_expression(agg)
            agg_exprs.append(expr)

        return df.groupBy(["time_bucket"] + group_by).agg(*agg_exprs)

    def _build_agg_expression(self, agg: AggregationColumn):
        """
        Build a PySpark aggregation expression from config.

        Args:
            agg: Aggregation column specification

        Returns:
            PySpark Column expression
        """
        source_col = F.col(agg.source_col)

        # Apply conditional filter if specified
        if agg.condition:
            source_col = F.when(F.expr(agg.condition), source_col)

        # Build aggregation based on function type
        agg_map = {
            "sum": lambda c: F.sum(F.coalesce(c, F.lit(0.0))),
            "count": lambda c: F.count("*"),
            "avg": lambda c: F.avg(c),
            "min": lambda c: F.min(c),
            "max": lambda c: F.max(c),
            "count_distinct": lambda c: F.countDistinct(c),
            "first": lambda c: F.first(c),
            "last": lambda c: F.last(c),
        }

        if agg.agg_func not in agg_map:
            raise ValueError(f"Unknown aggregation function: {agg.agg_func}")

        return agg_map[agg.agg_func](source_col).alias(agg.output_col)

    def _add_rolling_features(
        self,
        df: DataFrame,
        group_by: List[str],
        windows: List[RollingWindowSpec],
    ) -> DataFrame:
        """
        Calculate rolling window aggregations.

        Args:
            df: Aggregated DataFrame
            group_by: Partition columns for window
            windows: List of rolling window specifications

        Returns:
            DataFrame with rolling features added
        """
        # Add unix timestamp for range-based windows
        df = df.withColumn("_ts_unix", F.unix_timestamp("time_bucket"))

        for window_spec in windows:
            df = self._apply_rolling_window(df, group_by, window_spec)

        return df.drop("_ts_unix")

    def _apply_rolling_window(
        self,
        df: DataFrame,
        group_by: List[str],
        window_spec: RollingWindowSpec,
    ) -> DataFrame:
        """
        Apply a single rolling window to specified columns.

        Args:
            df: Input DataFrame
            group_by: Partition columns
            window_spec: Window specification

        Returns:
            DataFrame with rolling columns added
        """
        # Window: lookback period exclusive of current row
        w = (
            Window
            .partitionBy(group_by)
            .orderBy("_ts_unix")
            .rangeBetween(-window_spec.lookback_seconds, -1)
        )

        for col_name in window_spec.apply_to:
            output_name = f"{col_name}_{window_spec.name}"
            df = df.withColumn(output_name, F.sum(col_name).over(w))
            df = df.fillna(0, subset=[output_name])

        return df

    def _filter_to_output_range(self, df: DataFrame) -> DataFrame:
        """
        Filter to output date range, excluding lookback period.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        date_range = self.task_config.input.date_range
        if not date_range:
            return df

        date_from = date_range.date_from
        date_to = date_range.date_to

        if date_from:
            df = df.filter(F.col("time_bucket") >= date_from)
        if date_to:
            # If date_to is date-only, include the entire end date
            if len(date_to) == 10:  # "YYYY-MM-DD" format
                # Add 1 day and use < for exclusive upper bound
                from datetime import datetime, timedelta
                to_dt = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
                df = df.filter(F.col("time_bucket") < to_dt.strftime("%Y-%m-%d"))
            else:
                # Has time component, use as-is (exclusive)
                df = df.filter(F.col("time_bucket") < date_to)

        return df

    def _add_partition_columns(self, df: DataFrame) -> DataFrame:
        """
        Add date and hour columns for output partitioning.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with partition columns added
        """
        return (
            df
            .withColumn("date", F.date_format("time_bucket", "yyyy-MM-dd"))
            .withColumn("hour", F.date_format("time_bucket", "HH"))
        )

    def _save_output(self, result: DataFrame) -> None:
        """
        Save the result DataFrame to configured output location.

        Args:
            result: Transformed DataFrame to save
        """
        output_cfg = self.task_config.output

        print(f"   Path: {output_cfg.path}")
        print(f"   Format: {output_cfg.format}")
        print(f"   Mode: {output_cfg.mode}")

        writer = result.write.mode(output_cfg.mode)

        if output_cfg.partition_by:
            print(f"   Partition by: {output_cfg.partition_by}")
            writer = writer.partitionBy(*output_cfg.partition_by)

        # Write based on format
        format_writers = {
            "parquet": lambda w, p: w.parquet(p),
            "delta": lambda w, p: w.format("delta").save(p),
            "csv": lambda w, p: w.csv(p, header=True),
            "json": lambda w, p: w.json(p),
        }

        if output_cfg.format not in format_writers:
            raise ValueError(f"Unsupported output format: {output_cfg.format}")

        format_writers[output_cfg.format](writer, output_cfg.path)
        print(f"✅ Output saved to: {output_cfg.path}")
