"""
Enrichment Task for the ML Platform.

This task handles enriching a base dataset with features from other sources.
The typical use case is joining training data with pre-computed features.

Replaces manual scripts like [Data][DynamicRTBFeature][TrainingData].py
with configuration-driven execution.
"""

from typing import Any
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml_platform.tasks.base import BaseTask
from ml_platform.config.models import FeatureJoinSpec, AggregationColumn


class EnrichmentTask(BaseTask):
    """
    Task for enriching base data with features from other sources.

    Supports:
    - Aggregating base data before joining
    - Joining with multiple feature sources
    - Column renaming and null filling
    """

    def _process(self, df: DataFrame) -> DataFrame:
        """
        Execute the decoration pipeline.

        Args:
            df: Input DataFrame (base data)

        Returns:
            DataFrame decorated with features from joined sources
        """
        params = self.task_config.get_enrichment_params()

        # Stage 0: Add date/hour columns if needed (from source time column)
        df = self._add_time_columns(df)

        # Stage 1: Aggregate base data if needed
        if params.group_by and params.aggregations:
            df = self._aggregate_base(df, params.group_by, params.aggregations)

        # Stage 2: Join with each feature source
        for join_spec in params.feature_joins:
            df = self._join_features(df, join_spec)

        # Stage 3: Select output columns if specified
        if params.output_columns:
            df = df.select(*params.output_columns)

        return df

    def _add_time_columns(self, df: DataFrame) -> DataFrame:
        """
        Add date and hour columns from the source's time column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with date and hour columns added (if not present)
        """
        source_name = self.task_config.input.source
        source_config = self.project_config.sources.get(source_name)

        if not source_config or not source_config.time_column:
            return df

        time_col = source_config.time_column

        # Add date column if needed
        if "date" not in df.columns and time_col in df.columns:
            df = df.withColumn("date", F.date_format(time_col, "yyyy-MM-dd"))

        # Add hour column if needed
        if "hour" not in df.columns and time_col in df.columns:
            df = df.withColumn("hour", F.date_format(time_col, "HH"))

        return df

    def _aggregate_base(
        self,
        df: DataFrame,
        group_by: list,
        aggregations: list,
    ) -> DataFrame:
        """
        Aggregate the base data before joining with features.

        Args:
            df: Base DataFrame
            group_by: Columns to group by
            aggregations: Aggregation specifications

        Returns:
            Aggregated DataFrame
        """
        print(f"   Aggregating base data by: {group_by}")

        agg_exprs = []
        for agg in aggregations:
            if isinstance(agg, dict):
                agg = AggregationColumn(**agg)
            expr = self._build_agg_expression(agg)
            agg_exprs.append(expr)

        return df.groupBy(group_by).agg(*agg_exprs)

    def _build_agg_expression(self, agg: AggregationColumn):
        """Build a PySpark aggregation expression from config."""
        source_col = F.col(agg.source_col)

        if agg.condition:
            source_col = F.when(F.expr(agg.condition), source_col)

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

    def _join_features(self, df: DataFrame, join_spec: FeatureJoinSpec) -> DataFrame:
        """
        Join features from another source to the base DataFrame.

        Args:
            df: Base DataFrame
            join_spec: Specification for the feature join

        Returns:
            DataFrame with features joined
        """
        source_name = join_spec.source
        print(f"   Joining features from: {source_name}")

        # Load feature source
        source_config = self.project_config.sources.get(source_name)
        if not source_config:
            raise ValueError(f"Feature source '{source_name}' not found in configuration.")

        feature_df = self.spark.read.parquet(source_config.path)

        # Apply date filter if the base data has a date range
        date_range = self.task_config.input.date_range
        if date_range and date_range.date_from:
            # Find the time/date column in the feature source
            time_col = source_config.time_column or "date"
            if time_col in feature_df.columns:
                feature_df = feature_df.filter(F.col(time_col) >= date_range.date_from)
            elif "date" in feature_df.columns:
                feature_df = feature_df.filter(F.col("date") >= date_range.date_from)

        if date_range and date_range.date_to:
            time_col = source_config.time_column or "date"
            if time_col in feature_df.columns:
                feature_df = feature_df.filter(F.col(time_col) < date_range.date_to)
            elif "date" in feature_df.columns:
                feature_df = feature_df.filter(F.col("date") < date_range.date_to)

        # Select only needed columns (join keys + feature columns)
        feature_keys = list(join_spec.join_keys.values())
        select_cols = feature_keys + join_spec.select_cols
        feature_df = feature_df.select(*[c for c in select_cols if c in feature_df.columns])

        # Rename ALL feature columns to avoid conflicts (prefix with alias)
        prefix = f"_feat_{source_name}_"
        renamed_cols = {}
        renamed_keys = {}
        
        # Rename feature columns
        for col in join_spec.select_cols:
            if col in feature_df.columns:
                new_name = f"{prefix}{col}"
                feature_df = feature_df.withColumnRenamed(col, new_name)
                renamed_cols[col] = new_name

        # Rename join key columns in feature df to avoid ambiguity
        for base_col, feat_col in join_spec.join_keys.items():
            if feat_col in feature_df.columns:
                new_name = f"{prefix}{feat_col}"
                feature_df = feature_df.withColumnRenamed(feat_col, new_name)
                renamed_keys[feat_col] = new_name

        # Build join condition using renamed feature columns
        join_conditions = []
        for base_col, feat_col in join_spec.join_keys.items():
            feat_col_renamed = renamed_keys.get(feat_col, feat_col)
            # Handle null-safe equality for nullable columns
            join_conditions.append(
                F.col(base_col).eqNullSafe(F.col(feat_col_renamed))
            )

        join_expr = join_conditions[0]
        for cond in join_conditions[1:]:
            join_expr = join_expr & cond

        # Perform left join
        df = df.join(F.broadcast(feature_df), join_expr, how="left")

        # Drop the renamed feature join key columns (they're duplicates of base columns)
        for feat_col_renamed in renamed_keys.values():
            if feat_col_renamed in df.columns:
                df = df.drop(feat_col_renamed)

        # Rename columns to final names and fill nulls
        for orig_col in join_spec.select_cols:
            temp_name = renamed_cols.get(orig_col, orig_col)
            final_name = orig_col
            if join_spec.rename_cols and orig_col in join_spec.rename_cols:
                final_name = join_spec.rename_cols[orig_col]

            if temp_name in df.columns:
                df = df.withColumnRenamed(temp_name, final_name)
                if join_spec.fill_na is not None:
                    df = df.withColumn(
                        final_name,
                        F.coalesce(F.col(final_name), F.lit(join_spec.fill_na))
                    )

        return df

    def _save_output(self, result: DataFrame) -> None:
        """
        Save the decorated DataFrame to configured output location.

        Args:
            result: Decorated DataFrame to save
        """
        output_cfg = self.task_config.output

        print(f"   Path: {output_cfg.path}")
        print(f"   Format: {output_cfg.format}")
        print(f"   Mode: {output_cfg.mode}")

        writer = result.write.mode(output_cfg.mode)

        if output_cfg.partition_by:
            print(f"   Partition by: {output_cfg.partition_by}")
            writer = writer.partitionBy(*output_cfg.partition_by)

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

