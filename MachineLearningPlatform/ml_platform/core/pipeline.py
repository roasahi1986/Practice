"""
Feature Pipeline for the ML Platform.

This module provides the FeaturePipeline class that:
1. Loads data from configured sources
2. Applies column mappings (source columns -> features)
3. Applies feature transformations based on Feature definitions
4. Validates output contracts

The key design principle: Column mappings are provided by the Task config,
Features are defined individually (not in sets) to prevent duplication.
"""

from typing import Dict, List, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ml_platform.config.models import (
    DateRangeSpec,
    FeatureDefinition,
    SourceConfig,
    TaskInputSpec,
    ProjectConfig,
    ColumnMapping,
)


class FeaturePipeline:
    """
    Transforms raw data into model-ready features based on configuration.

    This class bridges the gap between raw data sources and feature requirements,
    using column mappings to connect them while applying transformations and
    validations as defined in the individual feature definitions.
    """

    def __init__(self, spark: SparkSession, project_config: ProjectConfig):
        """
        Initialize the pipeline.

        Args:
            spark: Active SparkSession
            project_config: Loaded ProjectConfig with sources and features
        """
        self.spark = spark
        self.config = project_config

    def create_dataset(self, input_spec: TaskInputSpec) -> DataFrame:
        """
        Create a feature dataset from source data using column mappings.

        Args:
            input_spec: Task input specification containing source and column_mappings

        Returns:
            DataFrame with transformed features

        Raises:
            ValueError: If source not found or mappings reference unknown features
        """
        source_cfg = self.config.sources.get(input_spec.source)
        if not source_cfg:
            raise ValueError(f"Source '{input_spec.source}' not found in configuration.")

        # Load raw data
        df = self._load_source(source_cfg, input_spec.date_range)

        # Apply optional filter
        if input_spec.filter_expr:
            df = df.filter(input_spec.filter_expr)

        # Apply column mappings and feature transformations
        if input_spec.column_mappings:
            df = self._apply_mappings(df, input_spec.column_mappings)

        return df

    def _load_source(
        self,
        source_cfg: SourceConfig,
        date_range: Optional["DateRangeSpec"] = None,
    ) -> DataFrame:
        """
        Load data from a configured source.

        Args:
            source_cfg: Source configuration
            date_range: Optional DateRangeSpec for filtering

        Returns:
            Raw DataFrame from source
        """
        print(f"   Loading from {source_cfg.type}: {source_cfg.path}")

        # TODO: Implement actual loading logic
        return self.spark.createDataFrame([], schema="event_id string")

    def _apply_mappings(
        self,
        df: DataFrame,
        mappings: List[ColumnMapping],
    ) -> DataFrame:
        """
        Apply column mappings and feature transformations.

        Args:
            df: Input DataFrame
            mappings: List of column mappings (source_col -> feature)

        Returns:
            DataFrame with transformed features
        """
        output_cols = []

        for mapping in mappings:
            source_col = mapping.source_col
            feature_name = mapping.feature
            output_col = mapping.output_col or feature_name

            # Get feature definition
            feat_def = self.config.features.get(feature_name)
            if not feat_def:
                raise ValueError(f"Feature '{feature_name}' not found in configuration")

            # Apply transformations
            col_expr = self._transform_column(source_col, feat_def)

            # Add to dataframe
            df = df.withColumn(output_col, col_expr)
            output_cols.append(output_col)

        return df.select(*output_cols)

    def _transform_column(
        self,
        source_col: str,
        feat_def: FeatureDefinition,
    ) -> F.Column:
        """
        Apply feature transformations to a column.

        Args:
            source_col: Source column name
            feat_def: Feature definition with rules

        Returns:
            Transformed PySpark Column expression
        """
        col_expr = F.col(source_col)

        # Apply transformation rules
        if feat_def.rules:
            col_expr = self._apply_rules(col_expr, feat_def)

        # Handle null values
        if feat_def.fill_na is not None:
            col_expr = F.coalesce(col_expr, F.lit(feat_def.fill_na))

        return col_expr

    def _apply_rules(
        self,
        col_expr: F.Column,
        feat_def: FeatureDefinition,
    ) -> F.Column:
        """
        Apply validation and transformation rules to a column.

        Args:
            col_expr: PySpark column expression
            feat_def: Feature definition

        Returns:
            Transformed column expression
        """
        rules = feat_def.rules

        # Normalize case
        if rules.format == "lowercase":
            col_expr = F.lower(col_expr)
        elif rules.format == "uppercase":
            col_expr = F.upper(col_expr)

        # Filter to allowed values
        if rules.allowed_values:
            fallback = feat_def.fill_na or "others"
            col_expr = F.when(
                col_expr.isin(rules.allowed_values),
                col_expr,
            ).otherwise(fallback)

        # Apply range constraints
        if rules.min_value is not None:
            col_expr = F.when(
                col_expr < rules.min_value,
                rules.min_value,
            ).otherwise(col_expr)

        if rules.max_value is not None:
            col_expr = F.when(
                col_expr > rules.max_value,
                rules.max_value,
            ).otherwise(col_expr)

        return col_expr
