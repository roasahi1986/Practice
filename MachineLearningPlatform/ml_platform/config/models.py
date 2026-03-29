"""
Configuration Models for the ML Platform.

This module defines the Pydantic models that enforce strict schemas for all configurations.
Based on the principle of separating:
1. Sources: Physical data location and schema (What we HAVE)
2. Features: Individual feature contracts (What we NEED)
3. Tasks: Workload definitions with column mappings (What we DO)

Key design decisions:
- Features are defined individually, NOT in sets (prevents duplication)
- Source columns declare which features they can provide (lineage)
- Column mappings in Task config are validated against this lineage
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Source Configuration (Physical Data Layer)
# =============================================================================

class ColumnDefinition(BaseModel):
    """
    Defines a column's physical properties in a data source.

    The `provides` field lists which features (from features.yaml) can be
    created from this column. This creates explicit lineage.
    """
    dtype: Literal["string", "int", "long", "float", "double", "boolean", "timestamp", "date"]
    format: Optional[str] = None
    description: Optional[str] = None
    nullable: bool = True
    # Lineage: which features this column can provide
    provides: Optional[List[str]] = None


class SourceConfig(BaseModel):
    """
    Defines a physical data source (Hive table, S3 path, etc.)
    This is the "inventory" of available data.
    """
    type: Literal["hive", "coba2", "s3_parquet", "s3_csv", "delta"]
    path: str
    columns: Dict[str, ColumnDefinition]
    description: Optional[str] = None
    partition_cols: Optional[List[str]] = None
    time_column: Optional[str] = None

    def get_provided_features(self) -> Dict[str, List[str]]:
        """
        Get a mapping of feature -> list of columns that can provide it.
        Useful for understanding data lineage.
        """
        feature_to_cols: Dict[str, List[str]] = {}
        for col_name, col_def in self.columns.items():
            if col_def.provides:
                for feature in col_def.provides:
                    if feature not in feature_to_cols:
                        feature_to_cols[feature] = []
                    feature_to_cols[feature].append(col_name)
        return feature_to_cols


# =============================================================================
# Feature Configuration (Logical Data Layer)
# =============================================================================

class FeatureValidationRules(BaseModel):
    """
    Rules for validating and transforming a feature.
    This defines the "contract" the consumer expects.
    """
    format: Optional[str] = None  # e.g., "lowercase", "iso_2_char"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    regex: Optional[str] = None
    max_null_percentage: Optional[float] = None


class FeatureDefinition(BaseModel):
    """
    Defines a single feature's contract (what the model expects).

    Features are defined individually (not in sets) to prevent duplication.
    The Task config specifies which features to use via column_mappings.
    """
    dtype: str
    description: Optional[str] = None
    rules: FeatureValidationRules = Field(default_factory=FeatureValidationRules)
    fill_na: Optional[Any] = None


# =============================================================================
# Column Mapping (Pipeline Layer - Connects Sources to Features)
# =============================================================================

class ColumnMapping(BaseModel):
    """
    Maps a source column to a feature.
    This is defined in the Task config, connecting sources to features.
    """
    source_col: str  # Column name in the data source
    feature: str  # Feature name in features.yaml
    output_col: Optional[str] = None  # Optional: rename output (defaults to feature name)


# =============================================================================
# Aggregation Configuration (For ETL Tasks)
# =============================================================================

class AggregationColumn(BaseModel):
    """Defines an aggregation to perform on a column."""
    source_col: str
    agg_func: Literal["sum", "count", "avg", "min", "max", "count_distinct", "first", "last"]
    output_col: str
    condition: Optional[str] = None


class RollingWindowSpec(BaseModel):
    """Defines a rolling window for time-series feature generation."""
    name: str
    lookback_seconds: int
    apply_to: List[str]


class OutputGrouping(BaseModel):
    """Defines a separate output grouping for aggregated results."""
    name: str
    group_cols: List[str]


# =============================================================================
# Task Configuration (Execution Layer)
# =============================================================================

class DateRangeSpec(BaseModel):
    """Date range specification. Values can be null to require runtime override."""
    model_config = {"populate_by_name": True}
    
    date_from: Optional[str] = Field(None, alias="from")
    date_to: Optional[str] = Field(None, alias="to")


class TaskInputSpec(BaseModel):
    """Defines the input for a task."""
    source: str
    date_range: Optional[DateRangeSpec] = None
    filter_expr: Optional[str] = None
    select_columns: Optional[List[str]] = None
    column_mappings: Optional[List[ColumnMapping]] = None


class TaskOutputSpec(BaseModel):
    """Defines the output for a task."""
    path: str
    format: Literal["parquet", "delta", "csv", "json", "html", "mlflow"] = "parquet"
    mode: Literal["overwrite", "append", "error", "ignore"] = "overwrite"
    partition_by: Optional[List[str]] = None


class ETLTaskParams(BaseModel):
    """Specific parameters for ETL tasks."""
    group_by: List[str]
    time_bucket: Optional[Literal["minute", "hour", "day"]] = "hour"
    time_column: str
    aggregations: Optional[List[AggregationColumn]] = None
    rolling_windows: Optional[List[RollingWindowSpec]] = None
    dedupe_by: Optional[str] = None
    dedupe_mode: Optional[Literal["none", "keep_first", "drop_all"]] = "keep_first"
    output_groupings: Optional[List[OutputGrouping]] = None

    def get_lookback_seconds(self) -> int:
        """Calculate required lookback from max rolling window."""
        if not self.rolling_windows:
            return 0
        return max(w.lookback_seconds for w in self.rolling_windows)

    def get_lookback_hours(self) -> int:
        """Calculate required lookback hours (rounded up)."""
        import math
        return math.ceil(self.get_lookback_seconds() / 3600)


class TrainingTaskParams(BaseModel):
    """Specific parameters for Training tasks."""
    model_config = {"protected_namespaces": ()}
    
    learner: str
    target_col: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    validation_split: float = 0.2


class FeatureJoinSpec(BaseModel):
    """Defines how to join a feature source to the base data."""
    source: str  # Name of the feature source (e.g., compute_rolling_win_price_by_publisher)
    join_keys: Dict[str, str]  # Mapping of base column -> feature column
    select_cols: List[str]  # Columns to select from the feature source
    rename_cols: Optional[Dict[str, str]] = None  # Optional: rename columns after join
    fill_na: Optional[float] = None  # Optional: fill nulls with this value


class EnrichmentTaskParams(BaseModel):
    """Specific parameters for Enrichment tasks (joining base data with features)."""
    # SQL query or filter for base data (can include aggregation)
    base_query: Optional[str] = None
    # Alternative: just filter the source
    filter_expr: Optional[str] = None
    # Columns to group by (if aggregating base data)
    group_by: Optional[List[str]] = None
    # Aggregations to perform on base data
    aggregations: Optional[List[AggregationColumn]] = None
    # Feature sources to join
    feature_joins: List[FeatureJoinSpec]
    # Output columns to select (if not specified, all columns are kept)
    output_columns: Optional[List[str]] = None


class TaskConfig(BaseModel):
    """Defines a complete task specification."""
    name: str
    type: Literal["training", "etl", "enrichment", "visualization", "inference", "evaluation", "simulation"]
    input: TaskInputSpec
    output: Optional[TaskOutputSpec] = None  # Optional for visualization tasks
    params: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None

    def get_etl_params(self) -> ETLTaskParams:
        """Parse params as ETLTaskParams for ETL tasks."""
        if self.type != "etl":
            raise ValueError("get_etl_params() called on non-ETL task")
        return ETLTaskParams(**self.params)

    def get_training_params(self) -> TrainingTaskParams:
        """Parse params as TrainingTaskParams for Training tasks."""
        if self.type != "training":
            raise ValueError("get_training_params() called on non-Training task")
        return TrainingTaskParams(**self.params)

    def get_enrichment_params(self) -> EnrichmentTaskParams:
        """Parse params as EnrichmentTaskParams for Enrichment tasks."""
        if self.type != "enrichment":
            raise ValueError("get_enrichment_params() called on non-Enrichment task")
        return EnrichmentTaskParams(**self.params)


# =============================================================================
# Global Settings
# =============================================================================

class GlobalSettings(BaseModel):
    """Global settings shared across all tasks."""
    output_root: Optional[str] = None


# =============================================================================
# Root Config Container
# =============================================================================

class ProjectConfig(BaseModel):
    """
    The root container holding all configurations.
    This is the single source of truth for the entire system.
    """
    sources: Dict[str, SourceConfig] = Field(default_factory=dict)
    features: Dict[str, FeatureDefinition] = Field(default_factory=dict)
    tasks: Dict[str, TaskConfig] = Field(default_factory=dict)
    settings: GlobalSettings = Field(default_factory=GlobalSettings)

    def get_task_output_path(self, task_name: str) -> str:
        """Get the output path for a task (already resolved by Jinja2 at load time)."""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found")
        return self.tasks[task_name].output.path

    @model_validator(mode="after")
    def validate_references(self):
        """
        Validate that all references in tasks point to valid sources and features,
        and that column mappings respect the declared lineage.
        """
        errors = []

        for task_name, task in self.tasks.items():
            source_name = task.input.source

            # Check source reference
            if source_name not in self.sources:
                errors.append(
                    f"Task '{task_name}' references unknown source '{source_name}'"
                )
                continue

            source = self.sources[source_name]

            # Check column mappings
            if task.input.column_mappings:
                for mapping in task.input.column_mappings:
                    # Check feature exists
                    if mapping.feature not in self.features:
                        errors.append(
                            f"Task '{task_name}' maps to unknown feature '{mapping.feature}'"
                        )
                        continue

                    # Check source column exists
                    if mapping.source_col not in source.columns:
                        errors.append(
                            f"Task '{task_name}': column '{mapping.source_col}' "
                            f"not found in source '{source_name}'"
                        )
                        continue

                    # Check lineage: column must declare it can provide this feature
                    col_def = source.columns[mapping.source_col]
                    if col_def.provides and mapping.feature not in col_def.provides:
                        errors.append(
                            f"Task '{task_name}': column '{mapping.source_col}' "
                            f"cannot provide feature '{mapping.feature}' "
                            f"(allowed: {col_def.provides})"
                        )

        if errors:
            raise ValueError(f"Configuration validation errors:\n" + "\n".join(errors))

        return self

    def get_unused_sources(self) -> List[str]:
        """Detect sources that are defined but never used."""
        used_sources = set()
        for task in self.tasks.values():
            # Primary input source
            used_sources.add(task.input.source)
            
            # Feature sources from enrichment tasks
            if task.type == "enrichment" and "feature_joins" in task.params:
                for join_spec in task.params["feature_joins"]:
                    if isinstance(join_spec, dict) and "source" in join_spec:
                        used_sources.add(join_spec["source"])

        all_sources = set(self.sources.keys())
        return list(all_sources - used_sources)

    def get_unused_features(self) -> List[str]:
        """Detect features that are defined but never used."""
        used_features = set()
        for task in self.tasks.values():
            if task.input.column_mappings:
                for mapping in task.input.column_mappings:
                    used_features.add(mapping.feature)

        all_features = set(self.features.keys())
        return list(all_features - used_features)

    def get_feature_lineage(self, feature_name: str) -> Dict[str, List[str]]:
        """
        Get all sources and columns that can provide a specific feature.

        Returns:
            Dict mapping source_name -> list of column names
        """
        lineage: Dict[str, List[str]] = {}
        for source_name, source in self.sources.items():
            for col_name, col_def in source.columns.items():
                if col_def.provides and feature_name in col_def.provides:
                    if source_name not in lineage:
                        lineage[source_name] = []
                    lineage[source_name].append(col_name)
        return lineage
