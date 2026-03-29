"""
Tests for ml_platform/config/models.py

Tests cover:
- ColumnDefinition validation and dtype constraints
- SourceConfig creation and get_provided_features()
- FeatureDefinition with validation rules
- DateRangeSpec with optional fields
- TaskInputSpec, TaskOutputSpec, TaskConfig
- ETLTaskParams including get_lookback_seconds/hours
- TrainingTaskParams validation
- ProjectConfig validation including:
  - Reference validation (sources, features)
  - Column mapping lineage validation
  - get_unused_sources/features
  - get_feature_lineage
"""

import pytest
from pydantic import ValidationError

from ml_platform.config.models import (
    ColumnDefinition,
    SourceConfig,
    FeatureDefinition,
    FeatureValidationRules,
    ColumnMapping,
    AggregationColumn,
    RollingWindowSpec,
    OutputGrouping,
    DateRangeSpec,
    TaskInputSpec,
    TaskOutputSpec,
    ETLTaskParams,
    TrainingTaskParams,
    TaskConfig,
    GlobalSettings,
    ProjectConfig,
)


# =============================================================================
# ColumnDefinition Tests
# =============================================================================

class TestColumnDefinition:
    """Tests for ColumnDefinition model."""

    def test_valid_dtypes(self):
        """Test all valid dtype values are accepted."""
        valid_dtypes = ["string", "int", "long", "float", "double", "boolean", "timestamp", "date"]
        for dtype in valid_dtypes:
            col = ColumnDefinition(dtype=dtype)
            assert col.dtype == dtype

    def test_invalid_dtype_rejected(self):
        """Test invalid dtype raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ColumnDefinition(dtype="invalid_type")
        assert "dtype" in str(exc_info.value)

    def test_optional_fields_default_to_none(self):
        """Test optional fields have correct defaults."""
        col = ColumnDefinition(dtype="string")
        assert col.format is None
        assert col.description is None
        assert col.nullable is True
        assert col.provides is None

    def test_provides_field_accepts_list(self):
        """Test provides field accepts a list of feature names."""
        col = ColumnDefinition(
            dtype="timestamp",
            provides=["date", "hour_of_day", "day_of_week"]
        )
        assert col.provides == ["date", "hour_of_day", "day_of_week"]

    def test_all_fields_populated(self):
        """Test ColumnDefinition with all fields populated."""
        col = ColumnDefinition(
            dtype="float",
            format="currency",
            description="Price in USD",
            nullable=False,
            provides=["price_feature"]
        )
        assert col.dtype == "float"
        assert col.format == "currency"
        assert col.description == "Price in USD"
        assert col.nullable is False
        assert col.provides == ["price_feature"]


# =============================================================================
# SourceConfig Tests
# =============================================================================

class TestSourceConfig:
    """Tests for SourceConfig model."""

    @pytest.fixture
    def sample_source(self):
        """Create a sample source config for testing."""
        return SourceConfig(
            type="hive",
            path="db.schema.table",
            columns={
                "event_id": ColumnDefinition(dtype="string"),
                "event_ts": ColumnDefinition(
                    dtype="timestamp",
                    provides=["date", "hour_of_day"]
                ),
                "price": ColumnDefinition(
                    dtype="double",
                    provides=["win_price"]
                ),
                "country": ColumnDefinition(
                    dtype="string",
                    provides=["country_code", "geo"]
                ),
            },
            description="Test source",
            partition_cols=["date"],
            time_column="event_ts"
        )

    def test_valid_source_types(self):
        """Test all valid source types are accepted."""
        valid_types = ["hive", "s3_parquet", "s3_csv", "delta"]
        for source_type in valid_types:
            source = SourceConfig(
                type=source_type,
                path="test/path",
                columns={"id": ColumnDefinition(dtype="string")}
            )
            assert source.type == source_type

    def test_invalid_source_type_rejected(self):
        """Test invalid source type raises ValidationError."""
        with pytest.raises(ValidationError):
            SourceConfig(
                type="invalid_type",
                path="test/path",
                columns={}
            )

    def test_get_provided_features_basic(self, sample_source):
        """Test get_provided_features returns correct mapping."""
        features = sample_source.get_provided_features()
        
        assert "date" in features
        assert "event_ts" in features["date"]
        
        assert "hour_of_day" in features
        assert "event_ts" in features["hour_of_day"]
        
        assert "win_price" in features
        assert "price" in features["win_price"]
        
        assert "country_code" in features
        assert "country" in features["country_code"]
        
        assert "geo" in features
        assert "country" in features["geo"]

    def test_get_provided_features_empty(self):
        """Test get_provided_features with no provides fields."""
        source = SourceConfig(
            type="hive",
            path="test.table",
            columns={
                "col1": ColumnDefinition(dtype="string"),
                "col2": ColumnDefinition(dtype="int"),
            }
        )
        features = source.get_provided_features()
        assert features == {}

    def test_get_provided_features_multiple_columns_same_feature(self):
        """Test when multiple columns provide the same feature."""
        source = SourceConfig(
            type="hive",
            path="test.table",
            columns={
                "country_code": ColumnDefinition(dtype="string", provides=["geo"]),
                "geo_id": ColumnDefinition(dtype="string", provides=["geo"]),
            }
        )
        features = source.get_provided_features()
        assert "geo" in features
        assert set(features["geo"]) == {"country_code", "geo_id"}


# =============================================================================
# FeatureDefinition Tests
# =============================================================================

class TestFeatureDefinition:
    """Tests for FeatureDefinition model."""

    def test_basic_feature(self):
        """Test basic feature definition."""
        feature = FeatureDefinition(dtype="float")
        assert feature.dtype == "float"
        assert feature.description is None
        assert feature.fill_na is None

    def test_feature_with_rules(self):
        """Test feature with validation rules."""
        feature = FeatureDefinition(
            dtype="float",
            description="Win price in cents",
            rules=FeatureValidationRules(
                min_value=0.0,
                max_value=10000.0,
                max_null_percentage=0.1
            ),
            fill_na=0.0
        )
        assert feature.rules.min_value == 0.0
        assert feature.rules.max_value == 10000.0
        assert feature.rules.max_null_percentage == 0.1
        assert feature.fill_na == 0.0

    def test_feature_with_allowed_values(self):
        """Test feature with allowed values constraint."""
        feature = FeatureDefinition(
            dtype="string",
            rules=FeatureValidationRules(
                allowed_values=["us", "uk", "de", "fr"]
            )
        )
        assert feature.rules.allowed_values == ["us", "uk", "de", "fr"]

    def test_feature_with_regex(self):
        """Test feature with regex constraint."""
        feature = FeatureDefinition(
            dtype="string",
            rules=FeatureValidationRules(
                regex=r"^[a-z]{2}$"
            )
        )
        assert feature.rules.regex == r"^[a-z]{2}$"


# =============================================================================
# DateRangeSpec Tests
# =============================================================================

class TestDateRangeSpec:
    """Tests for DateRangeSpec model."""

    def test_both_none(self):
        """Test DateRangeSpec with both fields None."""
        dr = DateRangeSpec()
        assert dr.date_from is None
        assert dr.date_to is None

    def test_both_set(self):
        """Test DateRangeSpec with both fields set."""
        dr = DateRangeSpec(date_from="2025-01-01", date_to="2025-01-31")
        assert dr.date_from == "2025-01-01"
        assert dr.date_to == "2025-01-31"

    def test_only_start_set(self):
        """Test DateRangeSpec with only date_from set."""
        dr = DateRangeSpec(date_from="2025-01-01")
        assert dr.date_from == "2025-01-01"
        assert dr.date_to is None

    def test_datetime_format(self):
        """Test DateRangeSpec with datetime format."""
        dr = DateRangeSpec(
            date_from="2025-01-01 00:00:00",
            date_to="2025-01-31 23:59:59"
        )
        assert dr.date_from == "2025-01-01 00:00:00"
        assert dr.date_to == "2025-01-31 23:59:59"
    
    def test_alias_from_yaml(self):
        """Test DateRangeSpec can be created using 'from' and 'to' aliases."""
        # This simulates how Pydantic parses from YAML with aliases
        data = {"from": "2025-01-01", "to": "2025-01-31"}
        dr = DateRangeSpec(**data)
        assert dr.date_from == "2025-01-01"
        assert dr.date_to == "2025-01-31"


# =============================================================================
# ETLTaskParams Tests
# =============================================================================

class TestETLTaskParams:
    """Tests for ETLTaskParams model."""

    def test_basic_etl_params(self):
        """Test basic ETL params creation."""
        params = ETLTaskParams(
            group_by=["rtb_id", "supply_name"],
            time_column="event_ts"
        )
        assert params.group_by == ["rtb_id", "supply_name"]
        assert params.time_column == "event_ts"
        assert params.time_bucket == "hour"  # default
        assert params.aggregations is None
        assert params.rolling_windows is None

    def test_time_bucket_values(self):
        """Test all valid time_bucket values."""
        for bucket in ["minute", "hour", "day", None]:
            params = ETLTaskParams(
                group_by=["id"],
                time_column="ts",
                time_bucket=bucket
            )
            assert params.time_bucket == bucket

    def test_get_lookback_seconds_no_windows(self):
        """Test get_lookback_seconds with no rolling windows."""
        params = ETLTaskParams(
            group_by=["id"],
            time_column="ts"
        )
        assert params.get_lookback_seconds() == 0

    def test_get_lookback_seconds_with_windows(self):
        """Test get_lookback_seconds with rolling windows."""
        params = ETLTaskParams(
            group_by=["id"],
            time_column="ts",
            rolling_windows=[
                RollingWindowSpec(name="1h", lookback_seconds=3600, apply_to=["col1"]),
                RollingWindowSpec(name="3h", lookback_seconds=10800, apply_to=["col1"]),
                RollingWindowSpec(name="24h", lookback_seconds=86400, apply_to=["col1"]),
            ]
        )
        assert params.get_lookback_seconds() == 86400

    def test_get_lookback_hours_rounded_up(self):
        """Test get_lookback_hours rounds up correctly."""
        params = ETLTaskParams(
            group_by=["id"],
            time_column="ts",
            rolling_windows=[
                RollingWindowSpec(name="90min", lookback_seconds=5400, apply_to=["col1"]),
            ]
        )
        # 5400 seconds = 1.5 hours, should round up to 2
        assert params.get_lookback_hours() == 2

    def test_etl_params_with_aggregations(self):
        """Test ETL params with aggregations."""
        params = ETLTaskParams(
            group_by=["rtb_id"],
            time_column="ts",
            aggregations=[
                AggregationColumn(source_col="price", agg_func="sum", output_col="total_price"),
                AggregationColumn(source_col="*", agg_func="count", output_col="event_count"),
            ]
        )
        assert len(params.aggregations) == 2
        assert params.aggregations[0].agg_func == "sum"
        assert params.aggregations[1].agg_func == "count"


# =============================================================================
# TrainingTaskParams Tests
# =============================================================================

class TestTrainingTaskParams:
    """Tests for TrainingTaskParams model."""

    def test_basic_training_params(self):
        """Test basic training params."""
        params = TrainingTaskParams(
            learner="xgboost",
            target_col="label"
        )
        assert params.learner == "xgboost"
        assert params.target_col == "label"
        assert params.hyperparameters == {}
        assert params.validation_split == 0.2

    def test_training_params_with_hyperparameters(self):
        """Test training params with hyperparameters."""
        params = TrainingTaskParams(
            learner="lightgbm",
            target_col="label",
            hyperparameters={
                "learning_rate": 0.1,
                "num_leaves": 31,
                "max_depth": 6
            },
            validation_split=0.3
        )
        assert params.hyperparameters["learning_rate"] == 0.1
        assert params.validation_split == 0.3


# =============================================================================
# TaskConfig Tests
# =============================================================================

class TestTaskConfig:
    """Tests for TaskConfig model."""

    def test_etl_task_config(self):
        """Test ETL task configuration."""
        task = TaskConfig(
            name="test_etl",
            type="etl",
            input=TaskInputSpec(source="test_source"),
            output=TaskOutputSpec(path="s3://bucket/path", format="parquet"),
            params={
                "group_by": ["id"],
                "time_column": "ts",
                "aggregations": []
            }
        )
        assert task.type == "etl"
        etl_params = task.get_etl_params()
        assert etl_params.group_by == ["id"]

    def test_training_task_config(self):
        """Test training task configuration."""
        task = TaskConfig(
            name="test_train",
            type="training",
            input=TaskInputSpec(source="test_source"),
            output=TaskOutputSpec(path="s3://bucket/model", format="mlflow"),
            params={
                "learner": "xgboost",
                "target_col": "label"
            }
        )
        assert task.type == "training"
        train_params = task.get_training_params()
        assert train_params.learner == "xgboost"

    def test_get_etl_params_on_non_etl_raises(self):
        """Test get_etl_params on non-ETL task raises error."""
        task = TaskConfig(
            name="test",
            type="training",
            input=TaskInputSpec(source="src"),
            output=TaskOutputSpec(path="/tmp"),
            params={"learner": "xgb", "target_col": "y"}
        )
        with pytest.raises(ValueError, match="non-ETL"):
            task.get_etl_params()

    def test_get_training_params_on_non_training_raises(self):
        """Test get_training_params on non-training task raises error."""
        task = TaskConfig(
            name="test",
            type="etl",
            input=TaskInputSpec(source="src"),
            output=TaskOutputSpec(path="/tmp"),
            params={"group_by": [], "time_column": "ts"}
        )
        with pytest.raises(ValueError, match="non-Training"):
            task.get_training_params()


# =============================================================================
# ProjectConfig Tests
# =============================================================================

class TestProjectConfig:
    """Tests for ProjectConfig model."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid project config for testing."""
        return ProjectConfig(
            sources={
                "source1": SourceConfig(
                    type="hive",
                    path="db.table1",
                    columns={
                        "id": ColumnDefinition(dtype="string"),
                        "price": ColumnDefinition(dtype="double", provides=["win_price"]),
                        "country": ColumnDefinition(dtype="string", provides=["country_code"]),
                    }
                ),
                "source2": SourceConfig(
                    type="hive",
                    path="db.table2",
                    columns={
                        "id": ColumnDefinition(dtype="string"),
                    }
                ),
            },
            features={
                "win_price": FeatureDefinition(dtype="float"),
                "country_code": FeatureDefinition(dtype="string"),
                "unused_feature": FeatureDefinition(dtype="int"),
            },
            tasks={
                "task1": TaskConfig(
                    name="task1",
                    type="etl",
                    input=TaskInputSpec(
                        source="source1",
                        column_mappings=[
                            ColumnMapping(source_col="price", feature="win_price"),
                        ]
                    ),
                    output=TaskOutputSpec(path="/out"),
                    params={"group_by": [], "time_column": "ts"}
                ),
            }
        )

    def test_valid_config_passes(self, valid_config):
        """Test valid configuration passes validation."""
        # Should not raise
        assert valid_config is not None
        assert len(valid_config.sources) == 2
        assert len(valid_config.features) == 3
        assert len(valid_config.tasks) == 1

    def test_invalid_source_reference_fails(self):
        """Test task referencing non-existent source fails."""
        with pytest.raises(ValueError, match="unknown source"):
            ProjectConfig(
                sources={},
                features={},
                tasks={
                    "task1": TaskConfig(
                        name="task1",
                        type="etl",
                        input=TaskInputSpec(source="nonexistent"),
                        output=TaskOutputSpec(path="/out"),
                        params={}
                    )
                }
            )

    def test_invalid_feature_reference_fails(self):
        """Test mapping to non-existent feature fails."""
        with pytest.raises(ValueError, match="unknown feature"):
            ProjectConfig(
                sources={
                    "src": SourceConfig(
                        type="hive",
                        path="db.t",
                        columns={"col": ColumnDefinition(dtype="string")}
                    )
                },
                features={},
                tasks={
                    "task1": TaskConfig(
                        name="task1",
                        type="etl",
                        input=TaskInputSpec(
                            source="src",
                            column_mappings=[
                                ColumnMapping(source_col="col", feature="nonexistent_feature")
                            ]
                        ),
                        output=TaskOutputSpec(path="/out"),
                        params={}
                    )
                }
            )

    def test_invalid_column_reference_fails(self):
        """Test mapping from non-existent column fails."""
        with pytest.raises(ValueError, match="not found in source"):
            ProjectConfig(
                sources={
                    "src": SourceConfig(
                        type="hive",
                        path="db.t",
                        columns={"col": ColumnDefinition(dtype="string")}
                    )
                },
                features={
                    "feat": FeatureDefinition(dtype="string")
                },
                tasks={
                    "task1": TaskConfig(
                        name="task1",
                        type="etl",
                        input=TaskInputSpec(
                            source="src",
                            column_mappings=[
                                ColumnMapping(source_col="nonexistent_col", feature="feat")
                            ]
                        ),
                        output=TaskOutputSpec(path="/out"),
                        params={}
                    )
                }
            )

    def test_lineage_violation_fails(self):
        """Test column that doesn't provide the mapped feature fails."""
        with pytest.raises(ValueError, match="cannot provide feature"):
            ProjectConfig(
                sources={
                    "src": SourceConfig(
                        type="hive",
                        path="db.t",
                        columns={
                            "col": ColumnDefinition(
                                dtype="string",
                                provides=["other_feature"]  # Not 'target_feature'
                            )
                        }
                    )
                },
                features={
                    "target_feature": FeatureDefinition(dtype="string"),
                    "other_feature": FeatureDefinition(dtype="string"),
                },
                tasks={
                    "task1": TaskConfig(
                        name="task1",
                        type="etl",
                        input=TaskInputSpec(
                            source="src",
                            column_mappings=[
                                ColumnMapping(source_col="col", feature="target_feature")
                            ]
                        ),
                        output=TaskOutputSpec(path="/out"),
                        params={}
                    )
                }
            )

    def test_get_unused_sources(self, valid_config):
        """Test get_unused_sources returns sources not used by any task."""
        unused = valid_config.get_unused_sources()
        assert "source2" in unused
        assert "source1" not in unused

    def test_get_unused_features(self, valid_config):
        """Test get_unused_features returns features not used in mappings."""
        unused = valid_config.get_unused_features()
        assert "unused_feature" in unused
        assert "country_code" in unused  # Not mapped in task1
        assert "win_price" not in unused  # Mapped in task1

    def test_get_feature_lineage(self, valid_config):
        """Test get_feature_lineage returns correct source/column mapping."""
        lineage = valid_config.get_feature_lineage("win_price")
        assert "source1" in lineage
        assert "price" in lineage["source1"]

    def test_get_feature_lineage_not_found(self, valid_config):
        """Test get_feature_lineage for feature not provided by any source."""
        lineage = valid_config.get_feature_lineage("nonexistent")
        assert lineage == {}

    def test_get_task_output_path(self, valid_config):
        """Test get_task_output_path returns correct path."""
        path = valid_config.get_task_output_path("task1")
        assert path == "/out"

    def test_get_task_output_path_not_found(self, valid_config):
        """Test get_task_output_path raises for unknown task."""
        with pytest.raises(ValueError, match="not found"):
            valid_config.get_task_output_path("nonexistent_task")


# =============================================================================
# GlobalSettings Tests
# =============================================================================

class TestGlobalSettings:
    """Tests for GlobalSettings model."""

    def test_default_settings(self):
        """Test default global settings."""
        settings = GlobalSettings()
        assert settings.output_root is None

    def test_custom_settings(self):
        """Test custom global settings."""
        settings = GlobalSettings(output_root="s3://bucket/output")
        assert settings.output_root == "s3://bucket/output"

