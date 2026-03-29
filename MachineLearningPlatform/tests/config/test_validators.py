"""
Tests for ml_platform/config/validators.py

Tests cover:
- ValidationResult dataclass
- ConfigValidator class:
  - validate_all() aggregation
  - _validate_references() for source/feature references
  - _validate_unused() for unused sources/features
  - _validate_tasks() for task-specific requirements
- diff_configs() for comparing configurations
- print_validation_report() output formatting
"""

import pytest
from io import StringIO
import sys

from ml_platform.config.models import (
    ColumnDefinition,
    SourceConfig,
    FeatureDefinition,
    ColumnMapping,
    TaskInputSpec,
    TaskOutputSpec,
    TaskConfig,
    ProjectConfig,
)
from ml_platform.config.validators import (
    ValidationResult,
    ConfigValidator,
    diff_configs,
    print_validation_report,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_config():
    """Create a fully valid project configuration."""
    return ProjectConfig(
        sources={
            "source1": SourceConfig(
                type="hive",
                path="db.table1",
                columns={
                    "id": ColumnDefinition(dtype="string"),
                    "price": ColumnDefinition(dtype="double", provides=["win_price"]),
                }
            ),
        },
        features={
            "win_price": FeatureDefinition(dtype="float"),
        },
        tasks={
            "etl_task": TaskConfig(
                name="etl_task",
                type="etl",
                description="Test ETL task",
                input=TaskInputSpec(source="source1"),
                output=TaskOutputSpec(path="/out"),
                params={
                    "group_by": ["id"],
                    "time_column": "ts",
                    "aggregations": [{"source_col": "price", "agg_func": "sum", "output_col": "total"}]
                }
            ),
            "training_task": TaskConfig(
                name="training_task",
                type="training",
                description="Test training task",
                input=TaskInputSpec(source="source1"),
                output=TaskOutputSpec(path="/model"),
                params={
                    "learner": "xgboost",
                    "target_col": "label"
                }
            ),
        }
    )


@pytest.fixture
def config_with_unused():
    """Create a config with unused sources and features."""
    return ProjectConfig(
        sources={
            "used_source": SourceConfig(
                type="hive",
                path="db.used",
                columns={"id": ColumnDefinition(dtype="string")}
            ),
            "unused_source": SourceConfig(
                type="hive",
                path="db.unused",
                columns={"id": ColumnDefinition(dtype="string")}
            ),
        },
        features={
            "used_feature": FeatureDefinition(dtype="string"),
            "unused_feature": FeatureDefinition(dtype="int"),
        },
        tasks={
            "task1": TaskConfig(
                name="task1",
                type="etl",
                input=TaskInputSpec(
                    source="used_source",
                    column_mappings=[
                        ColumnMapping(source_col="id", feature="used_feature")
                    ]
                ),
                output=TaskOutputSpec(path="/out"),
                params={"group_by": [], "time_column": "ts", "aggregations": []}
            ),
        }
    )


@pytest.fixture
def config_with_missing_params():
    """Create a config with tasks missing required params."""
    return ProjectConfig(
        sources={
            "src": SourceConfig(
                type="hive",
                path="db.t",
                columns={"id": ColumnDefinition(dtype="string")}
            ),
        },
        features={},
        tasks={
            "bad_etl": TaskConfig(
                name="bad_etl",
                type="etl",
                input=TaskInputSpec(source="src"),
                output=TaskOutputSpec(path="/out"),
                params={}  # Missing group_by, time_column, aggregations
            ),
            "bad_training": TaskConfig(
                name="bad_training",
                type="training",
                input=TaskInputSpec(source="src"),
                output=TaskOutputSpec(path="/out"),
                params={}  # Missing learner, target_col
            ),
            "no_description": TaskConfig(
                name="no_description",
                type="etl",
                input=TaskInputSpec(source="src"),
                output=TaskOutputSpec(path="/out"),
                params={"group_by": [], "time_column": "ts", "aggregations": []}
                # Missing description
            ),
        }
    )


# =============================================================================
# ValidationResult Tests
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_result_with_errors(self):
        """Test creating an invalid result with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1


# =============================================================================
# ConfigValidator Tests
# =============================================================================

class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_validate_all_valid_config(self, valid_config):
        """Test validate_all on a valid configuration."""
        validator = ConfigValidator(valid_config)
        result = validator.validate_all()
        
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_all_aggregates_results(self, config_with_missing_params):
        """Test validate_all aggregates errors and warnings."""
        validator = ConfigValidator(config_with_missing_params)
        result = validator.validate_all()
        
        # Should have errors for missing params
        assert result.is_valid is False
        assert len(result.errors) > 0
        
        # Should have warnings for missing descriptions
        assert len(result.warnings) > 0


class TestValidateReferences:
    """Tests for _validate_references method."""

    def test_valid_source_references(self, valid_config):
        """Test validation passes for valid source references."""
        validator = ConfigValidator(valid_config)
        result = validator._validate_references()
        
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_invalid_source_reference(self):
        """Test validation catches non-existent source reference."""
        # Note: We need to bypass ProjectConfig's built-in validation
        # by creating a minimal config that won't trigger model validation
        config = ProjectConfig(
            sources={},
            features={},
            tasks={}
        )
        # Manually add a task with bad reference (bypassing validation)
        config.tasks["bad_task"] = TaskConfig(
            name="bad_task",
            type="etl",
            input=TaskInputSpec(source="nonexistent_source"),
            output=TaskOutputSpec(path="/out"),
            params={}
        )
        
        validator = ConfigValidator(config)
        result = validator._validate_references()
        
        assert result.is_valid is False
        assert any("nonexistent_source" in err for err in result.errors)

    def test_invalid_feature_reference(self):
        """Test validation catches non-existent feature reference."""
        config = ProjectConfig(
            sources={
                "src": SourceConfig(
                    type="hive",
                    path="db.t",
                    columns={"col": ColumnDefinition(dtype="string")}
                )
            },
            features={},
            tasks={}
        )
        config.tasks["bad_task"] = TaskConfig(
            name="bad_task",
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
        
        validator = ConfigValidator(config)
        result = validator._validate_references()
        
        assert result.is_valid is False
        assert any("nonexistent_feature" in err for err in result.errors)


class TestValidateUnused:
    """Tests for _validate_unused method."""

    def test_detects_unused_sources(self, config_with_unused):
        """Test detection of unused sources."""
        validator = ConfigValidator(config_with_unused)
        result = validator._validate_unused()
        
        assert any("unused_source" in warn for warn in result.warnings)

    def test_detects_unused_features(self, config_with_unused):
        """Test detection of unused features."""
        validator = ConfigValidator(config_with_unused)
        result = validator._validate_unused()
        
        assert any("unused_feature" in warn for warn in result.warnings)

    def test_no_warnings_for_fully_used(self, valid_config):
        """Test no warnings when all sources are used."""
        validator = ConfigValidator(valid_config)
        result = validator._validate_unused()
        
        # May have some warnings about unused features (not all features mapped)
        # but should not have "source1" as unused
        assert not any("source1" in warn and "unused" in warn.lower() for warn in result.warnings)


class TestValidateTasks:
    """Tests for _validate_tasks method."""

    def test_valid_etl_task(self, valid_config):
        """Test valid ETL task passes validation."""
        validator = ConfigValidator(valid_config)
        result = validator._validate_tasks()
        
        etl_errors = [e for e in result.errors if "etl_task" in e]
        assert len(etl_errors) == 0

    def test_valid_training_task(self, valid_config):
        """Test valid training task passes validation."""
        validator = ConfigValidator(valid_config)
        result = validator._validate_tasks()
        
        training_errors = [e for e in result.errors if "training_task" in e]
        assert len(training_errors) == 0

    def test_etl_missing_group_by(self, config_with_missing_params):
        """Test ETL task missing group_by is flagged."""
        validator = ConfigValidator(config_with_missing_params)
        result = validator._validate_tasks()
        
        assert any("group_by" in err for err in result.errors)

    def test_etl_missing_time_column(self, config_with_missing_params):
        """Test ETL task missing time_column is flagged."""
        validator = ConfigValidator(config_with_missing_params)
        result = validator._validate_tasks()
        
        assert any("time_column" in err for err in result.errors)

    def test_etl_missing_aggregations(self, config_with_missing_params):
        """Test ETL task missing aggregations is flagged."""
        validator = ConfigValidator(config_with_missing_params)
        result = validator._validate_tasks()
        
        assert any("aggregations" in err for err in result.errors)

    def test_training_missing_learner(self, config_with_missing_params):
        """Test training task missing learner is flagged."""
        validator = ConfigValidator(config_with_missing_params)
        result = validator._validate_tasks()
        
        assert any("learner" in err for err in result.errors)

    def test_training_missing_target_col(self, config_with_missing_params):
        """Test training task missing target_col is flagged."""
        validator = ConfigValidator(config_with_missing_params)
        result = validator._validate_tasks()
        
        assert any("target_col" in err for err in result.errors)

    def test_task_missing_description_warning(self, config_with_missing_params):
        """Test task without description generates warning."""
        validator = ConfigValidator(config_with_missing_params)
        result = validator._validate_tasks()
        
        assert any("no description" in warn for warn in result.warnings)


# =============================================================================
# diff_configs Tests
# =============================================================================

class TestDiffConfigs:
    """Tests for diff_configs function."""

    def test_no_changes(self, valid_config):
        """Test diff with identical configs."""
        diff = diff_configs(valid_config, valid_config)
        
        assert diff["added_sources"] == []
        assert diff["removed_sources"] == []
        assert diff["added_features"] == []
        assert diff["removed_features"] == []
        assert diff["added_tasks"] == []
        assert diff["removed_tasks"] == []
        assert diff["modified_tasks"] == []

    def test_added_source(self, valid_config):
        """Test detecting added source."""
        old_config = ProjectConfig(sources={}, features={}, tasks={})
        
        diff = diff_configs(old_config, valid_config)
        
        assert "source1" in diff["added_sources"]

    def test_removed_source(self, valid_config):
        """Test detecting removed source."""
        new_config = ProjectConfig(sources={}, features={}, tasks={})
        
        diff = diff_configs(valid_config, new_config)
        
        assert "source1" in diff["removed_sources"]

    def test_added_feature(self, valid_config):
        """Test detecting added feature."""
        old_config = ProjectConfig(sources={}, features={}, tasks={})
        
        diff = diff_configs(old_config, valid_config)
        
        assert "win_price" in diff["added_features"]

    def test_removed_feature(self, valid_config):
        """Test detecting removed feature."""
        new_config = ProjectConfig(sources={}, features={}, tasks={})
        
        diff = diff_configs(valid_config, new_config)
        
        assert "win_price" in diff["removed_features"]

    def test_added_task(self, valid_config):
        """Test detecting added task."""
        old_config = ProjectConfig(
            sources=valid_config.sources,
            features=valid_config.features,
            tasks={}
        )
        
        diff = diff_configs(old_config, valid_config)
        
        assert "etl_task" in diff["added_tasks"]
        assert "training_task" in diff["added_tasks"]

    def test_removed_task(self, valid_config):
        """Test detecting removed task."""
        new_config = ProjectConfig(
            sources=valid_config.sources,
            features=valid_config.features,
            tasks={}
        )
        
        diff = diff_configs(valid_config, new_config)
        
        assert "etl_task" in diff["removed_tasks"]
        assert "training_task" in diff["removed_tasks"]

    def test_modified_task(self, valid_config):
        """Test detecting modified task."""
        # Create a new config with modified task
        new_config = ProjectConfig(
            sources=valid_config.sources,
            features=valid_config.features,
            tasks={
                "etl_task": TaskConfig(
                    name="etl_task",
                    type="etl",
                    description="MODIFIED description",  # Changed
                    input=TaskInputSpec(source="source1"),
                    output=TaskOutputSpec(path="/out"),
                    params={
                        "group_by": ["id"],
                        "time_column": "ts",
                        "aggregations": []
                    }
                ),
                "training_task": valid_config.tasks["training_task"],
            }
        )
        
        diff = diff_configs(valid_config, new_config)
        
        assert "etl_task" in diff["modified_tasks"]
        assert "training_task" not in diff["modified_tasks"]


# =============================================================================
# print_validation_report Tests
# =============================================================================

class TestPrintValidationReport:
    """Tests for print_validation_report function."""

    def test_print_valid_report(self):
        """Test printing a valid report."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Capture stdout
        captured = StringIO()
        sys.stdout = captured
        try:
            print_validation_report(result)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        assert "VALID" in output
        assert "Configuration Validation Report" in output

    def test_print_report_with_errors(self):
        """Test printing a report with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error message 1", "Error message 2"],
            warnings=[]
        )
        
        captured = StringIO()
        sys.stdout = captured
        try:
            print_validation_report(result)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        assert "ERRORS" in output
        assert "Error message 1" in output
        assert "Error message 2" in output

    def test_print_report_with_warnings(self):
        """Test printing a report with warnings."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Warning message 1"]
        )
        
        captured = StringIO()
        sys.stdout = captured
        try:
            print_validation_report(result)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        assert "WARNINGS" in output
        assert "Warning message 1" in output

    def test_print_report_with_both(self):
        """Test printing a report with both errors and warnings."""
        result = ValidationResult(
            is_valid=False,
            errors=["Critical error"],
            warnings=["Minor warning"]
        )
        
        captured = StringIO()
        sys.stdout = captured
        try:
            print_validation_report(result)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        assert "ERRORS" in output
        assert "WARNINGS" in output
        assert "Critical error" in output
        assert "Minor warning" in output

