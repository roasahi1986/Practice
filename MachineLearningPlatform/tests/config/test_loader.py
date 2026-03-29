"""
Tests for ml_platform/config/loader.py

Tests cover:
- ConfigLoader initialization
- Loading YAML files (single files and directories)
- Jinja2 template rendering in configs
- Error handling for missing/invalid files
- Full load_all() flow
- load_task() for single task loading
- load_config() convenience function
"""

import pytest
import tempfile
import os
from pathlib import Path

from ml_platform.config.loader import ConfigLoader, load_config
from ml_platform.config.models import ProjectConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        
        # Create features.yaml
        features_yaml = """
country_code:
  dtype: "string"
  description: "Country code"
  
win_price:
  dtype: "float"
  description: "Winning bid price"
"""
        (config_dir / "features.yaml").write_text(features_yaml)
        
        # Create sources directory
        sources_dir = config_dir / "sources"
        sources_dir.mkdir()
        
        # Create source files
        auction_logs_yaml = """
auction_logs:
  type: "hive"
  path: "db.auction_logs"
  time_column: "event_ts"
  columns:
    event_id:
      dtype: "string"
    event_ts:
      dtype: "timestamp"
    price:
      dtype: "double"
      provides:
        - "win_price"
    country:
      dtype: "string"
      provides:
        - "country_code"
"""
        (sources_dir / "auction_logs.yaml").write_text(auction_logs_yaml)
        
        # Create tasks.yaml with Jinja2 templates
        tasks_yaml = """
_settings:
  output_root: "s3://test-bucket/output"

aggregate_prices:
  type: "etl"
  description: "Aggregate auction prices"
  input:
    source: "auction_logs"
    date_range:
      from: ~
      to: ~
  params:
    group_by:
      - "country"
    time_column: "event_ts"
    aggregations:
      - source_col: "price"
        agg_func: "sum"
        output_col: "total_price"
  output:
    path: "{{ output_root }}/{{ type }}/{{ name }}"
    format: "parquet"
    mode: "overwrite"

train_model:
  type: "training"
  description: "Train prediction model"
  input:
    source: "auction_logs"
    date_range:
      from: "2025-01-01"
      to: "2025-01-31"
  params:
    learner: "xgboost"
    target_col: "win_price"
  output:
    path: "{{ output_root }}/models/{{ name }}"
    format: "mlflow"
"""
        (config_dir / "tasks.yaml").write_text(tasks_yaml)
        
        yield config_dir


@pytest.fixture
def minimal_config_dir():
    """Create a minimal config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        
        (config_dir / "features.yaml").write_text("")
        (config_dir / "sources.yaml").write_text("")
        (config_dir / "tasks.yaml").write_text("")
        
        yield config_dir


@pytest.fixture
def config_with_invalid_yaml():
    """Create a config directory with invalid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        
        # Invalid YAML syntax
        (config_dir / "features.yaml").write_text("invalid: yaml: syntax:")
        (config_dir / "sources.yaml").write_text("")
        (config_dir / "tasks.yaml").write_text("")
        
        yield config_dir


# =============================================================================
# ConfigLoader Initialization Tests
# =============================================================================

class TestConfigLoaderInit:
    """Tests for ConfigLoader initialization."""

    def test_init_with_string_path(self, temp_config_dir):
        """Test initialization with string path."""
        loader = ConfigLoader(str(temp_config_dir))
        assert loader.config_dir == temp_config_dir

    def test_init_default_path(self):
        """Test initialization with default path."""
        loader = ConfigLoader()
        assert loader.config_dir == Path("config")

    def test_internal_state_initialized(self, temp_config_dir):
        """Test internal state is properly initialized."""
        loader = ConfigLoader(str(temp_config_dir))
        assert loader._sources == {}
        assert loader._features == {}
        assert loader._tasks == {}


# =============================================================================
# YAML Loading Tests
# =============================================================================

class TestYamlLoading:
    """Tests for YAML file loading."""

    def test_load_yaml_file(self, temp_config_dir):
        """Test loading a single YAML file."""
        loader = ConfigLoader(str(temp_config_dir))
        data = loader._load_yaml("features.yaml")
        
        assert "country_code" in data
        assert "win_price" in data
        assert data["country_code"]["dtype"] == "string"

    def test_load_yaml_missing_file(self, temp_config_dir):
        """Test loading missing file returns empty dict."""
        loader = ConfigLoader(str(temp_config_dir))
        data = loader._load_yaml("nonexistent.yaml")
        assert data == {}

    def test_load_yaml_empty_file(self, minimal_config_dir):
        """Test loading empty YAML file returns empty dict."""
        loader = ConfigLoader(str(minimal_config_dir))
        data = loader._load_yaml("features.yaml")
        assert data == {}

    def test_load_yaml_dir(self, temp_config_dir):
        """Test loading all YAML files from a directory."""
        loader = ConfigLoader(str(temp_config_dir))
        data = loader._load_yaml_dir("sources")
        
        assert "auction_logs" in data
        assert data["auction_logs"]["type"] == "hive"

    def test_load_yaml_dir_missing(self, temp_config_dir):
        """Test loading from missing directory returns empty dict."""
        loader = ConfigLoader(str(temp_config_dir))
        data = loader._load_yaml_dir("nonexistent_dir")
        assert data == {}

    def test_load_yaml_dir_ignores_underscore_files(self, temp_config_dir):
        """Test that files starting with _ are ignored."""
        sources_dir = temp_config_dir / "sources"
        (sources_dir / "_internal.yaml").write_text("internal: true")
        
        loader = ConfigLoader(str(temp_config_dir))
        data = loader._load_yaml_dir("sources")
        
        assert "internal" not in data


# =============================================================================
# Sources Loading Tests
# =============================================================================

class TestSourcesLoading:
    """Tests for sources loading."""

    def test_load_sources_from_directory(self, temp_config_dir):
        """Test loading sources from sources/ directory."""
        loader = ConfigLoader(str(temp_config_dir))
        loader._load_sources()
        
        assert "auction_logs" in loader._sources
        source = loader._sources["auction_logs"]
        assert source.type == "hive"
        assert source.path == "db.auction_logs"
        assert "event_id" in source.columns

    def test_load_sources_fallback_to_file(self, minimal_config_dir):
        """Test fallback to sources.yaml when directory doesn't exist."""
        # Add a sources.yaml file
        sources_yaml = """
test_source:
  type: "hive"
  path: "db.test"
  columns:
    id:
      dtype: "string"
"""
        (minimal_config_dir / "sources.yaml").write_text(sources_yaml)
        
        loader = ConfigLoader(str(minimal_config_dir))
        loader._load_sources()
        
        assert "test_source" in loader._sources


# =============================================================================
# Features Loading Tests
# =============================================================================

class TestFeaturesLoading:
    """Tests for features loading."""

    def test_load_features(self, temp_config_dir):
        """Test loading features from features.yaml."""
        loader = ConfigLoader(str(temp_config_dir))
        loader._load_features()
        
        assert "country_code" in loader._features
        assert "win_price" in loader._features
        assert loader._features["country_code"].dtype == "string"


# =============================================================================
# Tasks Loading Tests
# =============================================================================

class TestTasksLoading:
    """Tests for tasks loading with Jinja2 template rendering."""

    def test_load_tasks(self, temp_config_dir):
        """Test loading tasks from tasks.yaml."""
        loader = ConfigLoader(str(temp_config_dir))
        loader._load_tasks()
        
        assert "aggregate_prices" in loader._tasks
        assert "train_model" in loader._tasks

    def test_settings_extracted(self, temp_config_dir):
        """Test _settings is extracted and not treated as a task."""
        loader = ConfigLoader(str(temp_config_dir))
        loader._load_tasks()
        
        assert "_settings" not in loader._tasks
        assert loader._settings.output_root == "s3://test-bucket/output"

    def test_jinja2_rendering_output_path(self, temp_config_dir):
        """Test Jinja2 templates are rendered in output paths."""
        loader = ConfigLoader(str(temp_config_dir))
        loader._load_tasks()
        
        # Check aggregate_prices task
        agg_task = loader._tasks["aggregate_prices"]
        expected_path = "s3://test-bucket/output/etl/aggregate_prices"
        assert agg_task.output.path == expected_path
        
        # Check train_model task
        train_task = loader._tasks["train_model"]
        expected_path = "s3://test-bucket/output/models/train_model"
        assert train_task.output.path == expected_path

    def test_task_name_auto_populated(self, temp_config_dir):
        """Test task name is auto-populated if not in config."""
        loader = ConfigLoader(str(temp_config_dir))
        loader._load_tasks()
        
        assert loader._tasks["aggregate_prices"].name == "aggregate_prices"


# =============================================================================
# Jinja2 Template Rendering Tests
# =============================================================================

class TestJinja2Rendering:
    """Tests for Jinja2 template rendering."""

    def test_render_string(self, temp_config_dir):
        """Test rendering a simple string template."""
        loader = ConfigLoader(str(temp_config_dir))
        context = {"name": "test", "type": "etl"}
        result = loader._render_jinja2("{{ name }}_{{ type }}", context)
        assert result == "test_etl"

    def test_render_no_template(self, temp_config_dir):
        """Test rendering a string without templates returns as-is."""
        loader = ConfigLoader(str(temp_config_dir))
        context = {"name": "test"}
        result = loader._render_jinja2("no_template_here", context)
        assert result == "no_template_here"

    def test_render_dict_recursive(self, temp_config_dir):
        """Test rendering templates in nested dict."""
        loader = ConfigLoader(str(temp_config_dir))
        context = {"root": "/data", "name": "test"}
        input_dict = {
            "path": "{{ root }}/{{ name }}",
            "nested": {
                "value": "{{ name }}_nested"
            }
        }
        result = loader._render_jinja2(input_dict, context)
        assert result["path"] == "/data/test"
        assert result["nested"]["value"] == "test_nested"

    def test_render_list_recursive(self, temp_config_dir):
        """Test rendering templates in list."""
        loader = ConfigLoader(str(temp_config_dir))
        context = {"prefix": "pre"}
        input_list = ["{{ prefix }}_one", "{{ prefix }}_two"]
        result = loader._render_jinja2(input_list, context)
        assert result == ["pre_one", "pre_two"]

    def test_render_non_string_passthrough(self, temp_config_dir):
        """Test non-string values pass through unchanged."""
        loader = ConfigLoader(str(temp_config_dir))
        context = {}
        assert loader._render_jinja2(42, context) == 42
        assert loader._render_jinja2(3.14, context) == 3.14
        assert loader._render_jinja2(True, context) is True
        assert loader._render_jinja2(None, context) is None


# =============================================================================
# Full Load Tests
# =============================================================================

class TestFullLoad:
    """Tests for full configuration loading."""

    def test_load_all_returns_project_config(self, temp_config_dir):
        """Test load_all returns a valid ProjectConfig."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.load_all()
        
        assert isinstance(config, ProjectConfig)
        assert len(config.sources) >= 1
        assert len(config.features) >= 1
        assert len(config.tasks) >= 1

    def test_load_all_validates_references(self, temp_config_dir):
        """Test load_all validates source references."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.load_all()
        
        # Should not raise because auction_logs exists
        assert config.tasks["aggregate_prices"].input.source == "auction_logs"

    def test_load_task_single(self, temp_config_dir):
        """Test loading a single task by name."""
        loader = ConfigLoader(str(temp_config_dir))
        task = loader.load_task("aggregate_prices")
        
        assert task.name == "aggregate_prices"
        assert task.type == "etl"

    def test_load_task_not_found(self, temp_config_dir):
        """Test loading non-existent task raises error."""
        loader = ConfigLoader(str(temp_config_dir))
        with pytest.raises(ValueError, match="not found"):
            loader.load_task("nonexistent_task")


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunction:
    """Tests for load_config convenience function."""

    def test_load_config_function(self, temp_config_dir):
        """Test load_config returns ProjectConfig."""
        config = load_config(str(temp_config_dir))
        assert isinstance(config, ProjectConfig)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in config loading."""

    def test_invalid_source_raises_value_error(self, temp_config_dir):
        """Test invalid source config raises ValueError."""
        sources_dir = temp_config_dir / "sources"
        (sources_dir / "invalid.yaml").write_text("""
invalid_source:
  type: "invalid_type"
  path: "test"
  columns: {}
""")
        
        loader = ConfigLoader(str(temp_config_dir))
        with pytest.raises(ValueError, match="Error parsing source"):
            loader._load_sources()

    def test_invalid_task_raises_value_error(self, temp_config_dir):
        """Test invalid task config raises ValueError."""
        tasks_yaml = """
invalid_task:
  type: "etl"
  # Missing required 'input' and 'output'
"""
        (temp_config_dir / "tasks.yaml").write_text(tasks_yaml)
        
        loader = ConfigLoader(str(temp_config_dir))
        with pytest.raises(ValueError, match="Error parsing task"):
            loader._load_tasks()


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_config_directory(self, minimal_config_dir):
        """Test loading from directory with empty files."""
        loader = ConfigLoader(str(minimal_config_dir))
        config = loader.load_all()
        
        assert config.sources == {}
        assert config.features == {}
        assert config.tasks == {}

    def test_multiple_source_files_merged(self, temp_config_dir):
        """Test multiple source files in directory are merged."""
        sources_dir = temp_config_dir / "sources"
        
        # Add another source file
        (sources_dir / "other_sources.yaml").write_text("""
other_source:
  type: "delta"
  path: "db.other"
  columns:
    id:
      dtype: "string"
""")
        
        loader = ConfigLoader(str(temp_config_dir))
        loader._load_sources()
        
        assert "auction_logs" in loader._sources
        assert "other_source" in loader._sources

