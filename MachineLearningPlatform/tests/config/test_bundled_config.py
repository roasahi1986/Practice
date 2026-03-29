"""
Tests for loading the bundled configuration files.

This test ensures that the YAML configs bundled with ml_platform
can be successfully loaded and validated.
"""

import pytest
from pathlib import Path

from ml_platform.config.loader import ConfigLoader
from ml_platform.config.models import ProjectConfig
import ml_platform.conf as conf_pkg


class TestBundledConfig:
    """Tests for loading the bundled configuration."""

    @pytest.fixture
    def config_dir(self):
        """Get the bundled config directory."""
        return str(Path(conf_pkg.__file__).parent)

    @pytest.fixture
    def loader(self, config_dir):
        """Create a ConfigLoader for the bundled config."""
        return ConfigLoader(config_dir)

    def test_config_dir_exists(self, config_dir):
        """Test that the bundled config directory exists."""
        assert Path(config_dir).exists()
        assert Path(config_dir).is_dir()

    def test_required_files_exist(self, config_dir):
        """Test that required config files exist."""
        config_path = Path(config_dir)
        
        assert (config_path / "features.yaml").exists()
        assert (config_path / "tasks.yaml").exists()
        assert (config_path / "sources").is_dir()

    def test_load_all_succeeds(self, loader):
        """Test that load_all() succeeds without errors."""
        config = loader.load_all()
        assert isinstance(config, ProjectConfig)

    def test_sources_loaded(self, loader):
        """Test that sources are loaded correctly."""
        config = loader.load_all()
        
        assert len(config.sources) >= 1
        
        # Check expected sources exist
        assert "downsampled_auction_logs" in config.sources
        assert "aggregate_hourly_win_price" in config.sources

    def test_features_loaded(self, loader):
        """Test that features are loaded correctly."""
        config = loader.load_all()
        
        assert len(config.features) >= 1
        
        # Check expected features exist
        assert "device_os" in config.features
        assert "device_region" in config.features

    def test_tasks_loaded(self, loader):
        """Test that tasks are loaded correctly."""
        config = loader.load_all()
        
        assert len(config.tasks) >= 1
        
        # Check expected tasks exist
        assert "aggregate_hourly_win_price" in config.tasks
        assert "compute_rolling_win_price" in config.tasks

    def test_etl_tasks_have_valid_params(self, loader):
        """Test that ETL tasks have valid parameters."""
        config = loader.load_all()
        
        for name, task in config.tasks.items():
            if task.type == "etl":
                # Should be able to parse ETL params
                etl_params = task.get_etl_params()
                assert etl_params.group_by is not None
                assert etl_params.time_column is not None

    def test_training_tasks_have_valid_params(self, loader):
        """Test that training tasks have valid parameters."""
        config = loader.load_all()
        
        for name, task in config.tasks.items():
            if task.type == "training":
                # Should be able to parse training params
                train_params = task.get_training_params()
                assert train_params.learner is not None
                assert train_params.target_col is not None

    def test_jinja2_templates_rendered(self, loader):
        """Test that Jinja2 templates in output paths are rendered."""
        config = loader.load_all()
        
        for name, task in config.tasks.items():
            # Output path should not contain {{ }} after rendering
            assert "{{" not in task.output.path
            assert "}}" not in task.output.path

    def test_task_sources_exist(self, loader):
        """Test that all tasks reference existing sources."""
        config = loader.load_all()
        
        for name, task in config.tasks.items():
            source_name = task.input.source
            assert source_name in config.sources, \
                f"Task '{name}' references non-existent source '{source_name}'"

    def test_column_mappings_valid(self, loader):
        """Test that column mappings reference valid columns and features."""
        config = loader.load_all()
        
        for task_name, task in config.tasks.items():
            if task.input.column_mappings:
                source = config.sources[task.input.source]
                
                for mapping in task.input.column_mappings:
                    # Source column must exist
                    assert mapping.source_col in source.columns, \
                        f"Task '{task_name}': column '{mapping.source_col}' not in source"
                    
                    # Feature must exist
                    assert mapping.feature in config.features, \
                        f"Task '{task_name}': feature '{mapping.feature}' not defined"

    def test_settings_loaded(self, loader):
        """Test that global settings are loaded."""
        config = loader.load_all()
        
        # Settings should be populated
        assert config.settings is not None
        assert config.settings.output_root is not None

    def test_source_columns_have_dtypes(self, loader):
        """Test that all source columns have valid dtypes."""
        config = loader.load_all()
        
        valid_dtypes = {"string", "int", "long", "float", "double", "boolean", "timestamp", "date"}
        
        for source_name, source in config.sources.items():
            for col_name, col_def in source.columns.items():
                assert col_def.dtype in valid_dtypes, \
                    f"Source '{source_name}' column '{col_name}' has invalid dtype '{col_def.dtype}'"

    def test_feature_lineage_available(self, loader):
        """Test that feature lineage can be retrieved."""
        config = loader.load_all()
        
        # Get lineage for a feature that should be provided by a source
        lineage = config.get_feature_lineage("device_os")
        
        # Should have at least one source providing this feature
        assert len(lineage) >= 1

    def test_get_unused_sources(self, loader):
        """Test that unused sources detection works."""
        config = loader.load_all()
        
        unused = config.get_unused_sources()
        # Should return a list (may or may not be empty)
        assert isinstance(unused, list)

    def test_get_unused_features(self, loader):
        """Test that unused features detection works."""
        config = loader.load_all()
        
        unused = config.get_unused_features()
        # Should return a list (may or may not be empty)
        assert isinstance(unused, list)

