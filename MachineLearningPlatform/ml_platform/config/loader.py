"""
Configuration Loader for the ML Platform.

This module handles loading and parsing YAML configuration files.
It supports:
- Loading from multiple YAML files
- Loading from directories (e.g., sources/ with multiple .yaml files)
- Jinja2 template variables in YAML values
- Merging configurations into a single ProjectConfig
- Error handling with clear messages

Template variables (Jinja2 syntax):
  {{ output_root }} - Global output root from _settings
  {{ type }}        - Task type (etl, training, etc.)
  {{ name }}        - Task name
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from jinja2 import Template

from ml_platform.config.models import (
    ProjectConfig,
    SourceConfig,
    FeatureDefinition,
    TaskConfig,
    GlobalSettings,
)


class ConfigLoader:
    """
    Loads and parses YAML configuration files.

    Usage:
        loader = ConfigLoader("config")
        config = loader.load_all()
        print(config.sources)
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._sources: Dict[str, SourceConfig] = {}
        self._features: Dict[str, FeatureDefinition] = {}
        self._tasks: Dict[str, TaskConfig] = {}
        self._settings: GlobalSettings = GlobalSettings()

    def load_all(self) -> ProjectConfig:
        """
        Load all configuration files and return a validated ProjectConfig.
        Raises ValueError if validation fails.
        """
        self._load_sources()
        self._load_features()
        self._load_tasks()

        config = ProjectConfig(
            sources=self._sources,
            features=self._features,
            tasks=self._tasks,
            settings=self._settings,
        )

        # Report unused configurations as warnings
        unused_sources = config.get_unused_sources()
        if unused_sources:
            print(f"  ⚠️  Unused sources: {unused_sources}")

        unused_features = config.get_unused_features()
        if unused_features:
            print(f"  ⚠️  Unused features: {unused_features}")

        return config

    def load_task(self, task_name: str) -> TaskConfig:
        """Load a single task by name."""
        self._load_tasks()
        if task_name not in self._tasks:
            raise ValueError(f"Task '{task_name}' not found in configuration.")
        return self._tasks[task_name]

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a single YAML file."""
        path = self.config_dir / filename
        if not path.exists():
            print(f"⚠️  Config file not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            return content if content else {}

    def _load_yaml_dir(self, dirname: str) -> Dict[str, Any]:
        """Load all YAML files from a directory and merge them."""
        dir_path = self.config_dir / dirname
        if not dir_path.exists() or not dir_path.is_dir():
            return {}

        merged: Dict[str, Any] = {}
        yaml_files = sorted(dir_path.glob("*.yaml")) + sorted(dir_path.glob("*.yml"))

        for yaml_file in yaml_files:
            if yaml_file.name.startswith("_"):
                continue
            with open(yaml_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if content and isinstance(content, dict):
                    merged.update(content)

        return merged

    def _load_sources(self) -> None:
        """
        Load sources from config/sources/ directory.
        Falls back to sources.yaml for backward compatibility.
        """
        sources_dir = self.config_dir / "sources"
        if sources_dir.exists() and sources_dir.is_dir():
            data = self._load_yaml_dir("sources")
        else:
            data = self._load_yaml("sources.yaml")

        for name, cfg in data.items():
            if not isinstance(cfg, dict):
                continue
            try:
                self._sources[name] = SourceConfig(**cfg)
            except Exception as e:
                raise ValueError(f"Error parsing source '{name}': {e}")

    def _load_features(self) -> None:
        """Load features.yaml (individual features at root level)."""
        data = self._load_yaml("features.yaml")
        # Features are at root level (no wrapper key)
        for name, cfg in data.items():
            if not isinstance(cfg, dict):
                continue
            try:
                self._features[name] = FeatureDefinition(**cfg)
            except Exception as e:
                raise ValueError(f"Error parsing feature '{name}': {e}")

    def _render_jinja2(self, value: Any, context: Dict[str, Any]) -> Any:
        """
        Recursively render Jinja2 templates in a value.
        Supports strings, dicts, and lists.
        """
        if isinstance(value, str) and "{{" in value:
            template = Template(value)
            return template.render(**context)
        elif isinstance(value, dict):
            return {k: self._render_jinja2(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._render_jinja2(item, context) for item in value]
        return value

    def _load_tasks(self) -> None:
        """Load tasks.yaml (tasks at root level, with optional _settings)."""
        data = self._load_yaml("tasks.yaml")

        # Extract global settings if present
        if "_settings" in data:
            settings_data = data.pop("_settings")
            if isinstance(settings_data, dict):
                self._settings = GlobalSettings(**settings_data)

        # Build base context for Jinja2 rendering
        base_context = {
            "output_root": self._settings.output_root or "",
        }

        # Parse tasks with Jinja2 template rendering
        for name, cfg in data.items():
            if not isinstance(cfg, dict):
                continue
            if "type" not in cfg:
                continue
            try:
                if "name" not in cfg:
                    cfg["name"] = name

                # Build task-specific context
                context = {
                    **base_context,
                    "name": name,
                    "type": cfg.get("type", ""),
                }

                # Render Jinja2 templates in the config
                cfg = self._render_jinja2(cfg, context)

                self._tasks[name] = TaskConfig(**cfg)
            except Exception as e:
                raise ValueError(f"Error parsing task '{name}': {e}")


def load_config(config_dir: str = "config") -> ProjectConfig:
    """Convenience function to load all configs."""
    return ConfigLoader(config_dir).load_all()
