"""
Configuration Validators for the ML Platform.

This module provides utilities to:
1. Validate configuration integrity
2. Detect unused or redundant settings
3. Check for common configuration errors
4. Generate configuration diff reports

These address the paper's requirements:
- Hard to make manual errors
- Detect unused settings
- Easy to validate facts
"""

from typing import List, Dict
from dataclasses import dataclass

from ml_platform.config.models import ProjectConfig


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigValidator:
    """
    Validates project configuration for integrity and best practices.
    """

    def __init__(self, config: ProjectConfig):
        self.config = config

    def validate_all(self) -> ValidationResult:
        """Run all validation checks and return aggregated results."""
        errors = []
        warnings = []

        # Check for dangling references
        ref_result = self._validate_references()
        errors.extend(ref_result.errors)
        warnings.extend(ref_result.warnings)

        # Check for unused configurations
        unused_result = self._validate_unused()
        warnings.extend(unused_result.warnings)

        # Check task configurations
        task_result = self._validate_tasks()
        errors.extend(task_result.errors)
        warnings.extend(task_result.warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_references(self) -> ValidationResult:
        """Check that all references point to existing entities."""
        errors = []
        warnings = []

        for task_name, task in self.config.tasks.items():
            # Check source reference
            if task.input.source not in self.config.sources:
                errors.append(
                    f"Task '{task_name}' references non-existent source '{task.input.source}'"
                )

            # Check column mappings reference valid features
            if task.input.column_mappings:
                for mapping in task.input.column_mappings:
                    if mapping.feature not in self.config.features:
                        errors.append(
                            f"Task '{task_name}' maps to non-existent feature '{mapping.feature}'"
                        )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    def _validate_unused(self) -> ValidationResult:
        """Detect unused configurations (Underutilized Data Dependencies)."""
        warnings = []

        unused_sources = self.config.get_unused_sources()
        for source in unused_sources:
            warnings.append(f"Source '{source}' is defined but never used")

        unused_features = self.config.get_unused_features()
        for feat in unused_features:
            warnings.append(f"Feature '{feat}' is defined but never used")

        return ValidationResult(is_valid=True, errors=[], warnings=warnings)

    def _validate_tasks(self) -> ValidationResult:
        """Validate task configurations."""
        errors = []
        warnings = []

        for task_name, task in self.config.tasks.items():
            # Check ETL tasks have required params
            if task.type == "etl":
                if "group_by" not in task.params:
                    errors.append(f"ETL task '{task_name}' missing required param 'group_by'")
                if "time_column" not in task.params:
                    errors.append(f"ETL task '{task_name}' missing required param 'time_column'")
                if "aggregations" not in task.params:
                    errors.append(f"ETL task '{task_name}' missing required param 'aggregations'")

            # Check training tasks have required params
            if task.type == "training":
                if "learner" not in task.params:
                    errors.append(f"Training task '{task_name}' missing param 'learner'")
                if "target_col" not in task.params:
                    errors.append(f"Training task '{task_name}' missing param 'target_col'")

            # Warn if no description
            if not task.description:
                warnings.append(f"Task '{task_name}' has no description")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def diff_configs(old_config: ProjectConfig, new_config: ProjectConfig) -> Dict[str, List[str]]:
    """
    Generate a diff between two configurations.

    Returns a dict with keys: added, removed, modified
    """
    diff = {
        "added_sources": [],
        "removed_sources": [],
        "added_features": [],
        "removed_features": [],
        "added_tasks": [],
        "removed_tasks": [],
        "modified_tasks": [],
    }

    # Compare sources
    old_sources = set(old_config.sources.keys())
    new_sources = set(new_config.sources.keys())
    diff["added_sources"] = list(new_sources - old_sources)
    diff["removed_sources"] = list(old_sources - new_sources)

    # Compare features
    old_features = set(old_config.features.keys())
    new_features = set(new_config.features.keys())
    diff["added_features"] = list(new_features - old_features)
    diff["removed_features"] = list(old_features - new_features)

    # Compare tasks
    old_tasks = set(old_config.tasks.keys())
    new_tasks = set(new_config.tasks.keys())
    diff["added_tasks"] = list(new_tasks - old_tasks)
    diff["removed_tasks"] = list(old_tasks - new_tasks)

    # Check for modified tasks
    for task_name in old_tasks & new_tasks:
        old_task = old_config.tasks[task_name]
        new_task = new_config.tasks[task_name]
        if old_task.model_dump() != new_task.model_dump():
            diff["modified_tasks"].append(task_name)

    return diff


def print_validation_report(result: ValidationResult) -> None:
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("📋 Configuration Validation Report")
    print("=" * 60)

    if result.errors:
        print("\n❌ ERRORS:")
        for err in result.errors:
            print(f"   • {err}")

    if result.warnings:
        print("\n⚠️  WARNINGS:")
        for warn in result.warnings:
            print(f"   • {warn}")

    if result.is_valid:
        print("\n✅ Configuration is VALID")
    else:
        print("\n❌ Configuration has ERRORS")

    print("=" * 60 + "\n")
