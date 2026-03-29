"""
ML Platform CLI Entry Point.

Usage (after pip install):
    ml-task --task <task_name>
    ml-task --task aggregate_hourly_win_price --from 2025-11-27 --to 2025-11-28
    ml-task --list-tasks

Usage (development):
    python main.py --task <task_name>
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from ml_platform.config.loader import ConfigLoader
from ml_platform.tasks import TrainingTask, ETLTask, EnrichmentTask, VisualizationTask


# =============================================================================
# CLI Arguments Definition (Parser-agnostic)
# =============================================================================

@dataclass
class CLIArgs:
    """Parsed CLI arguments. Parser-agnostic structure."""
    task: Optional[str] = None
    config_dir: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    list_tasks: bool = False


# =============================================================================
# Argument Parser (swappable)
# =============================================================================

def parse_args_argparse(argv: Optional[list] = None) -> CLIArgs:
    """
    Parse CLI arguments using argparse.
    
    To switch to a different parser (click, typer, fire), create a new
    parse_args_xxx() function and update parse_args() to call it.
    
    Args:
        argv: List of command-line arguments. If None, uses sys.argv[1:].
              Examples:
                - None: uses actual command line (default)
                - ["--list-tasks"]: list all tasks
                - ["--task", "aggregate_hourly_win_price", "--from", "2025-11-27", "--to", "2025-11-28"]
                - ["--list-tasks"]: list available tasks
                - ["--task", "train_rtb_model", "--config_dir", "/path/to/config"]
              
    Returns:
        CLIArgs dataclass with parsed values
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ML Platform Task Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ml-task --task aggregate_hourly_win_price --from 2025-11-27 --to 2025-11-28
  ml-task --task compute_rolling_win_price --from 2025-11-27 --to 2025-11-28
  ml-task --task train_rtb_model
  ml-task --list-tasks
        """,
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Name of the task to run (from tasks.yaml)",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Directory containing config files (default: bundled with package)",
    )
    parser.add_argument(
        "--from",
        dest="date_from",
        type=str,
        help="Filter data from this date (inclusive), e.g., 2026-01-01",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        type=str,
        help="Filter data to this date (inclusive), e.g., 2026-01-02",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit",
    )

    args = parser.parse_args(argv)
    
    return CLIArgs(
        task=args.task,
        config_dir=args.config_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        list_tasks=args.list_tasks,
    )


def parse_args(argv: Optional[list] = None) -> CLIArgs:
    """
    Parse CLI arguments.
    
    This is the entry point for argument parsing. To switch parsers,
    just change which parse_args_xxx() function is called here.
    """
    return parse_args_argparse(argv)


# =============================================================================
# Core CLI Logic (parser-agnostic)
# =============================================================================

def get_default_config_dir() -> str:
    """
    Get the config directory bundled with the ml_platform package.
    The YAML configs are in ml_platform/conf/
    """
    import ml_platform.conf as conf_pkg
    return str(Path(conf_pkg.__file__).parent)


def run_cli(args: CLIArgs) -> int:
    """
    Execute CLI commands based on parsed arguments.
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Determine config directory
    config_dir = args.config_dir or get_default_config_dir()
    print(f"📁 Using config from: {config_dir}")

    # Load configuration
    loader = ConfigLoader(config_dir)
    project_config = loader.load_all()

    # Handle --list-tasks
    if args.list_tasks:
        print("\n📋 Available Tasks:")
        print("-" * 60)
        for name, task in project_config.tasks.items():
            print(f"  {name}")
            print(f"      Type: {task.type}")
            print(f"      Description: {task.description or 'N/A'}")
            print()
        return 0


    # Require --task if not listing or validating
    if not args.task:
        print("❌ Error: --task is required")
        print("   Use --list-tasks to see available tasks")
        return 1

    # Check if task exists
    if args.task not in project_config.tasks:
        print(f"❌ Error: Task '{args.task}' not found in configuration.")
        print(f"   Available tasks: {list(project_config.tasks.keys())}")
        return 1

    task_cfg = project_config.tasks[args.task]

    # Build runtime overrides from CLI arguments
    runtime_overrides: Dict[str, Any] = {}
    if args.date_from or args.date_to:
        runtime_overrides["date_range"] = {}
        if args.date_from:
            runtime_overrides["date_range"]["date_from"] = args.date_from
        if args.date_to:
            runtime_overrides["date_range"]["date_to"] = args.date_to

    # Factory: Instantiate the correct Task class based on type
    if task_cfg.type == "training":
        task = TrainingTask(args.task, config_dir, runtime_overrides)
    elif task_cfg.type == "etl":
        task = ETLTask(args.task, config_dir, runtime_overrides)
    elif task_cfg.type == "enrichment":
        task = EnrichmentTask(args.task, config_dir, runtime_overrides)
    elif task_cfg.type == "visualization":
        task = VisualizationTask(args.task, config_dir, runtime_overrides)
    else:
        print(f"❌ Task type '{task_cfg.type}' not yet implemented.")
        print("   Supported types: training, etl, enrichment, visualization")
        return 1

    # Run the task
    task.run()
    return 0


def main(argv: Optional[list] = None, exit_on_complete: bool = False) -> int:
    """
    CLI entry point.
    
    Args:
        argv: Command-line arguments (None = use sys.argv)
        exit_on_complete: If True, call sys.exit(). Set False for notebook use.
        
    Returns:
        Exit code (0 = success)
    """
    args = parse_args(argv)
    exit_code = run_cli(args)
    
    if exit_on_complete:
        sys.exit(exit_code)
    
    return exit_code


if __name__ == "__main__":
    main()
