"""
Base Task for the ML Platform.

This module defines the Template Method pattern for all tasks.
Every task (Training, ETL, Visualization, etc.) inherits from BaseTask
and follows the same execution flow:

1. Load Configuration
2. Load & Validate Data (via Pipeline)
3. Process (task-specific logic)
4. Save Output

This ensures:
- Consistent data loading across all tasks (no "glue code" per script)
- Configuration-driven behavior
- Explicit error handling
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from ml_platform.config.models import TaskConfig, SourceConfig
from ml_platform.config.loader import ConfigLoader


class BaseTask(ABC):
    """
    Abstract base class for all tasks.

    Subclasses must implement:
    - _process(df) -> result
    - _save_output(result)
    """

    def __init__(
        self,
        task_name: str,
        config_dir: str = "config",
        runtime_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the task.

        Args:
            task_name: Name of the task (must exist in tasks.yaml)
            config_dir: Path to configuration directory
            runtime_overrides: Optional overrides for task input (e.g., date_range)
        """
        self.task_name = task_name
        self.runtime_overrides = runtime_overrides or {}
        self.config_loader = ConfigLoader(config_dir)
        self.project_config = self.config_loader.load_all()

        if task_name not in self.project_config.tasks:
            raise ValueError(f"Task '{task_name}' not found in tasks configuration.")

        self.task_config: TaskConfig = self.project_config.tasks[task_name]
        self._apply_runtime_overrides()
        self.spark = self._get_or_create_spark()

    def _apply_runtime_overrides(self) -> None:
        """Apply runtime overrides to task configuration."""
        if "date_range" in self.runtime_overrides:
            from ml_platform.config.models import DateRangeSpec
            
            override_range = self.runtime_overrides["date_range"]
            if self.task_config.input.date_range is None:
                self.task_config.input.date_range = DateRangeSpec()
            if "date_from" in override_range:
                self.task_config.input.date_range.date_from = override_range["date_from"]
            if "date_to" in override_range:
                self.task_config.input.date_range.date_to = override_range["date_to"]

    def _validate_required_params(self) -> None:
        """Validate that required runtime parameters are provided."""
        date_range = self.task_config.input.date_range
        if date_range:
            missing = []
            if not date_range.date_from:
                missing.append("--from")
            if not date_range.date_to:
                missing.append("--to")
            if missing:
                raise ValueError(
                    f"Task '{self.task_name}' requires date range. "
                    f"Missing: {', '.join(missing)}"
                )

    def _get_or_create_spark(self) -> SparkSession:
        """Get existing SparkSession or create a new one."""
        return (
            SparkSession.builder
            .appName(f"Task_{self.task_name}")
            .enableHiveSupport()
            .getOrCreate()
        )

    def run(self) -> Any:
        """
        Template method defining the execution flow.
        This is the main entry point for running a task.
        """
        # Validate required parameters before starting
        self._validate_required_params()

        print("=" * 80)
        print(f"🚀 Starting Task: {self.task_name}")
        print(f"   Type: {self.task_config.type}")
        print(f"   Description: {self.task_config.description or 'N/A'}")
        print("=" * 80)

        # 1. Load Data (Common step)
        print("\n📥 Step 1: Loading Data...")
        df = self._load_data()
        print(f"   Loaded {df.count():,} rows")

        # 2. Process (Task-specific step)
        print("\n⚙️ Step 2: Processing...")
        result = self._process(df)

        # 3. Save Output (Common/Task-specific step)
        print("\n💾 Step 3: Saving Output...")
        self._save_output(result)

        print("\n" + "=" * 80)
        print(f"✅ Task {self.task_name} Completed Successfully!")
        print("=" * 80)

        return result

    def _load_data(self) -> DataFrame:
        """
        Load data from the configured source.
        Handles Hive tables, S3 paths, date filtering, and lookback.
        """
        input_spec = self.task_config.input
        source_name = input_spec.source
        source_config = self.project_config.sources.get(source_name)

        if not source_config:
            raise ValueError(f"Source '{source_name}' not found in configuration.")

        print(f"   Source: {source_name} ({source_config.type})")
        print(f"   Path: {source_config.path}")

        # Load based on source type
        if source_config.type in ("hive", "coba2"):
            df = self._load_from_sql_table(source_config, input_spec)
        elif source_config.type in ("s3_parquet", "parquet"):
            df = self.spark.read.parquet(source_config.path)
        elif source_config.type == "s3_csv":
            df = self.spark.read.csv(source_config.path, header=True, inferSchema=True)
        elif source_config.type == "delta":
            df = self.spark.read.format("delta").load(source_config.path)
        else:
            raise ValueError(f"Unsupported source type: {source_config.type}")

        # Apply filter expression if provided
        if input_spec.filter_expr:
            print(f"   Applying filter: {input_spec.filter_expr}")
            df = df.filter(input_spec.filter_expr)

        # Select specific columns if provided (for raw ETL without feature_set)
        if input_spec.select_columns:
            df = df.select(*input_spec.select_columns)

        return df

    def _get_lookback_hours(self) -> int:
        """
        Get lookback hours from ETL params (derived from rolling windows).
        Returns 0 for non-ETL tasks or if no rolling windows defined.
        """
        if self.task_config.type == "etl":
            try:
                etl_params = self.task_config.get_etl_params()
                return etl_params.get_lookback_hours()
            except Exception:
                return 0
        return 0

    def _get_required_columns(self, source_config: SourceConfig) -> Optional[list]:
        """
        Infer required columns from task params (group_by, aggregations, etc.)
        Returns None if columns cannot be inferred (use SELECT *).
        """
        params = self.task_config.params
        required_cols = set()

        # Add group_by columns
        if "group_by" in params:
            required_cols.update(params["group_by"])

        # Add aggregation source columns
        if "aggregations" in params:
            for agg in params["aggregations"]:
                source_col = agg.get("source_col") if isinstance(agg, dict) else agg.source_col
                if source_col and source_col != "*":
                    required_cols.add(source_col)

        # Add dedupe column (ETL tasks)
        if "dedupe_by" in params:
            required_cols.add(params["dedupe_by"])

        # Add rolling window columns (ETL tasks)
        if "rolling_windows" in params:
            for window in params["rolling_windows"]:
                apply_to = window.get("apply_to") if isinstance(window, dict) else window.apply_to
                if apply_to:
                    required_cols.update(apply_to)

        # Add time column (needed for date/hour extraction and time bucketing)
        if source_config.time_column:
            required_cols.add(source_config.time_column)

        # Add time_column from params if different from source (e.g., time_bucket)
        if "time_column" in params:
            required_cols.add(params["time_column"])

        # If we found columns, return them; otherwise return None to use SELECT *
        return list(required_cols) if required_cols else None

    def _load_from_sql_table(self, source_config: SourceConfig, input_spec) -> DataFrame:
        """
        Load data from a SQL table (Hive, Coba2, etc.) with optional date range filtering.
        Handles lookback for rolling window calculations.
        
        Important: Discovers partition columns dynamically and filters by the first partition
        column to enable partition pruning and avoid expensive full-table metadata scans.
        """
        table_path = source_config.path
        date_range = input_spec.date_range
        lookback_hours = self._get_lookback_hours()
        time_column = source_config.time_column

        # Discover partition columns from table metadata
        first_partition_col = self._get_first_partition_column(table_path)

        # Infer required columns from task params, or use explicit select_columns
        select_cols = input_spec.select_columns or self._get_required_columns(source_config)

        # Build SQL query with specific columns if available
        if select_cols:
            columns_str = ", ".join(select_cols)
            sql = f"SELECT {columns_str} FROM {table_path}"
        else:
            sql = f"SELECT * FROM {table_path}"
        conditions = []

        if date_range and time_column:
            date_from = date_range.date_from.strip() if date_range.date_from else None
            date_to = date_range.date_to.strip() if date_range.date_to else None

            if date_from:
                # Parse start time (support both date and datetime formats)
                try:
                    from_dt = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    from_dt = datetime.strptime(date_from, "%Y-%m-%d")

                # Apply lookback for window calculations
                if lookback_hours > 0:
                    lookback_start = from_dt - timedelta(hours=lookback_hours)
                    actual_from = lookback_start.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"   Date range (with {lookback_hours}h lookback): {actual_from} to {date_to}")
                else:
                    actual_from = date_from
                    print(f"   Date range: {date_from} to {date_to}")

                conditions.append(f"{time_column} >= '{actual_from}'")

            if date_to:
                # If date_to is date-only, add 1 day to include the entire end date
                try:
                    to_dt = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
                    actual_to = date_to  # Already has time, use as-is (exclusive)
                except ValueError:
                    # Date-only: add 1 day so the filter includes the entire end date
                    to_dt = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
                    actual_to = to_dt.strftime("%Y-%m-%d")
                conditions.append(f"{time_column} < '{actual_to}'")

            # Add partition column filter for partition pruning (critical for large tables)
            # Uses the first partition column discovered from table metadata
            if first_partition_col and date_from and date_to:
                # Generate list of partition dates to filter
                # Use original end date (not adjusted to_dt which has +1 day for exclusive filter)
                original_end_dt = datetime.strptime(date_to.split()[0], "%Y-%m-%d")
                partition_dates = self._generate_date_range(
                    from_dt - timedelta(hours=lookback_hours) if lookback_hours > 0 else from_dt,
                    original_end_dt
                )
                if partition_dates:
                    dt_list = ", ".join(f"'{d}'" for d in partition_dates)
                    conditions.append(f"{first_partition_col} IN ({dt_list})")
                    print(f"   Partition filter: {first_partition_col} IN ({partition_dates[0]}...{partition_dates[-1]}) [{len(partition_dates)} partitions]")

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        print(f"   SQL: {sql[:100]}..." if len(sql) > 100 else f"   SQL: {sql}")
        return self.spark.sql(sql)

    def _get_first_partition_column(self, table_path: str) -> Optional[str]:
        """
        Discover the first partition column from table metadata.
        
        Uses SHOW PARTITIONS to get partition info and extracts the first
        partition column name (e.g., 'dt' from 'dt=2025-12-07/hr=00/...').
        
        Returns None if table is not partitioned or query fails.
        """
        try:
            # Get first partition to extract column names
            partitions_df = self.spark.sql(f"SHOW PARTITIONS {table_path}")
            first_row = partitions_df.first()
            
            if first_row:
                # Partition format: "dt=2025-12-07/hr=00/mn=00/az=us-east-1b-eks-1"
                partition_str = first_row[0]
                # Extract first partition column name (before the '=')
                first_part = partition_str.split("/")[0]
                first_col = first_part.split("=")[0]
                print(f"   Discovered partition column: {first_col}")
                return first_col
        except Exception as e:
            # Table might not be partitioned or query failed
            print(f"   Could not discover partitions: {e}")
        
        return None

    def _generate_date_range(self, start_dt: datetime, end_dt: datetime) -> List[str]:
        """Generate list of date strings between start and end (inclusive)."""
        dates = []
        current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates

    @abstractmethod
    def _process(self, df: DataFrame) -> Any:
        """
        Process the loaded data. Must be implemented by subclasses.

        Args:
            df: Input DataFrame

        Returns:
            Result artifact (DataFrame, Model, Metrics dict, Plot, etc.)
        """
        pass

    @abstractmethod
    def _save_output(self, result: Any) -> None:
        """
        Save the result to the configured output location.
        Must be implemented by subclasses.

        Args:
            result: The result from _process()
        """
        pass
