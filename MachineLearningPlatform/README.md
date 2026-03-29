# ML Platform

> Configuration-driven Machine Learning Platform for RTB Feature Engineering and Model Training

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ML Platform is a configuration-driven framework that standardizes ML workflows across feature engineering, model training, and data visualization. Define your data sources, features, and tasks in YAML—the platform handles the rest.

### Key Features

- **Configuration-First Design**: Define data sources, features, and tasks entirely in YAML
- **Multiple Task Types**: ETL, Training, Enrichment, and Visualization pipelines
- **Jinja2 Templates**: Dynamic path resolution and configuration templating
- **Pydantic Validation**: Strict schema enforcement with clear error messages
- **Rolling Window Support**: Built-in time-series feature generation
- **Spark-Native**: Optimized for PySpark and Databricks environments

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        tasks.yaml                           │
│              (Workloads: What we DO)                        │
│   ┌──────────┬──────────┬─────────────┬───────────────┐    │
│   │   ETL    │ Training │ Enrichment  │ Visualization │    │
│   └────┬─────┴────┬─────┴──────┬──────┴───────┬───────┘    │
└────────┼──────────┼────────────┼──────────────┼────────────┘
         │          │            │              │
         ▼          ▼            ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Pipeline                          │
│         (Column Mappings → Transformations → Validation)     │
└─────────────────────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────────────┐    ┌─────────────────────────────┐
│     sources/*.yaml      │    │       features.yaml         │
│ (Physical: What we HAVE)│    │  (Contracts: What we NEED)  │
└─────────────────────────┘    └─────────────────────────────┘
```

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/xlu/ml-platform.git
cd ml-platform

# Install in development mode
pip install -e ".[dev]"

# Or with all optional dependencies
pip install -e ".[all]"
```

### From Wheel

```bash
pip install dist/ml_platform-0.1.0-py3-none-any.whl
```

### For Databricks

```bash
# Minimal install (Databricks provides Spark, Pandas, etc.)
pip install dist/ml_platform-0.1.0-py3-none-any.whl
```

## Quick Start

### List Available Tasks

```bash
ml-task --list-tasks
```

### Run a Task

```bash
# Run ETL task with date range
ml-task --task aggregate_hourly_win_price --from 2025-11-27 --to 2025-11-28

# Run with custom config directory
ml-task --task compute_rolling_win_price --config_dir /path/to/config --from 2025-11-27 --to 2025-11-28
```

### Development Mode

```bash
# Run via main.py during development
python main.py --task aggregate_hourly_win_price --from 2025-11-27 --to 2025-11-28
python main.py --list-tasks
```

## Configuration

### Directory Structure

```
ml_platform/conf/
├── tasks.yaml              # Task definitions (what to execute)
├── features.yaml           # Feature contracts (data requirements)
└── sources/
    ├── project_report.yaml  # Data source: project report
    ├── auction_logs.yaml    # Data source: Auction logs
    └── task_outputs.yaml   # Task outputs as sources
```

### Sources Configuration

Sources define physical data locations and schemas:

```yaml
# sources/project_logs.yaml
downsampled_auction_logs:
  type: "hive"
  path: "catalog.schema.downsampled_auction_logs"
  time_column: "auction_timestamp"
  columns:
    event_id:
      dtype: "string"
      description: "Unique auction event ID"
    auction_winner_price:
      dtype: "double"
      description: "Winning bid price in micros"
```

### Features Configuration

Features define data contracts for model inputs:

```yaml
# features.yaml
country_code:
  dtype: "string"
  description: "ISO 3166-1 alpha-2 country code"
  rules:
    format: "uppercase"
    max_null_percentage: 5.0
  fill_na: "XX"
```

### Tasks Configuration

Tasks define workloads to execute:

```yaml
# tasks.yaml
_settings:
  output_root: "s3://bucket/project"

aggregate_hourly_win_price:
  type: "etl"
  description: "Aggregate raw auction data to hourly win price metrics"

  input:
    source: "downsampled_auction_logs"
    date_range:
      from: ~  # Required: provide via --from
      to: ~    # Required: provide via --to
    filter_expr: "is_tpd_winner = true"

  params:
    dedupe_by: "event_id"
    group_by: ["rtb_id", "supply_name", "req_country"]
    time_bucket: "hour"
    time_column: "auction_timestamp"
    aggregations:
      - source_col: "auction_winner_price"
        agg_func: "sum"
        output_col: "hourly_win_price_sum"

  output:
    path: "{{ output_root }}/{{ type }}/{{ name }}"
    format: "parquet"
    mode: "overwrite"
    partition_by: ["date", "hour"]
```

## Task Types

### ETL Tasks

Transform and aggregate raw data:

- Deduplication by key
- Time bucketing (minute/hour/day)
- Aggregations (sum, count, avg, min, max)
- Rolling window calculations

### Training Tasks

Train ML models with configured features:

- Feature extraction via column mappings
- Hyperparameter configuration
- Validation split handling

### Enrichment Tasks

Join base data with feature sources:

- Multi-source joins
- Feature selection
- Null value handling

### Visualization Tasks

Generate data visualizations:

- Histogram comparisons
- Time-based decay analysis
- Plotly integration

## Development

### Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=ml_platform --cov-report=html

# Format code
black ml_platform/
ruff check ml_platform/ --fix
```

### Building Wheels

```bash
# Build wheel for distribution
./scripts/build_wheel.sh

# Or manually
pip install build
python -m build --wheel
```

### Project Structure

```
MachineLearningPlatform/
├── ml_platform/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── conf/               # Bundled configuration files
│   │   ├── tasks.yaml
│   │   ├── features.yaml
│   │   └── sources/
│   ├── config/
│   │   ├── loader.py       # YAML loading with Jinja2
│   │   ├── models.py       # Pydantic models
│   │   └── validators.py   # Custom validators
│   ├── core/
│   │   └── pipeline.py     # Feature pipeline
│   ├── features/
│   │   └── __init__.py
│   └── tasks/
│       ├── base.py         # BaseTask template method
│       ├── etl.py          # ETL task implementation
│       ├── training.py     # Training task implementation
│       ├── enrichment.py   # Enrichment task implementation
│       └── visualization.py # Visualization task implementation
├── tests/
├── scripts/
├── main.py                 # Development entry point
├── pyproject.toml
└── requirements.txt
```

## Dependencies

### Core (installed by default)
- `pyyaml` - YAML parsing
- `jinja2` - Template rendering
- `pycountry` - Country code utilities
- `pydantic>=2.0.0`

### Local Development (`pip install -e ".[local]"`)
- `pyspark>=3.4.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `boto3>=1.28.0`

### Visualization (`pip install -e ".[viz]"`)
- `plotly>=5.15.0`

### Development (`pip install -e ".[dev]"`)
- `pytest>=7.0.0`
- `pytest-cov>=4.0.0`
- `black>=23.0.0`
- `ruff>=0.1.0`

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Lu Xu