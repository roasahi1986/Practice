"""
Visualization Task for the ML Platform.

This task handles data visualization workflows:
1. Load and aggregate data
2. Compute derived metrics (e.g., decay factors)
3. Generate interactive plots using Plotly

Supports configurable visualizations for data analysis and reporting.
Plots are displayed inline (in notebooks) rather than saved to files.
"""

from typing import Any, Dict, List
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml_platform.tasks.base import BaseTask


class VisualizationTask(BaseTask):
    """
    Task for generating data visualizations.

    Supports:
    - Data aggregation before visualization
    - Derived column computation (decay factors, etc.)
    - Multiple plot types (histogram, line, scatter, etc.)
    - Interactive Plotly plots displayed inline
    """

    def _process(self, df: DataFrame) -> Dict[str, Any]:
        """
        Execute the visualization pipeline.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing processed data and plot specifications
        """
        params = self.task_config.params

        # Stage 0: Add date column from time column if needed
        df = self._add_date_column(df)

        # Stage 1: Aggregate if specified
        if params.get("group_by") and params.get("aggregations"):
            df = self._aggregate(df, params["group_by"], params["aggregations"])

        # Stage 2: Compute derived columns
        if params.get("derived_columns"):
            df = self._compute_derived_columns(df, params["derived_columns"])

        # Stage 3: Apply decay if specified
        if params.get("decay"):
            df = self._apply_decay(df, params["decay"])

        # Stage 4: Generate and display visualizations
        self._generate_and_display_plots(df, params.get("plots", []))

        return {"dataframe": df, "status": "completed"}

    def _add_date_column(self, df: DataFrame) -> DataFrame:
        """Add date column from source time column if not present."""
        source_name = self.task_config.input.source
        source_config = self.project_config.sources.get(source_name)

        if not source_config or not source_config.time_column:
            return df

        time_col = source_config.time_column

        if "date" not in df.columns and time_col in df.columns:
            df = df.withColumn("date", F.date_format(time_col, "yyyy-MM-dd"))

        return df

    def _aggregate(
        self,
        df: DataFrame,
        group_by: List[str],
        aggregations: List[Dict],
    ) -> DataFrame:
        """Aggregate data by specified columns."""
        print(f"   Aggregating by: {group_by}")

        agg_exprs = []
        for agg in aggregations:
            source_col = agg["source_col"]
            agg_func = agg["agg_func"]
            output_col = agg["output_col"]

            if agg_func == "sum":
                expr = F.sum(F.coalesce(F.col(source_col), F.lit(0.0))).alias(output_col)
            elif agg_func == "count":
                expr = F.count("*").alias(output_col)
            elif agg_func == "avg":
                expr = F.avg(source_col).alias(output_col)
            else:
                raise ValueError(f"Unsupported aggregation: {agg_func}")

            agg_exprs.append(expr)

        return df.groupBy(group_by).agg(*agg_exprs)

    def _compute_derived_columns(
        self,
        df: DataFrame,
        derived_columns: List[Dict],
    ) -> DataFrame:
        """Compute derived columns based on expressions."""
        for col_spec in derived_columns:
            col_name = col_spec["name"]
            expr = col_spec["expr"]
            print(f"   Computing derived column: {col_name}")
            df = df.withColumn(col_name, F.expr(expr))

        return df

    def _apply_decay(self, df: DataFrame, decay_config: Dict) -> DataFrame:
        """
        Apply time-based decay to specified columns.

        Decay formula: value * (decay_factor ** days_to_end)
        Where days_to_end = end_date - row_date
        """
        decay_factor = decay_config["factor"]
        date_col = decay_config.get("date_column", "date")
        columns = decay_config["columns"]

        # Get end date from config or use date_range.to
        end_date = decay_config.get("end_date")
        if not end_date:
            date_range = self.task_config.input.date_range
            if date_range and date_range.date_to:
                end_date = date_range.date_to
            else:
                # Use max date in data
                max_date_row = df.agg(F.max(date_col)).collect()[0]
                end_date = max_date_row[0]

        print(f"   Applying decay factor {decay_factor} with end_date={end_date}")

        # Compute days to end
        df = df.withColumn(
            "_days_to_end",
            F.datediff(F.lit(end_date), F.col(date_col))
        )

        # Apply decay to each specified column
        for col_spec in columns:
            source_col = col_spec["source"]
            output_col = col_spec["output"]
            df = df.withColumn(
                output_col,
                F.col(source_col) * F.pow(F.lit(decay_factor), F.col("_days_to_end"))
            )

        # Drop helper column
        df = df.drop("_days_to_end")

        return df

    def _generate_and_display_plots(self, df: DataFrame, plot_specs: List[Dict]) -> None:
        """
        Generate and display plots using Plotly.

        Args:
            df: DataFrame with data to visualize
            plot_specs: List of plot specifications
        """
        if not plot_specs:
            print("   No plots specified")
            return

        # Import Plotly
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("   ⚠️ Plotly not available. Install with: pip install plotly")
            return

        # Convert to pandas for plotting
        print("   Converting to pandas for visualization...")
        pdf = df.toPandas()

        for spec in plot_specs:
            plot_type = spec["type"]
            title = spec.get("title", "")

            if plot_type == "histogram_comparison":
                fig = self._create_histogram_comparison_plotly(pdf, spec, title)
            elif plot_type == "line":
                fig = self._create_line_plot_plotly(pdf, spec, title)
            else:
                print(f"   ⚠️ Unknown plot type: {plot_type}")
                continue

            # Display the figure
            fig.show()
            print(f"   ✅ Displayed: {title}")

    def _create_histogram_comparison_plotly(self, pdf, spec: Dict, title: str):
        """Create overlaid histogram comparison using Plotly."""
        import plotly.graph_objects as go

        columns = spec["columns"]
        bins = spec.get("bins", 50)
        log_scale = spec.get("log_scale", False)

        fig = go.Figure()

        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, col in enumerate(columns):
            values = pdf[col].dropna()

            fig.add_trace(go.Histogram(
                x=values,
                name=col,
                opacity=0.6,
                nbinsx=bins,
                histnorm='probability density',  # Normalize to density
                marker_color=colors[i % len(colors)],
            ))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Density",
            barmode='overlay',  # Overlay histograms
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            height=500,
        )

        if log_scale:
            fig.update_xaxes(type="log")

        return fig

    def _create_line_plot_plotly(self, pdf, spec: Dict, title: str):
        """Create line plot using Plotly."""
        import plotly.graph_objects as go

        x_col = spec["x"]
        y_cols = spec["y"] if isinstance(spec["y"], list) else [spec["y"]]

        fig = go.Figure()

        for col in y_cols:
            fig.add_trace(go.Scatter(
                x=pdf[x_col],
                y=pdf[col],
                mode='lines',
                name=col,
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title="Value",
            template='plotly_white',
            height=500,
        )

        return fig

    def _save_output(self, result: Dict[str, Any]) -> None:
        """
        Visualization tasks display plots inline, no file output needed.

        Args:
            result: Dictionary containing status
        """
        # No file output for visualization tasks - plots are displayed inline
        print("   ✅ Visualization complete (plots displayed inline)")
