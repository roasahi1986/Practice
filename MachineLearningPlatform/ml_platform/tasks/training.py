"""
Training Task for the ML Platform.

This task handles learner training workflows:
1. Load and prepare training data
2. Train learner with configured hyperparameters
3. Save learner artifact to specified location
"""

from typing import Any
from pyspark.sql import DataFrame

from ml_platform.tasks.base import BaseTask


class TrainingTask(BaseTask):
    """
    Task for training machine learning models.

    Supports various learner types (XGBoost, LightGBM, etc.) through
    configuration-driven hyperparameters.
    """

    def _process(self, df: DataFrame) -> Any:
        """
        Train a learner on the input data.

        Args:
            df: Training DataFrame with features

        Returns:
            Trained learner artifact (dict containing learner metadata)
        """
        params = self.task_config.params
        print(f"   Learner: {params.get('learner', 'unknown')}")
        print(f"   Target column: {params.get('target_col', 'N/A')}")
        print(f"   Hyperparameters: {params.get('hyperparameters', {})}")

        # TODO: Implement actual model training
        # Example for XGBoost:
        # features = df.select(feature_columns)
        # target = df.select(params["target_col"])
        # model = XGBClassifier(**params["hyperparameters"])
        # model.fit(features, target)

        # Mock implementation
        learner_artifact = {
            "name": params.get("learner", "unknown"),
            "params": params,
            "status": "trained",
        }
        return learner_artifact

    def _save_output(self, result: Any) -> None:
        """
        Save the trained learner to the configured output path.

        Args:
            result: Learner artifact from _process()
        """
        output_path = self.task_config.output.path
        output_format = self.task_config.output.format

        print(f"   Output path: {output_path}")
        print(f"   Output format: {output_format}")

        # TODO: Implement actual learner saving
        # if output_format == "mlflow":
        #     mlflow.sklearn.log_model(result, output_path)
        # elif output_format == "pickle":
        #     with open(output_path, "wb") as f:
        #         pickle.dump(result, f)

        print(f"✅ Learner artifact saved (mock)")
