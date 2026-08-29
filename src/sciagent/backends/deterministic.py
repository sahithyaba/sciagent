from __future__ import annotations

from typing import Any

from sciagent.tools.data import load_dataset


class EnvironmentalAnomalyBackend:
    """Deterministic planner used to validate the agent loop before an LLM.

    This backend follows the reference workflow for the first benchmark case.
    It is intentionally transparent: every decision is encoded as Python so
    architecture failures can be separated from model failures.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.step = 0
        self.data = None
        self.anomaly_indices: list[int] = []

    def next_action(
        self,
        question: str,
        observations: list[dict[str, Any]],
        tools: list[dict[str, str]],
    ) -> dict[str, Any]:
        self.step += 1

        if self.step == 1:
            return {
                "type": "tool",
                "name": "load_dataset",
                "arguments": {"path": self.dataset_path},
            }

        if self.step == 2:
            df = observations[-1]["result"]
            self.data = df
            return {
                "type": "tool",
                "name": "detect_zscore_anomalies",
                "arguments": {"series": df["pm25"], "threshold": 3.0},
            }

        if self.step == 3:
            anomaly = observations[-1]["result"]
            self.anomaly_indices = anomaly["anomaly_indices"]
            normal = self.data.loc[~self.data.index.isin(self.anomaly_indices), "pm25"]
            anomalous = self.data.loc[self.data.index.isin(self.anomaly_indices), "pm25"]
            return {
                "type": "tool",
                "name": "welch_t_test",
                "arguments": {"a": anomalous, "b": normal},
            }

        if self.step == 4:
            return {
                "type": "tool",
                "name": "pearson_correlation",
                "arguments": {
                    "x": self.data["temperature"],
                    "y": self.data["pm25"],
                },
            }

        test = next(
            o["result"]
            for o in reversed(observations)
            if o.get("tool") == "welch_t_test"
        )
        return {
            "type": "final",
            "conclusion": (
                f"The PM2.5 series contains {len(self.anomaly_indices)} observations "
                f"with an absolute z-score above 3. The Welch t-test p-value was "
                f"{test['p_value']:.6g}; this reference workflow therefore treats "
                "the anomalous and normal groups as statistically different when "
                "the p-value is below 0.05."
            ),
        }
