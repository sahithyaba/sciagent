from __future__ import annotations

from pathlib import Path
import json

from sciagent.tools.data import load_dataset
from sciagent.tools.numerical import detect_zscore_anomalies
from sciagent.tools.statistics import pearson_correlation, summary_statistics, welch_t_test


def build_ground_truth(dataset_path: str | Path) -> dict:
    """Compute benchmark truth from the deterministic reference methods."""
    df = load_dataset(dataset_path)
    anomaly = detect_zscore_anomalies(df["pm25"], threshold=3.0)
    indices = set(anomaly["anomaly_indices"])
    normal = df.loc[~df.index.isin(indices), "pm25"]
    anomalous = df.loc[df.index.isin(indices), "pm25"]

    return {
        "dataset": str(dataset_path),
        "rows": len(df),
        "anomaly_detection": anomaly,
        "normal_pm25": summary_statistics(normal),
        "anomalous_pm25": summary_statistics(anomalous),
        "welch_t_test": welch_t_test(anomalous, normal),
        "temperature_pm25_correlation": pearson_correlation(df["temperature"], df["pm25"]),
    }


def write_ground_truth(dataset_path: str | Path, output_path: str | Path) -> dict:
    truth = build_ground_truth(dataset_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth
