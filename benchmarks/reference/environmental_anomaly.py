"""Reference analysis for environmental_anomaly_v1.

This implementation is intentionally deterministic and independent of an AI
agent. It establishes the expected scientific workflow against which an agent
can later be evaluated.
"""

from pathlib import Path
import json

from sciagent.tools.data import load_dataset, inspect_dataset
from sciagent.tools.numerical import detect_zscore_anomalies
from sciagent.tools.statistics import pearson_correlation, welch_t_test

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "environmental_sensor_data.csv"
OUTPUT = ROOT / "experiments" / "baseline" / "environmental_anomaly_reference.json"


def run() -> dict:
    df = load_dataset(DATASET)
    inspection = inspect_dataset(df)

    anomaly = detect_zscore_anomalies(df["pm25"], threshold=3.0)
    anomaly_indices = set(anomaly["anomaly_indices"])

    normal = df.loc[~df.index.isin(anomaly_indices), "pm25"]
    anomalous = df.loc[df.index.isin(anomaly_indices), "pm25"]

    significance = welch_t_test(anomalous, normal)
    correlation = pearson_correlation(df["temperature"], df["pm25"])

    result = {
        "benchmark_id": "environmental_anomaly_v1",
        "inspection": inspection,
        "anomaly_detection": anomaly,
        "significance_test": significance,
        "temperature_pm25_correlation": correlation,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
