from pathlib import Path
import json
import pandas as pd

from sciagent.tools.data import load_dataset, inspect_dataset
from sciagent.tools.numerical import detect_zscore_anomalies
from sciagent.tools.statistics import welch_t_test, pearson_correlation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "environmental_sensor_data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = load_dataset(DATA)
inspection = inspect_dataset(df)

# Reference anomaly rule for the first benchmark:
# flag rows where PM2.5 has an absolute z-score > 3.
anomaly = detect_zscore_anomalies(df["pm25"], threshold=3.0)
mask = df.index.isin(anomaly["anomaly_indices"])

normal = df.loc[~mask, "pm25"]
anomalous = df.loc[mask, "pm25"]

if len(anomalous) and len(normal):
    test = welch_t_test(anomalous, normal)
else:
    test = {"error": "Not enough observations for comparison."}

correlation = pearson_correlation(df["temperature"], df["pm25"])

result = {
    "dataset": str(DATA.relative_to(ROOT)),
    "inspection": inspection,
    "anomaly_detection": anomaly,
    "anomalous_vs_normal_pm25": test,
    "temperature_pm25_correlation": correlation,
}

(OUT / "baseline_result.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)

print(json.dumps(result, indent=2))
