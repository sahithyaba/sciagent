from pathlib import Path

from sciagent.agent import ScientificAgent
from sciagent.backends import EnvironmentalAnomalyBackend
from sciagent.tools.default_registry import build_default_registry


def test_environmental_backend_completes_workflow():
    root = Path(__file__).resolve().parents[1]
    dataset = root / "datasets" / "environmental_sensor_data.csv"
    agent = ScientificAgent(
        EnvironmentalAnomalyBackend(str(dataset)),
        build_default_registry(),
        max_steps=6,
    )

    result = agent.run(
        "Identify unusual PM2.5 behavior.",
        {"path": str(dataset), "target": "pm25"},
    )

    assert result.conclusion
    calls = result.trace["tool_calls"]
    assert [call["tool"] for call in calls] == [
        "load_dataset",
        "detect_zscore_anomalies",
        "welch_t_test",
        "pearson_correlation",
    ]
