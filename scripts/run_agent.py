from pathlib import Path
import json

from sciagent.agent import ScientificAgent
from sciagent.backends import EnvironmentalAnomalyBackend
from sciagent.tools.default_registry import build_default_registry

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "environmental_sensor_data.csv"
OUTPUT = ROOT / "experiments" / "agent"
OUTPUT.mkdir(parents=True, exist_ok=True)

question = (
    "Identify unusual behavior in the PM2.5 measurements and determine whether "
    "the anomalous period differs significantly from normal observations."
)

backend = EnvironmentalAnomalyBackend(str(DATASET))
agent = ScientificAgent(backend, build_default_registry(), max_steps=6)
result = agent.run(question, {"path": str(DATASET), "target": "pm25"})

payload = {
    "question": question,
    "conclusion": result.conclusion,
    "trace": result.trace,
}

path = OUTPUT / "environmental_anomaly_agent.json"
path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
print(json.dumps(payload, indent=2, default=str))
