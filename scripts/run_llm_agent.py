"""Run the environmental anomaly benchmark with an LLM backend.

Required environment variables:
  SCIAGENT_API_KEY
Optional:
  SCIAGENT_MODEL (default: gpt-4.1-mini)
  SCIAGENT_BASE_URL (for OpenAI-compatible providers)

A local .env file is loaded when python-dotenv is installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from sciagent.agent import ScientificAgent
from sciagent.backends.openai_compatible import OpenAICompatibleBackend
from sciagent.evaluation.benchmark import evaluate_case, load_case
from sciagent.tools.data import inspect_dataset, load_dataset
from sciagent.tools.dataset_registry import build_dataset_registry

ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "benchmarks" / "cases" / "environmental_anomaly.json"
DATASET_PATH = ROOT / "datasets" / "environmental_sensor_data.csv"
OUTPUT_PATH = ROOT / "experiments" / "llm" / "environmental_anomaly.json"


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    if not os.environ.get("SCIAGENT_API_KEY"):
        raise SystemExit("SCIAGENT_API_KEY is required")

    case = load_case(CASE_PATH)
    df = load_dataset(DATASET_PATH)
    dataset_description = inspect_dataset(df)

    # The agent operates on the already-loaded benchmark dataset. Tools expose
    # column names rather than filesystem paths or arbitrary Python objects.
    agent = ScientificAgent(
        OpenAICompatibleBackend(),
        build_dataset_registry(df),
        max_steps=10,
    )
    result = agent.run(case["question"], dataset_description)

    observed_tools = [call["tool"] for call in result.trace["tool_calls"]]
    evaluation = evaluate_case(
        case,
        observed_tools=observed_tools,
        conclusion=result.conclusion,
    )

    artifact = {
        "benchmark_id": case["id"],
        "model": agent.backend.model,
        "conclusion": result.conclusion,
        "trace": result.trace,
        "evaluation": evaluation,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(json.dumps(artifact, indent=2, default=str))


if __name__ == "__main__":
    main()
