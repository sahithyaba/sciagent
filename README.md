# SciAgent

**Can AI Agents Do Scientific Computing?**

SciAgent is an experimental open-source project for building and evaluating agentic scientific workflows with Python.

The goal is not to prove that an AI agent can produce plausible Python. The goal is to measure whether an agent can:

- choose appropriate scientific tools,
- execute multi-step analyses,
- interpret numerical/statistical results,
- recognize uncertainty and failure,
- and produce reproducible workflows.

## Current experiment

The first experiment uses synthetic environmental sensor data containing temperature, humidity, pressure, CO2, and particulate matter measurements.

The agent is asked to investigate an unusual period in the dataset.

## Scientific Python stack

- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib

## Project status

🚧 Early experimental prototype.

The benchmark and agent evaluation are intentionally being built incrementally. Results will be added only after experiments are actually run.

## Planned architecture

Question → Agent → Scientific tools → Results → Validation → Agent interpretation

See [`docs/architecture.md`](docs/architecture.md).

## Run the deterministic baseline

```bash
python -m pip install -e .
python scripts/run_baseline.py
```

This establishes a reference analysis before introducing an AI agent.

## Next milestones

1. Baseline scientific analysis
2. Tool registry
3. Agent adapter
4. Agent-vs-reference benchmark
5. Failure-mode experiments
6. Scientific guardrails
7. Reproducible experiment reports
