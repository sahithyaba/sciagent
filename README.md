# SciAgent

**Can AI Agents Do Scientific Computing?**

SciAgent is an experimental open-source project for building and evaluating **agentic scientific workflows with Python**.

The goal is not to prove that an AI agent can produce plausible Python. The goal is to measure whether an agent can:

- choose appropriate scientific tools,
- execute multi-step analyses,
- interpret numerical and statistical results,
- recognize uncertainty and failure,
- and produce reproducible, scientifically defensible workflows.

## Why SciAgent?

AI agents can increasingly call tools, analyze data, and decide what to do next. But a plausible final answer is not necessarily evidence of a correct scientific workflow.

SciAgent therefore evaluates the **process as well as the result**. Agent runs record the tools selected, arguments passed, observations returned, and the final conclusion. These traces can be compared with a reference scientific workflow.

## Current benchmark

The first benchmark, `environmental_anomaly_v1`, uses synthetic environmental sensor data containing:

- PM2.5
- temperature
- humidity
- pressure
- CO2

The agent is asked to investigate an unusual period in the dataset.

The reference workflow currently includes:

1. PM2.5 anomaly detection using an absolute z-score threshold of 3
2. Welch's independent two-sample t-test
3. Pearson correlation between temperature and PM2.5

## First LLM experiment

An initial run with **OpenAI-compatible access to `openai/gpt-oss-120b` through Hugging Face Inference Providers** successfully detected a contiguous anomalous segment of 25 PM2.5 observations (indices 600–624).

However, the agent did not reproduce the complete reference workflow. It selected:

```text
1. detect_zscore_anomalies
2. summary_statistics
```

instead of the reference sequence:

```text
1. detect_zscore_anomalies
2. welch_t_test
3. pearson_correlation
```

The preliminary workflow sequence score was **33.3% for this single run**. This is an early observation, not a final benchmark result; repeated runs and additional benchmark cases are planned.

This illustrates the central research question behind SciAgent:

> **Can an AI agent produce a plausible scientific conclusion without completing the scientific workflow required to support it?**

## Architecture

```text
Scientific Question
        ↓
     AI Agent
        ↓
  Tool Selection
        ↓
Scientific Python Tools
        ↓
     Observation
        ↓
   Next Agent Action
        ↓
     Evaluation
        ↓
Scientific Conclusion
```

The agent operates on an already-loaded benchmark dataset. The dataset-scoped tool registry exposes scientific operations without giving the model arbitrary filesystem access.

See [`docs/architecture.md`](docs/architecture.md).

## Scientific Python stack

- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib

## LLM support

SciAgent uses an OpenAI-compatible backend, allowing it to work with compatible model providers.

For the current experiment, Hugging Face Inference Providers is used with `openai/gpt-oss-120b`.

Create a local `.env` file:

```env
SCIAGENT_API_KEY=your_token_here
SCIAGENT_MODEL=openai/gpt-oss-120b:fastest
SCIAGENT_BASE_URL=https://router.huggingface.co/v1
```

**Never commit `.env` or an API token to GitHub.**

Install the project and LLM dependencies:

```bash
python -m pip install -e .
python -m pip install -e ".[llm]"
python -m pip install python-dotenv
```

Run the LLM benchmark:

```bash
python scripts/run_llm_agent.py
```

The run prints the complete artifact and also writes it to:

```text
experiments/llm/environmental_anomaly.json
```

If you do not want to use a `.env` file, `SCIAGENT_API_KEY` can also be provided as an environment variable.

## Run the deterministic baseline

```bash
python -m pip install -e .
python scripts/run_baseline.py
```

The deterministic baseline establishes the reference scientific analysis before introducing an AI agent.

## Evaluation

SciAgent is being developed to measure multiple dimensions of agentic scientific behavior, including:

- **Tool-selection accuracy** — did the agent select an appropriate scientific operation?
- **Workflow/sequence accuracy** — did it complete the expected analysis steps?
- **Numerical correctness** — are computed results correct?
- **Scientific validity** — does the conclusion follow from the analysis actually performed?
- **Reproducibility** — can the workflow and results be inspected and repeated?
- **Failure modes** — where does the agent stop early, select an inappropriate method, or make unsupported claims?

## Project status

🚧 **Early experimental prototype.**

The benchmark and evaluation framework are intentionally being built incrementally. Quantitative claims will be updated only after repeated experiments and additional benchmark cases are completed.

## Repository structure

```text
sciagent/
├── benchmarks/          # Scientific benchmark cases
├── datasets/            # Benchmark datasets
├── docs/                # Architecture and project documentation
├── experiments/         # Reproducible experiment artifacts
├── scripts/             # Baseline and LLM experiment runners
├── src/sciagent/
│   ├── backends/        # LLM backends
│   ├── evaluation/      # Benchmark evaluation
│   └── tools/            # Scientific tool implementations
└── tests/               # Automated tests
```

## Roadmap

1. ✅ Deterministic scientific baseline
2. ✅ Scientific tool registry
3. ✅ LLM agent adapter
4. ✅ Agent-vs-reference workflow evaluation
5. 🔄 Repeated baseline experiments
6. 🔄 Additional scientific benchmark cases
7. ⬜ Failure-mode experiments
8. ⬜ Scientific guardrails
9. ⬜ Reproducible experiment reports
10. ⬜ Cross-model evaluation

## Research direction

SciAgent is being developed around a simple premise:

**“The answer is correct” is not enough for scientific computing.**

An AI agent should also select appropriate methods, execute the necessary computational workflow, expose its intermediate results, and make conclusions that are supported by the evidence it actually computed.

The project explores how the scientific Python ecosystem can be used to make these workflows **observable, reproducible, and measurable**.

## License

MIT