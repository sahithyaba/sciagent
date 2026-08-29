# Scientific Benchmarks

SciAgent evaluates agentic workflows against independently defined reference analyses.

## Benchmark principles

A benchmark case should specify:

- the scientific question,
- the dataset,
- an independently defined reference workflow,
- expected outputs,
- evaluation criteria, and
- known failure modes.

The benchmark must not treat a successful-looking agent response as ground truth.

## Current case

### `environmental_anomaly_v1`

The first controlled experiment asks an agent to investigate unusual PM2.5 behavior in an environmental sensor dataset.

The reference workflow uses:

1. absolute PM2.5 z-score > 3 for anomaly detection,
2. Welch's independent two-sample t-test for anomalous vs normal PM2.5,
3. Pearson correlation for temperature vs PM2.5.

These choices form the reference, not a claim that they are universally optimal scientific methods. Future benchmark cases should test whether an agent can recognize when a reference method is inappropriate and justify an alternative.
