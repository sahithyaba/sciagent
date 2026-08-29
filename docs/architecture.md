# Architecture

SciAgent separates probabilistic reasoning from deterministic scientific computation.

```text
                     Scientific question
                              |
                              v
                        +-----------+
                        |   Agent   |
                        | plan      |
                        | select    |
                        | interpret |
                        +-----+-----+
                              |
                         tool calls
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
          pandas            SciPy           NumPy
             |                |                |
             +----------------+----------------+
                              |
                              v
                           Results
                              |
                              v
                        +-----------+
                        | Validation|
                        +-----+-----+
                              |
                     pass / fail / replan
```

## Design principle

The agent should orchestrate scientific software rather than replace it.

This lets us evaluate two separate questions:

1. Did the agent make a reasonable scientific decision?
2. Did the deterministic scientific implementation produce the correct result?

That separation is central to the project.
