class ScientificAgent:
    """Placeholder agent interface.

    The first milestone intentionally keeps the model adapter separate from
    the scientific tools. A later milestone will implement an open-model
    adapter and structured tool calling.
    """

    def __init__(self, tool_registry=None):
        self.tool_registry = tool_registry or {}

    def plan(self, question: str, dataset_description: dict) -> list[str]:
        raise NotImplementedError(
            "Agent backend is intentionally not implemented in v0.1. "
            "Build and verify the deterministic baseline first."
        )
