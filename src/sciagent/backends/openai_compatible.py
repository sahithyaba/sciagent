from __future__ import annotations

import json
import os
from typing import Any

from .json_parser import parse_json_action
from .prompt import build_prompt


class OpenAICompatibleBackend:
    """LLM backend for OpenAI-compatible chat-completions APIs.

    The provider is configured through environment variables so credentials and
    provider-specific settings never become part of benchmark artifacts.
    """

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("SCIAGENT_API_KEY"),
                base_url=os.environ.get("SCIAGENT_BASE_URL") or None,
            )
        self.client = client
        self.model = model or os.environ.get("SCIAGENT_MODEL", "gpt-4.1-mini")

    def next_action(
        self,
        question: str,
        observations: list[dict[str, Any]],
        tools: list[dict[str, str]],
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": build_prompt(question, observations, tools)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned an empty response")
        return parse_json_action(content)
