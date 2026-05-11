"""
Thin wrapper around OpenAI's API.

This module is the *only* place in our codebase that talks to OpenAI directly.
Everything else (navigation, synthesis, etc.) calls these functions instead of
making API calls itself. This means:
  - Retry logic lives in one place
  - Model selection happens in one place
  - Token counting is consistent across the whole app
  - Swapping providers later (e.g., to Anthropic) means changing one file
"""
import time
from typing import Optional

import tiktoken
from openai import OpenAI, OpenAIError

from app.config import settings

# Models we use throughout the app.
# NAVIGATION_MODEL is called many times per query (once per tree node), so it
# must be cheap. SYNTHESIS_MODEL is called once per query for the final answer,
# so quality matters more than cost.
NAVIGATION_MODEL = "gpt-5.4-mini"
SYNTHESIS_MODEL = "gpt-5.4"

# A single shared OpenAI client. Reused across all calls to avoid the overhead
# of creating new HTTP connections every time.
_client = OpenAI(api_key=settings.openai_api_key)


class LLMError(Exception):
    """Raised when an OpenAI call fails after all retries are exhausted."""
    pass


def call_llm(
    prompt: str,
    *,
    model: str,
    system: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    retries: int = 3,
) -> tuple[str, dict]:
    """
    Call OpenAI with a single user prompt.

    Returns:
        A tuple of (text, usage_dict) where usage_dict has keys:
            input_tokens, output_tokens, total_tokens.

    Raises:
        LLMError: if all retries are exhausted.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    uses_completion_tokens = (
        model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")
    )

    kwargs: dict = {
        "model": model,
        "messages": messages,
    }
    if uses_completion_tokens:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature

    last_error: Optional[Exception] = None

    for attempt in range(retries):
        try:
            response = _client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            usage = {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
            return text, usage
        except OpenAIError as e:
            last_error = e
            if attempt == retries - 1:
                break
            wait_seconds = 2 ** attempt
            time.sleep(wait_seconds)

    raise LLMError(f"OpenAI call failed after {retries} attempts: {last_error}")


def empty_usage() -> dict:
    """Helper for accumulating usage across multiple calls."""
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def add_usage(into: dict, more: dict) -> None:
    """Accumulate usage from a call into a running total. Mutates `into`."""
    into["input_tokens"] += more.get("input_tokens", 0)
    into["output_tokens"] += more.get("output_tokens", 0)
    into["total_tokens"] += more.get("total_tokens", 0)


# Tokenizer cache — building one is slow, so we keep them around per-model.
_encoders: dict[str, tiktoken.Encoding] = {}


def count_tokens(text: str, model: str = SYNTHESIS_MODEL) -> int:
    """
    Count how many tokens a piece of text will use under a given model.

    Important for: cost estimation, context-window budgeting, knowing when
    to stop adding more content to a prompt.
    """
    if model not in _encoders:
        try:
            _encoders[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to the GPT-4 encoding for unknown model names.
            _encoders[model] = tiktoken.get_encoding("cl100k_base")

    return len(_encoders[model].encode(text))