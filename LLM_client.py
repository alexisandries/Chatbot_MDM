"""Single gateway for every LLM call made by the application.

All other modules (translation UI, chatbot UI, glossary checker, ...)
must go through the functions defined here - never through the
`anthropic` SDK directly. This guarantees three things:

1. Models are always resolved through their ROLE (see models_config.py),
   so model upgrades never require touching feature code.
2. Error handling and API-key management live in exactly one place.
3. Adding a second provider later (e.g. Mistral) only means extending
   the dispatch inside this module; callers will not notice.

Public API:
    complete(...)       -> str            One-shot completion.
    complete_json(...)  -> dict | list    Completion parsed as JSON.
    stream(...)         -> Iterator[str]  Streaming completion (chatbot).
    LLMError                              Exception raised on any failure.

Secrets:
    The Anthropic API key is read from st.secrets["ANTHROPIC_API_KEY"].
    See secrets.toml.example for the expected structure.
"""

import json
from collections.abc import Iterator

import anthropic
import streamlit as st

from models_config import ModelSpec, get_model_for_role


class LLMError(Exception):
    """Raised when an LLM call fails for any reason.

    The message is safe to display to end users: it never contains the
    API key or raw provider internals, only a short human-readable
    explanation of what went wrong.
    """


# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_anthropic_client() -> anthropic.Anthropic:
    """Create (once per process) the Anthropic SDK client.

    Returns:
        A configured anthropic.Anthropic client, cached by Streamlit so
        every rerun of the script reuses the same instance.

    Raises:
        LLMError: If the API key is missing from st.secrets.
    """
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except KeyError as exc:
        raise LLMError(
            "ANTHROPIC_API_KEY is missing from secrets. "
            "Add it to .streamlit/secrets.toml (see secrets.toml.example)."
        ) from exc
    return anthropic.Anthropic(api_key=api_key)


def _resolve(role: str) -> ModelSpec:
    """Map a role to its ModelSpec and verify the provider is supported.

    Args:
        role: A model role defined in models_config.ROLE_TO_MODEL.

    Returns:
        The ModelSpec assigned to that role.

    Raises:
        LLMError: If the role is unknown or the model's provider is not
            implemented in this module yet.
    """
    try:
        spec = get_model_for_role(role)
    except KeyError as exc:
        raise LLMError(str(exc)) from exc
    if spec.provider != "anthropic":
        # Placeholder for future providers (e.g. Mistral): add a branch
        # in complete() / stream() and extend this check.
        raise LLMError(
            f"Provider '{spec.provider}' is not implemented. "
            f"Only 'anthropic' is currently supported."
        )
    return spec


def _normalize_messages(
    prompt: str | None,
    messages: list[dict] | None,
) -> list[dict]:
    """Turn the caller's input into the messages list the API expects.

    Callers can pass EITHER a single `prompt` string (convenient for
    one-shot tasks like translation) OR a full `messages` list
    (necessary for multi-turn chat). Exactly one of the two must be
    provided.

    Args:
        prompt: A single user message, or None.
        messages: A list of {"role": "user"|"assistant", "content": str}
            dicts in conversation order, or None.

    Returns:
        A messages list suitable for the Anthropic Messages API.

    Raises:
        LLMError: If both or neither of prompt/messages are given.
    """
    if (prompt is None) == (messages is None):
        raise LLMError(
            "Pass exactly one of 'prompt' or 'messages' to the LLM layer."
        )
    if prompt is not None:
        return [{"role": "user", "content": prompt}]
    return messages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def complete(
    role: str,
    system: str,
    prompt: str | None = None,
    messages: list[dict] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Run a one-shot (non-streaming) LLM completion.

    This is the workhorse for translation, refinement and any other
    task where the full answer is needed before display.

    Args:
        role: Model role ("economy", "standard", "premium", "utility").
            The concrete model is resolved from models_config.py.
        system: System prompt defining the model's task and constraints.
        prompt: Single user message. Mutually exclusive with `messages`.
        messages: Full conversation history as a list of
            {"role": ..., "content": ...} dicts. Mutually exclusive
            with `prompt`.
        temperature: Sampling temperature. Defaults to the model's
            default_temperature from the registry.
        max_tokens: Output token cap. Defaults to the model's
            default_max_tokens from the registry.

    Returns:
        The model's text response, stripped of leading/trailing
        whitespace. Multiple text blocks (rare) are joined with
        newlines.

    Raises:
        LLMError: On configuration problems or any API failure
            (authentication, rate limit, overload, network, ...). The
            message is suitable for display in the UI via st.error().
    """
    spec = _resolve(role)
    client = _get_anthropic_client()
    try:
        response = client.messages.create(
            model=spec.api_id,
            system=system,
            messages=_normalize_messages(prompt, messages),
            temperature=(
                temperature if temperature is not None
                else spec.default_temperature
            ),
            max_tokens=(
                max_tokens if max_tokens is not None
                else spec.default_max_tokens
            ),
        )
    except anthropic.AuthenticationError as exc:
        raise LLMError(
            "Authentication with the Anthropic API failed. "
            "Check ANTHROPIC_API_KEY in secrets."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise LLMError(
            "The Anthropic API rate limit was reached. "
            "Wait a moment and try again."
        ) from exc
    except anthropic.APIError as exc:
        raise LLMError(f"The Anthropic API returned an error: {exc}") from exc
    except Exception as exc:  # network issues, timeouts, ...
        raise LLMError(f"Unexpected error while calling the LLM: {exc}") from exc

    text_parts = [
        block.text for block in response.content if block.type == "text"
    ]
    if not text_parts:
        raise LLMError("The model returned an empty response.")
    return "\n".join(text_parts).strip()


def complete_json(
    role: str,
    system: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
):
    """Run a completion and parse the answer as JSON.

    Intended for internal machinery that needs structured output, such
    as the glossary term detector. The system/prompt you pass must
    already instruct the model to answer with JSON only; this function
    adds robustness by stripping Markdown code fences before parsing.

    Args:
        role: Model role, typically "utility".
        system: System prompt (must demand JSON-only output).
        prompt: User message containing the task and the data.
        temperature: Sampling temperature. Defaults to 0.0 because
            structured extraction should be deterministic.
        max_tokens: Output token cap. Defaults to the model's registry
            value.

    Returns:
        The parsed JSON value (usually a dict or a list).

    Raises:
        LLMError: If the call fails, or if the model's answer cannot be
            parsed as JSON even after removing code fences.
    """
    raw = complete(
        role=role,
        system=system,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove an opening fence like ``` or ```json and a closing ```.
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"The model did not return valid JSON: {exc}. "
            f"Raw output started with: {raw[:200]!r}"
        ) from exc


def stream(
    role: str,
    system: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """Run a streaming completion, yielding text chunks as they arrive.

    Intended for the chatbot interface, where chunks are fed directly
    to st.write_stream() so the user sees the answer being typed.

    Args:
        role: Model role (usually whatever the user picked in the
            sidebar: "economy", "standard" or "premium").
        system: System prompt for the conversation.
        messages: Full conversation history as a list of
            {"role": "user"|"assistant", "content": str} dicts.
        temperature: Sampling temperature. Defaults to the model's
            registry value.
        max_tokens: Output token cap. Defaults to the model's registry
            value.

    Yields:
        Text fragments in order. Concatenating all fragments gives the
        complete answer.

    Raises:
        LLMError: On configuration problems or any API failure. Note
            that because this is a generator, the exception is raised
            when the stream is consumed, not when stream() is called.
    """
    spec = _resolve(role)
    client = _get_anthropic_client()
    try:
        with client.messages.stream(
            model=spec.api_id,
            system=system,
            messages=_normalize_messages(None, messages),
            temperature=(
                temperature if temperature is not None
                else spec.default_temperature
            ),
            max_tokens=(
                max_tokens if max_tokens is not None
                else spec.default_max_tokens
            ),
        ) as event_stream:
            yield from event_stream.text_stream
    except anthropic.AuthenticationError as exc:
        raise LLMError(
            "Authentication with the Anthropic API failed. "
            "Check ANTHROPIC_API_KEY in secrets."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise LLMError(
            "The Anthropic API rate limit was reached. "
            "Wait a moment and try again."
        ) from exc
    except anthropic.APIError as exc:
        raise LLMError(f"The Anthropic API returned an error: {exc}") from exc
    except Exception as exc:
        raise LLMError(f"Unexpected error while streaming from the LLM: {exc}") from exc
