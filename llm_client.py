"""Single gateway for every LLM call made by the application.

PURPOSE AND ROLE IN THE ARCHITECTURE
=====================================
This module is the ONLY place in the codebase that touches the Anthropic
SDK. All other modules — translation UI, chatbot UI, glossary checker,
future refinement pipelines — import from here and never instantiate an
`anthropic.Anthropic` client themselves.

That constraint buys three concrete benefits:

  1. Model resolution by role, never by hard-coded string.
     Callers ask for role="standard" or role="utility". The actual API
     model string (e.g. "claude-sonnet-4-6") is resolved through
     models_config.py. When Anthropic releases a new model and we want
     to upgrade a tier, we change one line in models_config.py and
     nothing else in the application changes.

  2. Centralised error handling.
     Every exception that the Anthropic SDK can raise — authentication
     failures, rate limits, transient network errors, empty responses —
     is caught here and re-raised as LLMError with a message that is
     safe and meaningful to show directly in the UI via st.error(). No
     try/except blocks are scattered through feature code.

  3. Provider abstraction.
     The `provider` field in ModelSpec and the guard in _resolve() are
     the hooks for adding Mistral (or any other provider) later. The
     change will be: add the new SDK import, add a branch in complete()
     and stream(), extend the guard. Callers will not notice.


PUBLIC API SUMMARY
==================
Three functions and one exception class form the public contract:

  complete(role, system, prompt=..., messages=..., ...)  -> str
      Non-streaming, one-shot completion. Use for translation and any
      task where the full answer must be assembled before display.
      Accepts either a single `prompt` string or a `messages` list, but
      not both (enforced by _normalize_messages).

  complete_json(role, system, prompt, ...)  -> dict | list
      Thin wrapper around complete() that strips Markdown code fences
      and parses the response as JSON. Designed for internal machinery
      (glossary term detection) that needs structured output. Callers
      must ask for JSON-only output in their system prompt; this
      function adds resilience on top of that.

  stream(role, system, messages, ...)  -> Iterator[str]
      Streaming completion; yields text fragments in order. Intended for
      the chatbot interface: pass the return value directly to
      st.write_stream() so the user sees the answer build up in real
      time. Because this is a generator, exceptions surface when the
      iterator is consumed, not when stream() is called.

  LLMError(Exception)
      Raised by all three functions on any failure. The message string
      is always human-readable and safe to pass to st.error(). It never
      leaks the API key or raw SDK internals.


PRIVATE HELPERS (not for import)
=================================
  _get_anthropic_client()   Reads the API key from st.secrets and
                            returns a cached SDK client (@cache_resource
                            ensures a single instance per process across
                            all Streamlit reruns).

  _resolve(role)            Maps a role string to a ModelSpec via
                            models_config.get_model_for_role(), then
                            verifies the provider is supported. Raises
                            LLMError (not KeyError) so callers always
                            deal with a single exception type.

  _normalize_messages()     Enforces the prompt-XOR-messages contract
                            and converts a bare prompt string into the
                            list format the Anthropic Messages API
                            expects.


HOW TO CALL THIS MODULE — QUICK EXAMPLES
=========================================
One-shot translation (most common case):

    from llm_client import complete, LLMError

    try:
        result = complete(
            role="standard",
            system="You are a professional translator...",
            prompt="Translate the following text: ...",
        )
    except LLMError as exc:
        st.error(str(exc))

Structured glossary check:

    from llm_client import complete_json, LLMError

    try:
        data = complete_json(
            role="utility",
            system="Return only a JSON object with a 'matches' key...",
            prompt=f"Source text: {text}\nGlossary: {glossary}",
        )
        matches = data.get("matches", [])
    except LLMError as exc:
        st.error(str(exc))

Streaming chatbot response:

    from llm_client import stream, LLMError

    try:
        with st.chat_message("assistant"):
            response = st.write_stream(
                stream(role="standard", system=SYSTEM, messages=history)
            )
    except LLMError as exc:
        st.error(str(exc))


SECRETS
=======
The Anthropic API key is read exclusively from:
    st.secrets["ANTHROPIC_API_KEY"]

Add it to .streamlit/secrets.toml (see secrets.toml.example). The key
is never logged, never included in error messages, and never passed
through function arguments.
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
    params = {
        "model": spec.api_id,
        "system": system,
        "messages": _normalize_messages(prompt, messages),
        "max_tokens": (
            max_tokens if max_tokens is not None else spec.default_max_tokens
        ),
    }
    # Only send temperature to models that accept it; some models reject
    # the parameter outright.
    if spec.supports_temperature:
        params["temperature"] = (
            temperature if temperature is not None else spec.default_temperature
        )
    try:
        response = client.messages.create(**params)
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
    params = {
        "model": spec.api_id,
        "system": system,
        "messages": _normalize_messages(None, messages),
        "max_tokens": (
            max_tokens if max_tokens is not None else spec.default_max_tokens
        ),
    }
    if spec.supports_temperature:
        params["temperature"] = (
            temperature if temperature is not None else spec.default_temperature
        )
    try:
        with client.messages.stream(**params) as event_stream:
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
