"""Chatbot view: a conversational interface backed by Claude.

This module renders a chat interface in the style of consumer LLM apps:
a running conversation history, streamed responses, an easy way to copy
any answer, and a button to start a new conversation. It also exposes two
capabilities in the sidebar: web search and an extended-thinking level.
It holds the UI only; the model call goes through the LLM gateway.

The whole conversation is kept in session_state under "chat_messages" as
a list of {"role": "user"|"assistant", "content": str} dicts, which is
exactly the format the LLM gateway expects.
"""

import streamlit as st

import llm_client
import tools_config
from chatbot_prompts import build_chatbot_system_prompt
from session import select_chatbot_model


def _render_sidebar_controls() -> tuple[bool, str]:
    """Render the chatbot's sidebar controls and return the user choices.

    Adds, below the model selector: a web-search toggle, a reasoning-level
    selector, and a button to start a new conversation.

    Returns:
        A (web_search_enabled, thinking_level) tuple, where
        thinking_level is one of the keys of tools_config.THINKING_LEVELS.
    """
    with st.sidebar:
        web_search_enabled = st.toggle(
            "Web search",
            value=True,
            key="chat_web_search",
            help=(
                "Let the assistant search the web for current information. "
                "May increase cost and response time."
            ),
        )
        thinking_level = st.select_slider(
            "Reasoning",
            options=list(tools_config.THINKING_LEVELS),
            value="Off",
            key="chat_thinking",
            help=(
                "How much the assistant may think before answering. Higher "
                "levels improve hard questions but cost more and are slower."
            ),
        )
        if st.button("New conversation", width="stretch", key="chat_new"):
            st.session_state.chat_messages = []
            st.rerun()

    return web_search_enabled, thinking_level


def _render_assistant_message(content: str) -> None:
    """Render an assistant message with an easy way to copy its text.

    Shows the answer as Markdown for readability, plus a small popover
    holding the same text in a code block, which Streamlit renders with a
    built-in copy button.

    Args:
        content: The assistant message text.
    """
    st.markdown(content)
    with st.popover("📋 Copy"):
        # A code block exposes Streamlit's native one-click copy button.
        st.code(content, language=None)


def _render_history() -> None:
    """Render the full conversation stored in session_state.

    User messages are shown as Markdown; assistant messages additionally
    get a copy control.
    """
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_message(message["content"])
            else:
                st.markdown(message["content"])


def _handle_new_message(
    prompt: str,
    role: str,
    tools: list[dict],
    thinking_budget: int | None,
) -> None:
    """Append the user's message, stream the reply, and store both.

    Args:
        prompt: The text the user just submitted.
        role: The model role to answer with (from the sidebar selector).
        tools: Server tools to enable for this turn (may be empty).
        thinking_budget: Extended-thinking token budget, or None to
            disable thinking.
    """
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                llm_client.stream(
                    role=role,
                    system=build_chatbot_system_prompt(),
                    messages=st.session_state.chat_messages,
                    tools=tools,
                    thinking_budget=thinking_budget,
                )
            )
        except llm_client.LLMError as exc:
            st.error(str(exc))
            # Drop the user turn that produced no answer, so a retry does
            # not resend a dangling message.
            st.session_state.chat_messages.pop()
            return

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": response}
    )


def render() -> None:
    """Render the full Chatbot view.

    This is the entry point the app's navigation calls for this page.
    """
    role = select_chatbot_model()
    web_search_enabled, thinking_level = _render_sidebar_controls()

    st.header("Chatbot")

    _render_history()

    prompt = st.chat_input("Type your message")
    if prompt:
        tools = tools_config.build_chatbot_tools(
            web_search_enabled=web_search_enabled
        )
        budget = tools_config.thinking_budget(thinking_level)
        _handle_new_message(prompt, role, tools, budget)
