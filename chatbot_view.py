"""Chatbot view: a conversational interface backed by Claude.

This module renders a chat interface in the style of consumer LLM apps:
a running conversation history, streamed responses, an easy way to copy
any answer, and a button to start a new conversation. It holds the UI
only; the model call goes through the LLM gateway.

The whole conversation is kept in session_state under "chat_messages" as
a list of {"role": "user"|"assistant", "content": str} dicts, which is
exactly the format the LLM gateway expects.
"""

import streamlit as st

import llm_client
from chatbot_prompts import build_chatbot_system_prompt
from session import select_chatbot_model


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


def _handle_new_message(prompt: str, role: str) -> None:
    """Append the user's message, stream the reply, and store both.

    Args:
        prompt: The text the user just submitted.
        role: The model role to answer with (from the sidebar selector).
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

    with st.sidebar:
        if st.button("New conversation", width="stretch", key="chat_new"):
            st.session_state.chat_messages = []
            st.rerun()

    st.header("Chatbot")

    _render_history()

    prompt = st.chat_input("Type your message")
    if prompt:
        _handle_new_message(prompt, role)
