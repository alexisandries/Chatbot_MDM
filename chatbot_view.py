"""Chatbot view: a conversational interface backed by Claude.

This module renders a chat interface in the style of consumer LLM apps:
a running conversation history, streamed responses, an easy way to copy
any answer, file attachments, and a button to start a new conversation.
It also exposes two capabilities in the sidebar: web search and an
extended-thinking level. It holds the UI only; the model call goes
through the LLM gateway.

ATTACHMENTS (current-message scope)
===================================
Files attached in the chat input apply to the message they are sent with
only. They are sent to the model with that single turn and are NOT
re-sent on later turns: afterwards the conversation keeps just a short
"📎 filename" trace. This keeps follow-up turns cheap, since a large PDF
or image is never re-transmitted. The model's own answer remains in the
history, so the thread stays coherent without resending the file.

The conversation is stored in session_state under "chat_messages" as a
list of {"role", "content": str} dicts. Stored content is always the
lightweight text (never the heavy file data); the file blocks exist only
transiently, while the message that carries them is being sent.
"""

import streamlit as st

import attachments
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


def _build_display_text(text: str, labels: list[str]) -> str:
    """Build the lightweight text stored and shown for a user turn.

    Combines the typed text with a short note listing any attached files.

    Args:
        text: The text the user typed (may be empty).
        labels: File names attached to this message (may be empty).

    Returns:
        The text to display and persist for this turn.
    """
    if not labels:
        return text
    note = "📎 " + ", ".join(labels)
    return f"{text}\n\n{note}" if text else note


def _handle_new_message(
    text: str,
    files: list,
    role: str,
    tools: list[dict],
    thinking_budget: int | None,
) -> None:
    """Send a user message (with optional attachments) and store the turn.

    Attachments are sent with this turn only; the stored history keeps a
    lightweight trace instead of the file data.

    Args:
        text: The text the user typed (may be empty if only files sent).
        files: Files attached to this message (may be empty).
        role: The model role to answer with.
        tools: Server tools to enable for this turn (may be empty).
        thinking_budget: Extended-thinking token budget, or None.
    """
    # Convert attachments to content blocks for this turn only.
    attachment_blocks: list[dict] = []
    labels: list[str] = []
    if files:
        try:
            attachment_blocks, labels = attachments.build_attachment_blocks(files)
        except attachments.AttachmentError as exc:
            st.error(str(exc))
            return

    # Content actually sent to the model: attachments first, then text.
    api_content: list[dict] = list(attachment_blocks)
    if text:
        api_content.append({"type": "text", "text": text})
    if not api_content:
        return  # nothing to send

    display_text = _build_display_text(text, labels)
    with st.chat_message("user"):
        st.markdown(display_text)

    # Prior (lightweight) history plus this turn's full content.
    api_messages = list(st.session_state.chat_messages)
    api_messages.append({"role": "user", "content": api_content})

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                llm_client.stream(
                    role=role,
                    system=build_chatbot_system_prompt(),
                    messages=api_messages,
                    tools=tools,
                    thinking_budget=thinking_budget,
                )
            )
        except llm_client.LLMError as exc:
            st.error(str(exc))
            return

    # Persist only the lightweight versions (no heavy file data re-sent).
    st.session_state.chat_messages.append(
        {"role": "user", "content": display_text}
    )
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
    st.caption(
        "📎 Attachments: images and PDFs are read in full, including their "
        "visuals. Office files (Word, PowerPoint, Excel) can be attached too, "
        "but only their text is read — embedded images are not. If a document's "
        "visuals matter, convert it to PDF first."
    )

    _render_history()

    submission = st.chat_input(
        "Type your message (and attach files if needed)",
        accept_file="multiple",
        file_type=attachments.ALLOWED_EXTENSIONS,
        key="chat_input",
    )
    if submission and (submission.text or submission.files):
        tools = tools_config.build_chatbot_tools(
            web_search_enabled=web_search_enabled
        )
        budget = tools_config.thinking_budget(thinking_level)
        _handle_new_message(
            submission.text or "", submission.files, role, tools, budget
        )
