"""Chatbot view: a conversational interface backed by Claude.

This module renders a chat interface in the style of consumer LLM apps:
a running conversation history, streamed responses, an easy way to copy
any answer, file attachments, and a button to start a new conversation.
It also exposes capabilities in the sidebar: web search, an extended-
thinking level, and conversation-level documents. It holds the UI only;
the model call goes through the LLM gateway.

TWO WAYS TO ATTACH FILES
========================
Current-message attachments (chat input)
    Files attached in the chat input apply to that one message only. They
    are sent with that single turn and are NOT re-sent afterwards; the
    history keeps just a short "📎 filename" trace. Cheap for follow-ups.

Conversation documents (sidebar uploader)
    Files uploaded in the sidebar stay available for the whole
    conversation. They are re-sent with every turn so the model can keep
    referring to them, but they are prompt-cached: after the first turn,
    re-reading them costs a fraction of the normal token price. Best for a
    reference document the user asks several questions about.

STORED STATE
============
The conversation is stored in session_state under "chat_messages" as a
list of {"role", "content": str} dicts. Stored content is always the
lightweight text (never heavy file data): current-message attachments are
reduced to a "📎" trace, and conversation documents live only in the
sidebar uploader, rebuilt into the request each turn.
"""

import streamlit as st

import attachments
import llm_client
import tools_config
from chatbot_prompts import build_chatbot_system_prompt
from session import select_chatbot_model


# Synthetic assistant turn placed right after the conversation documents,
# so the message roles keep alternating. It sits after the cache
# breakpoint, so it does not affect what gets cached.
_DOCUMENTS_ACK = "Understood. I'll use these documents as needed."


def _render_sidebar_controls() -> tuple[bool, str, list]:
    """Render the chatbot's sidebar controls and return the user choices.

    Adds, below the model selector: a web-search toggle, a reasoning-level
    selector, a conversation-documents uploader, and a button to start a
    new conversation (which also clears the documents).

    Returns:
        A (web_search_enabled, thinking_level, conversation_files) tuple.
        conversation_files is the list of files uploaded for the whole
        conversation (possibly empty).
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

        st.markdown("**Conversation documents**")
        st.caption(
            "Available to the whole conversation and re-used each turn "
            "(cached to limit cost). For images inside Office files, "
            "convert to PDF first."
        )
        conversation_files = st.file_uploader(
            "Add documents",
            type=attachments.ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            key=f"chat_docs_{st.session_state.chat_docs_nonce}",
            label_visibility="collapsed",
        )

        if st.button("New conversation", width="stretch", key="chat_new"):
            st.session_state.chat_messages = []
            # Bump the nonce so the document uploader resets to empty.
            st.session_state.chat_docs_nonce += 1
            st.rerun()

    return web_search_enabled, thinking_level, conversation_files or []


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

    Combines the typed text with a short note listing any files attached
    to this specific message.

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


def _build_document_prefix(conversation_files: list) -> list[dict]:
    """Build the cached conversation-documents prefix messages.

    Converts the conversation documents into content blocks (with a cache
    breakpoint) and wraps them as a user message followed by a short
    synthetic assistant acknowledgement, so the message roles alternate.

    Args:
        conversation_files: Files uploaded for the whole conversation.

    Returns:
        A list of zero or two messages to prepend to the request. Empty
        when there are no conversation documents.

    Raises:
        attachments.AttachmentError: If a document is too large or of an
            unsupported type.
    """
    if not conversation_files:
        return []
    document_blocks, _labels = attachments.build_attachment_blocks(
        conversation_files, cache_last_block=True
    )
    return [
        {"role": "user", "content": document_blocks},
        {"role": "assistant", "content": _DOCUMENTS_ACK},
    ]


def _handle_new_message(
    text: str,
    files: list,
    conversation_files: list,
    role: str,
    tools: list[dict],
    thinking_budget: int | None,
) -> None:
    """Send a user message and store the turn.

    Builds the request as: cached conversation-documents prefix, then the
    prior conversation history, then the current user turn (which may
    carry its own current-message attachments). Only lightweight text is
    stored in the history.

    Args:
        text: The text the user typed (may be empty if only files sent).
        files: Files attached to this specific message (may be empty).
        conversation_files: Documents shared across the whole conversation.
        role: The model role to answer with.
        tools: Server tools to enable for this turn (may be empty).
        thinking_budget: Extended-thinking token budget, or None.
    """
    # Current-message attachments (this turn only).
    attachment_blocks: list[dict] = []
    labels: list[str] = []
    if files:
        try:
            attachment_blocks, labels = attachments.build_attachment_blocks(files)
        except attachments.AttachmentError as exc:
            st.error(str(exc))
            return

    # Conversation-level documents (cached, re-sent every turn).
    try:
        document_prefix = _build_document_prefix(conversation_files)
    except attachments.AttachmentError as exc:
        st.error(str(exc))
        return

    # Content actually sent for this turn: attachments first, then text.
    api_content: list[dict] = list(attachment_blocks)
    if text:
        api_content.append({"type": "text", "text": text})
    if not api_content:
        return  # nothing to send

    display_text = _build_display_text(text, labels)
    with st.chat_message("user"):
        st.markdown(display_text)

    # Document prefix, then prior (lightweight) history, then this turn.
    api_messages = list(document_prefix)
    api_messages.extend(st.session_state.chat_messages)
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
    web_search_enabled, thinking_level, conversation_files = (
        _render_sidebar_controls()
    )

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
            submission.text or "",
            submission.files,
            conversation_files,
            role,
            tools,
            budget,
        )
