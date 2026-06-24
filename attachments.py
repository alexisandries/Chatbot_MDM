"""Turn user-uploaded files into Anthropic message content blocks.

The chatbot lets users attach files to a message. This module converts
each uploaded file into the content-block format the Anthropic Messages
API expects, choosing the best representation per type:

- Images (PNG, JPEG, GIF, WebP)  -> an "image" block, sent natively.
- PDF                            -> a "document" block, sent natively so
                                    Claude reads both text and layout.
- Word / PowerPoint / Excel      -> a "text" block holding the text
                                    extracted from the file (these
                                    formats are not read natively, so the
                                    file_readers extractors are reused).

It also returns a short label per file (its name), used to show the user
what they attached and to keep a lightweight trace in the conversation
once the file itself is no longer re-sent.

SIZE LIMITS
===========
Images are capped lower than other files because the API limits a single
image to a few megabytes, whereas PDFs may be larger. Oversized files
raise AttachmentError with a message suitable for display.
"""

import base64

import file_readers


# Per-type maximum upload sizes, in megabytes.
_MAX_IMAGE_MB = 5
_MAX_FILE_MB = 20

# MIME types Claude can read as native images.
_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}

# MIME type Claude can read as a native document.
_PDF_MIME_TYPE = "application/pdf"

# Office MIME types we cannot send natively: their text is extracted with
# the matching file_readers function instead.
_OFFICE_READERS = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_readers.read_docx,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        file_readers.read_pptx,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        file_readers.read_excel,
}

# File extensions to allow in the chat input's uploader. Kept here so the
# view and this module agree on what is accepted.
ALLOWED_EXTENSIONS = [
    "png", "jpg", "jpeg", "gif", "webp",  # images
    "pdf",                                 # native document
    "docx", "pptx", "xlsx",               # office (text extracted)
]


class AttachmentError(Exception):
    """Raised when an attachment is too large or of an unsupported type.

    The message is human-readable and safe to show to the user.
    """


def _encode_base64(data: bytes) -> str:
    """Return the standard base64 encoding of bytes as an ASCII string."""
    return base64.standard_b64encode(data).decode("utf-8")


def _check_size(uploaded_file, max_mb: int) -> None:
    """Raise AttachmentError if an uploaded file exceeds a size limit.

    Args:
        uploaded_file: The uploaded file (exposes .name and .size).
        max_mb: The maximum allowed size in megabytes.
    """
    if uploaded_file.size > max_mb * 1024 * 1024:
        raise AttachmentError(
            f"'{uploaded_file.name}' is too large (max {max_mb} MB)."
        )


def _image_block(uploaded_file) -> dict:
    """Build a native image content block from an uploaded image."""
    _check_size(uploaded_file, _MAX_IMAGE_MB)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": uploaded_file.type,
            "data": _encode_base64(uploaded_file.getvalue()),
        },
    }


def _pdf_block(uploaded_file) -> dict:
    """Build a native document content block from an uploaded PDF."""
    _check_size(uploaded_file, _MAX_FILE_MB)
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": _PDF_MIME_TYPE,
            "data": _encode_base64(uploaded_file.getvalue()),
        },
    }


def _office_text_block(uploaded_file) -> dict:
    """Build a text content block from an uploaded Office file.

    The file's text is extracted with the matching file_readers function
    and wrapped with a header naming the file, so the model knows where
    the text came from.
    """
    _check_size(uploaded_file, _MAX_FILE_MB)
    reader = _OFFICE_READERS[uploaded_file.type]
    text = reader(uploaded_file)
    return {
        "type": "text",
        "text": f"Attached file '{uploaded_file.name}':\n\n{text}",
    }


def build_attachment_blocks(
    uploaded_files,
    cache_last_block: bool = False,
) -> tuple[list[dict], list[str]]:
    """Convert uploaded files into API content blocks and labels.

    Args:
        uploaded_files: A list of uploaded files (each exposes .name,
            .type, .size and the usual file-like interface), as returned
            by st.chat_input(...).files or st.file_uploader(...).
        cache_last_block: When True, a prompt-cache breakpoint is placed
            on the last block, so that everything up to and including it
            (typically the system prompt plus all these document blocks)
            is cached and reused cheaply on later turns. Use this for
            conversation-level documents that are re-sent every turn;
            leave it False for one-off, current-message attachments.

    Returns:
        A (blocks, labels) tuple where blocks is the list of content
        blocks ready to include in a user message, and labels is the list
        of file names, in the same order.

    Raises:
        AttachmentError: If any file is too large or of an unsupported
            type. The message is suitable for display to the user.
    """
    blocks: list[dict] = []
    labels: list[str] = []

    for uploaded_file in uploaded_files:
        mime_type = uploaded_file.type
        if mime_type in _IMAGE_MIME_TYPES:
            blocks.append(_image_block(uploaded_file))
        elif mime_type == _PDF_MIME_TYPE:
            blocks.append(_pdf_block(uploaded_file))
        elif mime_type in _OFFICE_READERS:
            blocks.append(_office_text_block(uploaded_file))
        else:
            raise AttachmentError(
                f"'{uploaded_file.name}' has an unsupported type "
                f"({mime_type})."
            )
        labels.append(uploaded_file.name)

    if cache_last_block and blocks:
        # Mark a cache breakpoint on the final block. The cached prefix
        # then covers the system prompt and every document block, so the
        # heavy document tokens are re-read cheaply on subsequent turns.
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}

    return blocks, labels
