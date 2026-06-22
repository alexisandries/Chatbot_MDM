"""Text chunking for long inputs.

Some texts are too long to send to a language model in a single call.
This module splits a long text into smaller pieces ("chunks") while
trying to keep each chunk's boundaries on natural sentence breaks, so
that no sentence is cut in half unless it is itself longer than the
allowed chunk size.

The functions here are pure: they take strings and return strings, with
no dependency on Streamlit or any external service. This makes them easy
to test in isolation and safe to call from anywhere in the application.
"""

import re


# Sentence boundary: a period, question mark or exclamation mark,
# followed by whitespace and a capital letter. This is a heuristic, not
# a full linguistic parser, but it is good enough to avoid splitting in
# the middle of most sentences.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")


def _hard_split(text: str, max_len: int) -> list[str]:
    """Split a text into fixed-size pieces, ignoring sentence boundaries.

    Used as a fallback when a single sentence is longer than max_len, or
    when sentence-based splitting cannot be applied.

    Args:
        text: The text to split.
        max_len: Maximum length in characters for each piece.

    Returns:
        A list of pieces, each at most max_len characters long. Empty if
        the input is empty.
    """
    if not text:
        return []
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]


def chunk_text(text: str, max_len: int = 5000) -> list[str]:
    """Split a long text into chunks no longer than max_len characters.

    The text is first broken into sentences. Sentences are then packed
    into chunks so that each chunk stays within max_len. A sentence that
    is on its own longer than max_len is split into fixed-size pieces.

    Args:
        text: The input text. May be empty.
        max_len: Maximum number of characters per chunk. Choose a value
            comfortably below the model's input limit, leaving room for
            the prompt and instructions that will accompany each chunk.

    Returns:
        A list of text chunks, in order. Concatenating the chunks (with
        spaces) reproduces the input's content. Returns an empty list
        for empty input.
    """
    if not text:
        return []

    sentences = _SENTENCE_BOUNDARY.split(text)

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # A single sentence longer than the limit cannot be packed; flush
        # the current chunk, then split the oversized sentence on its own.
        if len(sentence) > max_len:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.extend(_hard_split(sentence, max_len))
            continue

        # Would adding this sentence (plus a separating space) overflow?
        separator_len = 1 if current_chunk else 0
        if len(current_chunk) + separator_len + len(sentence) <= max_len:
            current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
        else:
            # Current chunk is full: store it and start a fresh one.
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    # Safety net: if sentence splitting produced nothing usable but the
    # input was non-empty, fall back to a plain fixed-size split.
    if not chunks:
        return _hard_split(text, max_len)

    return chunks
