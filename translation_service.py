"""Translation pipeline: the business logic behind the Translation view.

This module orchestrates everything needed to turn a source text into a
translation, and to upgrade an existing translation. It ties together the
glossary checker, the text chunker, the prompt builders, and the LLM
gateway. It contains NO Streamlit UI code, so it can be tested on its own
and reused.

WHAT THE VIEW CALLS
===================
    estimate_tokens(text)              Rough, free token estimate, used by
                                       the UI to warn before an expensive
                                       translation.
    estimate_document_tokens(files)    Same, for natively-read documents.
    stream_text_translation(...)       Detect glossary terms, then stream
                                       the translation of a text (chunked
                                       if long).
    stream_document_translation(...)   Stream the translation of a PDF or
                                       image read natively.
    stream_upgrade(...)                Stream an improved version of an
                                       existing translation, using a
                                       stronger model.
    check_compliance(...)              Passive downstream check: report
                                       official terms missing from a
                                       translation.

EVERYTHING STREAMS
==================
The three text-producing functions return their output as an iterator of
text fragments rather than a finished string, so the view can display the
translation as it is written instead of leaving the user in front of a
spinner. Two consequences worth knowing:

  1. Validation happens immediately, streaming happens later.
     Each function validates its arguments and raises TranslationError
     straight away, before any fragment is produced. Failures that only
     appear during the API call (rate limit, network) surface later, when
     the caller consumes the iterator. The caller therefore needs a
     try/except around the consumption too, not only around the call.

  2. A mid-stream failure loses the partial text.
     Fragments already yielded are gone once the exception propagates.
     This is acceptable because a half-finished translation is not usable
     anyway.

The metadata a caller needs to store alongside the text (glossary
instructions, source language) is known before the first fragment, so it
is returned up front in a TranslationStream. See that class for details.

ERRORS
======
Every problem the view should surface to the user is raised as a single
exception type, TranslationError, whose message is safe to display via
st.error(). Internally this wraps the lower-level LLMError so the view
only has to catch one thing.

LANGUAGE CODES
==============
All functions take ISO codes ("fr", "nl", "en", or a code typed by the
user for the "Other" option). Human-readable names for the prompts are
derived internally via the language module.

GLOSSARY IS ENFORCED AT EVERY TEXT-PRODUCING STEP
=================================================
Glossary instructions are injected both when translating and when
upgrading, because an upgrade can otherwise reintroduce a deprecated term
while polishing for fluency. The passive compliance check is offered
separately so the view can flag any remaining discrepancy to the user.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import fitz  # PyMuPDF, used to count PDF pages

import attachments
import chunking
import glossary
import language
import llm_client
import translation_prompts

# Maximum characters per translation chunk. Translation output is roughly
# as long as its input, so this is kept well within the models' output
# token limits while still being large enough that most texts translate
# in a single call (which preserves cross-sentence consistency).
TRANSLATION_CHUNK_CHARS = 12000

# Maximum length of a translation that can be upgraded in a single pass.
# The upgrade step needs the whole translation in one call so the editor
# can see the full text; beyond this size the improved output risks being
# truncated, so we ask the user to upgrade in smaller sections instead.
UPGRADE_MAX_CHARS = 40000

# Above this estimated token count, the view should warn the user about
# cost before translating. Purely advisory; nothing is blocked.
TOKEN_WARNING_THRESHOLD = 30000

# Maximum number of pages in a PDF translated natively in one request,
# matching the Anthropic API's per-request document limit.
MAX_DOCUMENT_PAGES = 100

# Reasoning level (see models_config.REASONING_LEVELS) applied to each
# step. Translating is a transformation task, not a reasoning one, so
# thinking is off: it would add cost and latency without improving the
# result. The upgrade is the deliberate quality step, where weighing
# register, idiom and terminology against each other does pay off.
_TRANSLATION_REASONING = "Off"
_UPGRADE_REASONING = "Standard"

# Rough characters-per-token ratio for European languages. Used only for
# the advisory estimate, never for billing or hard limits.
_CHARS_PER_TOKEN = 4

# Rough token cost of a natively-read PDF page and image, for the advisory
# estimate only.
_PDF_TOKENS_PER_PAGE = 2000
_IMAGE_TOKENS = 1600

# Separator inserted between two translated chunks of the same text.
_CHUNK_SEPARATOR = " "

# Placeholder inserted when the user upgrades without typing any feedback,
# so the upgrade prompt always has a consistent structure.
_NO_FEEDBACK_PLACEHOLDER = "(no specific feedback provided)"

# Placeholder used as the "source" when upgrading a translation that was
# produced from a natively-read document (no plain-text source exists).
_NO_SOURCE_PLACEHOLDER = (
    "(The source was a document read directly, so no plain-text source is "
    "available. Focus on improving quality and applying the glossary.)"
)


class TranslationError(Exception):
    """Raised for any translation problem the user should see.

    The message is always human-readable and safe to display directly in
    the UI. This is the only exception type the Translation view needs to
    catch from this module.
    """


@dataclass(frozen=True)
class TranslationStream:
    """A translation that is about to be produced, fragment by fragment.

    Returned by the two stream_*_translation functions. The metadata
    fields are final as soon as the object exists; the text itself only
    materialises as `fragments` is consumed, and can be consumed exactly
    once.

    Attributes:
        fragments: Iterator of text fragments in order. Concatenating
            them all gives the complete translation. Consuming it is what
            triggers the API call, so this is also where a mid-stream
            TranslationError can be raised.
        glossary_instructions: The terminology instruction block injected
            into the translation prompt. Callers keep this so an upgrade
            can reuse it without paying for glossary detection again.
        source_language_code: The ISO code the text is translated from.
            Empty for natively-read documents, where the source language
            is not detected.
        source_language_name: Human-readable source language name,
            convenient for display. "Document" for natively-read files.
    """

    fragments: Iterator[str]
    glossary_instructions: str
    source_language_code: str
    source_language_name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text.

    This is a deliberately cheap, offline approximation (characters
    divided by a fixed ratio). It is meant only to drive a cost warning
    in the UI, not to compute exact usage or enforce hard limits.

    Args:
        text: The text to estimate.

    Returns:
        An approximate token count (always >= 0).
    """
    if not text:
        return 0
    return len(text) // _CHARS_PER_TOKEN


def _glossary_instructions(
    text: str,
    source_language_code: str,
    target_language_code: str,
) -> str:
    """Detect glossary terms in a text and format them for a prompt.

    Combines the two glossary steps (detection then formatting) into the
    single string the prompt builders expect. Always returns a non-empty
    string: when no terms apply, the formatter returns a short note to
    that effect.

    Args:
        text: The source text to scan for glossary terms.
        source_language_code: ISO code of the source language.
        target_language_code: ISO code of the target language.

    Returns:
        The terminology instruction block for the prompt.
    """
    matches = glossary.detect_glossary_terms(
        text, source_language_code, target_language_code
    )
    return glossary.format_terminology_for_prompt(
        matches, source_language_code, target_language_code
    )


def _stream_prompt(
    role: str,
    system_prompt: str,
    content,
    reasoning: str,
) -> Iterator[str]:
    """Stream one LLM call, converting gateway errors into view errors.

    Every call this module makes goes through here, so the translation of
    LLMError into TranslationError happens in exactly one place.

    Args:
        role: The model role to call.
        system_prompt: System prompt for the call.
        content: The user message content. Either a plain string, or the
            list of content blocks used when a document is attached.
        reasoning: Reasoning level name passed to the gateway.

    Yields:
        Text fragments in order.

    Raises:
        TranslationError: If the call fails. Because this is a generator,
            the error surfaces while the caller consumes the fragments.
    """
    try:
        yield from llm_client.stream(
            role=role,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
            reasoning=reasoning,
        )
    except llm_client.LLMError as exc:
        raise TranslationError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Base translation
# ---------------------------------------------------------------------------

def _iter_text_translation(
    chunks: list[str],
    glossary_instructions: str,
    system_prompt: str,
    role: str,
) -> Iterator[str]:
    """Stream the translation of a text, chunk after chunk.

    Each chunk is a separate API call, but the fragments of all chunks
    form one continuous stream, so the reader sees a single text being
    written rather than a series of blocks.

    Args:
        chunks: The source text already split into translatable pieces.
        glossary_instructions: Terminology block to inject in each call.
        system_prompt: The translation system prompt.
        role: The model role to translate with.

    Yields:
        Text fragments in order, with a separator between chunks.

    Raises:
        TranslationError: If any chunk's API call fails.
    """
    for index, chunk in enumerate(chunks):
        if index:
            yield _CHUNK_SEPARATOR

        user_prompt = translation_prompts.build_translation_user_prompt(
            chunk, glossary_instructions
        )
        yield from _stream_prompt(
            role, system_prompt, user_prompt, _TRANSLATION_REASONING
        )


def stream_text_translation(
    text: str,
    source_language_code: str,
    target_language_code: str,
    role: str,
) -> TranslationStream:
    """Prepare a streamed translation of a text, enforcing the glossary.

    Detects glossary terms in the source and injects them as binding
    instructions. Long texts are split into chunks translated one after
    the other; the caller sees a single continuous stream.

    Glossary detection runs here, before returning, so the caller has the
    terminology block available immediately.

    Args:
        text: The source text to translate.
        source_language_code: ISO code of the source language (e.g.
            "fr"). Typically obtained from language.detect_language().
        target_language_code: ISO code of the target language (e.g.
            "nl"). Typically chosen by the user.
        role: The model role to translate with ("economy", "standard" or
            "premium").

    Returns:
        A TranslationStream. Consume its `fragments` to produce the text.

    Raises:
        TranslationError: Immediately if the text is empty or if source
            and target languages are the same; later, while the fragments
            are consumed, if an API call fails.
    """
    if not text or not text.strip():
        raise TranslationError("There is no text to translate.")

    if source_language_code and source_language_code == target_language_code:
        raise TranslationError(
            "The source and target languages are the same. "
            "Choose a different target language."
        )

    source_name = language.language_name(source_language_code)
    target_name = language.language_name(target_language_code)

    glossary_instructions = _glossary_instructions(
        text, source_language_code, target_language_code
    )
    system_prompt = translation_prompts.build_translation_system_prompt(
        source_name, target_name
    )
    chunks = chunking.chunk_text(text, max_len=TRANSLATION_CHUNK_CHARS)

    return TranslationStream(
        fragments=_iter_text_translation(
            chunks, glossary_instructions, system_prompt, role
        ),
        glossary_instructions=glossary_instructions,
        source_language_code=source_language_code,
        source_language_name=source_name,
    )


# ---------------------------------------------------------------------------
# Native document translation (PDF / image)
# ---------------------------------------------------------------------------

def _pdf_page_count(uploaded_file) -> int:
    """Return the number of pages in an uploaded PDF.

    Args:
        uploaded_file: The uploaded PDF (exposes .getvalue()).

    Returns:
        The page count, or 0 if the PDF cannot be read.
    """
    try:
        with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
            return doc.page_count
    except Exception:
        return 0


def estimate_document_tokens(files) -> int:
    """Estimate the input token cost of natively translating documents.

    For PDFs the estimate is based on the page count; for images, a flat
    per-image figure. This is a coarse, advisory figure used only to warn
    about cost in the UI, never for billing.

    Args:
        files: The uploaded PDF/image files to be translated natively.

    Returns:
        An approximate token count (always >= 0).
    """
    total = 0
    for uploaded_file in files:
        if uploaded_file.type == "application/pdf":
            pages = _pdf_page_count(uploaded_file) or 1
            total += pages * _PDF_TOKENS_PER_PAGE
        else:
            total += _IMAGE_TOKENS
    return total


def stream_document_translation(
    files,
    target_language_code: str,
    role: str,
) -> TranslationStream:
    """Prepare a streamed translation of a natively-read PDF or image.

    The document is sent to the model as-is (not extracted to text first),
    so the model sees its layout and can drop artefacts, follow the correct
    reading order, and translate text embedded in visuals. The full
    glossary is injected as binding guidance, since there is no extracted
    source text in which to detect terms beforehand.

    Args:
        files: The uploaded PDF/image files to translate (usually one).
        target_language_code: ISO code of the target language.
        role: The model role to translate with ("economy", "standard" or
            "premium").

    Returns:
        A TranslationStream whose source_language_code is empty (the
        source language is not detected for native documents) and whose
        source_language_name is "Document".

    Raises:
        TranslationError: Immediately if there is no document, if a PDF
            exceeds the page limit, or if a file cannot be attached;
            later, while the fragments are consumed, if the API call
            fails.
    """
    if not files:
        raise TranslationError("There is no document to translate.")

    for uploaded_file in files:
        if uploaded_file.type == "application/pdf":
            pages = _pdf_page_count(uploaded_file)
            if pages > MAX_DOCUMENT_PAGES:
                raise TranslationError(
                    f"This PDF has {pages} pages; the maximum for a single "
                    f"translation is {MAX_DOCUMENT_PAGES}. Please split it "
                    "into smaller files."
                )

    target_name = language.language_name(target_language_code)
    glossary_instructions = glossary.format_full_glossary_for_prompt(
        target_language_code
    )

    try:
        document_blocks, _labels = attachments.build_attachment_blocks(files)
    except attachments.AttachmentError as exc:
        raise TranslationError(str(exc)) from exc

    system_prompt = translation_prompts.build_native_translation_system_prompt(
        target_name
    )
    user_text = translation_prompts.build_native_translation_user_text(
        glossary_instructions
    )

    content = list(document_blocks)
    content.append({"type": "text", "text": user_text})

    return TranslationStream(
        fragments=_stream_prompt(
            role, system_prompt, content, _TRANSLATION_REASONING
        ),
        glossary_instructions=glossary_instructions,
        source_language_code="",
        source_language_name="Document",
    )


# ---------------------------------------------------------------------------
# Upgrade / refinement
# ---------------------------------------------------------------------------

def stream_upgrade(
    source_text: str,
    current_translation: str,
    user_feedback: str,
    source_language_code: str,
    target_language_code: str,
    role: str = "premium",
    glossary_instructions: str | None = None,
) -> Iterator[str]:
    """Prepare a streamed, improved version of an existing translation.

    Runs the translation through a stronger model with an editor prompt
    that checks fidelity against the source, applies optional user
    feedback, and keeps the institutional glossary binding so that no
    official term is lost during polishing.

    Args:
        source_text: The original text that was translated. Needed so the
            editor can verify fidelity and (if not supplied) re-detect
            glossary terms.
        current_translation: The translation to improve.
        user_feedback: Free-text guidance from the user. May be empty.
        source_language_code: ISO code of the source language.
        target_language_code: ISO code of the target language.
        role: The model role used for the upgrade. Defaults to "premium"
            so an upgrade automatically uses the strongest model.
        glossary_instructions: The terminology block from a prior
            translation. If provided, glossary detection is skipped to
            avoid a redundant call; if None, it is computed from the
            source text.

    Returns:
        An iterator of text fragments forming the improved translation.

    Raises:
        TranslationError: Immediately if there is no translation to
            upgrade or if it is too long for a single pass; later, while
            the fragments are consumed, if the API call fails.
    """
    if not current_translation or not current_translation.strip():
        raise TranslationError("There is no translation to upgrade.")

    if len(current_translation) > UPGRADE_MAX_CHARS:
        raise TranslationError(
            "This translation is too long to upgrade in one pass "
            f"(over {UPGRADE_MAX_CHARS} characters). "
            "Upgrade it in smaller sections instead."
        )

    target_name = language.language_name(target_language_code)

    if glossary_instructions is None:
        glossary_instructions = _glossary_instructions(
            source_text, source_language_code, target_language_code
        )

    # A translation produced from a natively-read document has no
    # plain-text source. Substitute a placeholder so the editor focuses on
    # quality and terminology rather than a source it cannot see.
    source_for_prompt = (
        source_text
        if source_text and source_text.strip()
        else _NO_SOURCE_PLACEHOLDER
    )

    feedback = user_feedback.strip() if user_feedback else ""
    if not feedback:
        feedback = _NO_FEEDBACK_PLACEHOLDER

    system_prompt = translation_prompts.build_upgrade_system_prompt(target_name)
    user_prompt = translation_prompts.build_upgrade_user_prompt(
        source_for_prompt, current_translation, feedback, glossary_instructions
    )

    return _stream_prompt(
        role, system_prompt, user_prompt, _UPGRADE_REASONING
    )


# ---------------------------------------------------------------------------
# Downstream passive compliance check
# ---------------------------------------------------------------------------

def check_compliance(
    source_text: str,
    translated_text: str,
    source_language_code: str,
    target_language_code: str,
) -> list[dict]:
    """Report official glossary terms missing from a translation.

    A thin wrapper around the glossary compliance check, exposed here so
    the view imports a single module. The check is passive: it only
    reports discrepancies for the user to review and never edits the
    translation.

    Args:
        source_text: The original text.
        translated_text: The translation to verify.
        source_language_code: ISO code of the source language.
        target_language_code: ISO code of the target language.

    Returns:
        A list of discrepancy dicts, each with "glossary_term" and
        "official_translation". Empty when the translation is compliant
        or when no glossary terms apply.
    """
    return glossary.check_translation_compliance(
        source_text, translated_text, source_language_code, target_language_code
    )
