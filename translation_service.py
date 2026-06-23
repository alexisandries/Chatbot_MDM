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
    translate_text(...)                Detect glossary terms, translate the
                                       text (chunked if long), and return a
                                       TranslationResult.
    upgrade_translation(...)           Improve an existing translation with
                                       a stronger model and optional user
                                       feedback.
    check_compliance(...)              Passive downstream check: report
                                       official terms missing from a
                                       translation.

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

from dataclasses import dataclass

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

# Rough characters-per-token ratio for European languages. Used only for
# the advisory estimate, never for billing or hard limits.
_CHARS_PER_TOKEN = 4

# Placeholder inserted when the user upgrades without typing any feedback,
# so the upgrade prompt always has a consistent structure.
_NO_FEEDBACK_PLACEHOLDER = "(no specific feedback provided)"


class TranslationError(Exception):
    """Raised for any translation problem the user should see.

    The message is always human-readable and safe to display directly in
    the UI. This is the only exception type the Translation view needs to
    catch from this module.
    """


@dataclass(frozen=True)
class TranslationResult:
    """The outcome of a base translation.

    Attributes:
        text: The translated text.
        glossary_instructions: The terminology instruction block that was
            injected into the translation prompt. The view keeps this so
            an upgrade can reuse it without paying for glossary detection
            again.
        source_language_code: The ISO code the text was translated from.
        source_language_name: The human-readable source language name,
            convenient for display.
    """

    text: str
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


# ---------------------------------------------------------------------------
# Base translation
# ---------------------------------------------------------------------------

def translate_text(
    text: str,
    source_language_code: str,
    target_language_code: str,
    role: str,
) -> TranslationResult:
    """Translate a text, enforcing the institutional glossary.

    Detects glossary terms in the source, injects them as binding
    instructions, then translates the text. Long texts are split into
    chunks and translated piece by piece; the pieces are joined back
    together.

    Args:
        text: The source text to translate.
        source_language_code: ISO code of the source language (e.g.
            "fr"). Typically obtained from language.detect_language().
        target_language_code: ISO code of the target language (e.g.
            "nl"). Typically chosen by the user.
        role: The model role to translate with ("economy" or "standard").

    Returns:
        A TranslationResult with the translated text and the glossary
        instructions that were used.

    Raises:
        TranslationError: If the text is empty, if source and target
            languages are the same, or if the LLM call fails. The message
            is suitable for display to the user.
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
    translated_pieces = []
    for chunk in chunks:
        user_prompt = translation_prompts.build_translation_user_prompt(
            chunk, glossary_instructions
        )
        try:
            piece = llm_client.complete(
                role=role, system=system_prompt, prompt=user_prompt
            )
        except llm_client.LLMError as exc:
            raise TranslationError(str(exc)) from exc
        translated_pieces.append(piece)

    return TranslationResult(
        text=" ".join(translated_pieces),
        glossary_instructions=glossary_instructions,
        source_language_code=source_language_code,
        source_language_name=source_name,
    )


# ---------------------------------------------------------------------------
# Upgrade / refinement
# ---------------------------------------------------------------------------

def upgrade_translation(
    source_text: str,
    current_translation: str,
    user_feedback: str,
    source_language_code: str,
    target_language_code: str,
    role: str = "premium",
    glossary_instructions: str | None = None,
) -> str:
    """Produce an improved version of an existing translation.

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
        The improved translation text.

    Raises:
        TranslationError: If there is no translation to upgrade, if the
            translation is too long to upgrade in one pass, or if the LLM
            call fails. The message is suitable for display to the user.
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

    feedback = user_feedback.strip() if user_feedback else ""
    if not feedback:
        feedback = _NO_FEEDBACK_PLACEHOLDER

    system_prompt = translation_prompts.build_upgrade_system_prompt(target_name)
    user_prompt = translation_prompts.build_upgrade_user_prompt(
        source_text, current_translation, feedback, glossary_instructions
    )

    try:
        return llm_client.complete(
            role=role, system=system_prompt, prompt=user_prompt
        )
    except llm_client.LLMError as exc:
        raise TranslationError(str(exc)) from exc


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
