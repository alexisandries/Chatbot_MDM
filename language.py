"""Language detection and ISO-code/name mapping.

Two parts of the application need to reason about languages in different
forms:
- The glossary and the translation API work with ISO codes ("fr", "nl",
  "en").
- The prompts and the user interface read better with human-readable
  names ("French", "Dutch", "English").

This module centralises both: it detects the language of a text and
converts between codes and names. Keeping this logic in one place avoids
scattering ad-hoc language tables across the codebase.
"""

import langid


# Maps ISO 639-1 codes to English language names. Used to display a
# detected language and to phrase prompts naturally. The list covers the
# languages langid can return; codes not present here fall back to the
# code itself (see language_name).
LANGUAGE_NAMES: dict[str, str] = {
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "ko": "Korean",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "cs": "Czech",
    "el": "Greek",
    "hu": "Hungarian",
}


def language_name(code: str) -> str:
    """Return the human-readable name for an ISO language code.

    Args:
        code: An ISO 639-1 language code (e.g. "fr"). May be a code the
            table does not know (for instance one typed by the user via
            the "Other" option).

    Returns:
        The English language name if the code is known, otherwise the
        code itself unchanged. Returns "Unknown" for an empty code.
    """
    if not code:
        return "Unknown"
    return LANGUAGE_NAMES.get(code, code)


def detect_language(text: str) -> tuple[str, str]:
    """Detect the language of a text.

    Uses the langid classifier, which identifies the most likely
    language from the text's content.

    Args:
        text: The text whose language should be identified.

    Returns:
        A (code, name) tuple, where code is the ISO 639-1 code (e.g.
        "fr") and name is the human-readable name (e.g. "French"). If
        the text is empty or detection fails, returns ("", "Unknown") so
        the caller can prompt the user to choose a language manually
        rather than proceeding with a wrong guess.
    """
    if not text or not text.strip():
        return "", "Unknown"
    try:
        code, _score = langid.classify(text)
    except Exception:
        return "", "Unknown"
    return code, language_name(code)
