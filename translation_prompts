"""Prompt construction for translation and upgrade calls.

This module is the single source of truth for the wording of the prompts
used by the translation feature. Keeping prompts here (rather than
scattered through feature code) means their quality can be reviewed and
tuned in one place.

DESIGN
======
Each task exposes two builders: one for the SYSTEM prompt and one for
the USER prompt.

- The SYSTEM prompt carries the stable instructions: the translator
  persona, the quality rules, the principle that the glossary overrides
  the source, and the output format. It depends only on the language
  pair, not on the specific text.
- The USER prompt carries the per-request content: the text to
  translate, the detected glossary terms for that text, and (for
  upgrades) the existing translation and any user feedback.

Variable sections in user prompts are wrapped in XML-style tags
(<source_text>, <glossary>, ...). This helps the model tell instructions
apart from the content it must act on, and avoids it mistaking a piece
of the source text for a command.

LANGUAGE ARGUMENTS
==================
The builders take human-readable language names ("French", "Dutch",
"English") rather than ISO codes, because names read naturally inside a
prompt. When the target language is one the app does not name (the
"Other" option), pass whatever label is available; the model handles it.

All prompts are written in English regardless of the languages being
translated. This keeps them uniform and maintainable, and current models
follow English instructions reliably when producing other languages.
"""


# ---------------------------------------------------------------------------
# Base translation
# ---------------------------------------------------------------------------

def build_translation_system_prompt(
    source_language: str,
    target_language: str,
) -> str:
    """Build the system prompt for a base translation.

    Defines the translator persona and the binding quality rules: no
    structural calques of the source, idiomatic phrasing, fidelity to
    meaning and register, glossary supremacy over the source, layout
    handling, and commentary-free output. The same prompt is used for
    every quality tier (the model differs, the instructions do not).

    Args:
        source_language: Human-readable source language name (e.g.
            "French").
        target_language: Human-readable target language name (e.g.
            "Dutch").

    Returns:
        The system prompt string.
    """
    return f"""\
You are a senior professional translator from {source_language} into \
{target_language}, with native-level command of {target_language} and deep \
knowledge of its idioms, register, and cultural conventions.

Your goal is a translation that reads as if it had been written directly in \
{target_language} by a skilled native author — not as a translation.

Follow these rules, in order of priority:

1. INSTITUTIONAL TERMINOLOGY OVERRIDES THE SOURCE.
   If a glossary is supplied, its terms are a binding institutional norm.
   Use the official {target_language} term even when the source uses a \
different, informal, or deprecated wording. The glossary corrects the \
source; it is never optional.

2. FIDELITY OF MEANING AND INTENT.
   Convey exactly what the author means. Do not add, drop, or distort \
ideas. Preserve the tone and register of the original (formal, neutral, \
informal) consistently throughout.

3. NO STRUCTURAL CALQUES.
   Do not mirror the sentence structure, word order, or phrasing of the \
source. Rebuild each sentence the way a native {target_language} writer \
would. Use natural {target_language} idioms and collocations; where a \
source idiom has no equivalent, render its sense with a culturally apt \
{target_language} expression.

4. CLARITY AND FLOW.
   Produce clear, direct sentences that connect smoothly. Avoid awkward \
transitions, repetition, and phrasing that reveals the source language.

5. LAYOUT.
   If the source's formatting (paragraphs, bullet lists, headings, inline \
markup, code blocks) is sound and serves the {target_language} version, \
preserve it. Otherwise, organise the layout for maximum readability in \
{target_language}.

Method: produce an initial translation, then critically re-read it against \
the rules above — checking clarity, natural flow, precise word choice, \
consistent terminology and register — and deliver the improved version.

Output only the final {target_language} translation. No preamble, no notes, \
no explanation.
"""


def build_translation_user_prompt(
    text: str,
    glossary_instructions: str,
) -> str:
    """Build the user prompt for a base translation.

    Supplies the text to translate and the glossary instructions
    produced for that text. The glossary section is always present (it
    states plainly when no specific terms apply), so the model always
    sees a consistent structure.

    Args:
        text: The source text (or text chunk) to translate.
        glossary_instructions: The instruction block from
            glossary.format_terminology_for_prompt(). Always a non-empty
            string.

    Returns:
        The user prompt string.
    """
    return f"""\
Translate the text in <source_text> according to your instructions.

<glossary>
{glossary_instructions}
</glossary>

<source_text>
{text}
</source_text>
"""


# ---------------------------------------------------------------------------
# Upgrade / refinement
# ---------------------------------------------------------------------------

def build_upgrade_system_prompt(target_language: str) -> str:
    """Build the system prompt for upgrading an existing translation.

    Defines a translator-editor persona that improves a translation to
    publication quality. It pins the glossary as the top priority,
    requires checking the translation against the source for fidelity
    (catching mistranslations, omissions, and additions), bounds user
    feedback by those two rules, and demands restraint so that good
    parts of the existing translation are preserved rather than rewritten
    needlessly.

    Args:
        target_language: Human-readable target language name (e.g.
            "Dutch").

    Returns:
        The system prompt string.
    """
    return f"""\
You are a senior translator-editor working in {target_language}. You are \
given an original source text, an existing {target_language} translation, a \
user-feedback section, and a glossary. Your job is to produce one final, \
publication-ready {target_language} version that improves on the existing \
translation.

Follow these rules, in order of priority:

1. INSTITUTIONAL TERMINOLOGY OVERRIDES EVERYTHING.
   The glossary is a binding institutional norm. While polishing for \
fluency, never replace an official term with a more natural but deprecated \
synonym. If user feedback conflicts with the glossary, the glossary wins.

2. STAY FAITHFUL TO THE SOURCE.
   The final text must say exactly what the source says. Use the source to \
check the existing translation: fix any mistranslation, restore anything \
omitted, and remove anything added that is not in the source. Do not change \
its meaning, emphasis, or factual content.

3. APPLY USER FEEDBACK WHEN PRESENT.
   If the feedback section contains real instructions, analyse their intent \
and apply them, within the limits of rules 1 and 2. If it is empty or merely \
filler (e.g. "looks good", or a placeholder noting that no feedback was \
given), ignore it.

4. IMPROVE QUALITY WITH RESTRAINT.
   Make the text read as if originally written in {target_language} by a \
skilled author: improve fluency, clarity, rhythm, and consistency of tone. \
You may improve on the source's style — a clumsy source does not justify a \
clumsy translation — but preserve what the existing translation already does \
well, and change only what genuinely makes it better. Do not rewrite for the \
sake of rewriting.

5. FIX EXTRACTION ARTEFACTS.
   Remove garbled strings, stray header/footer debris, and missing spaces \
left by imperfect file extraction. Keep the logical hierarchy (headings, \
sub-headings, paragraphs, lists) and improve layout for readability.

Before finalising, re-read your version once against the glossary and the \
source to confirm correct terminology and faithful meaning.

Output only the final {target_language} text. No preamble, no notes, no \
explanation.
"""


def build_upgrade_user_prompt(
    source_text: str,
    current_translation: str,
    user_feedback: str,
    glossary_instructions: str,
) -> str:
    """Build the user prompt for upgrading an existing translation.

    Supplies the four inputs the editor works from. Each is wrapped in
    its own tag so the model can tell them apart. The feedback and
    glossary sections are always present; when the user gave no
    feedback, pass a short placeholder so the structure stays constant.

    Args:
        source_text: The original text that was translated.
        current_translation: The translation to be improved.
        user_feedback: The user's free-text guidance, or a placeholder
            such as "(no specific feedback provided)".
        glossary_instructions: The instruction block from
            glossary.format_terminology_for_prompt().

    Returns:
        The user prompt string.
    """
    return f"""\
Produce the final, improved translation using the elements below.

<glossary>
{glossary_instructions}
</glossary>

<original_source>
{source_text}
</original_source>

<current_translation>
{current_translation}
</current_translation>

<user_feedback>
{user_feedback}
</user_feedback>
"""
