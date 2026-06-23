"""Translation view: upload or paste text, translate, upgrade, download.

This module renders the Translation interface and wires together the
input widgets, the translation service, the glossary compliance check,
and the Word export. It holds the UI only; all translation logic lives in
translation_service.

USER FLOW
=========
1. Pick a target language and a model tier (sidebar).
2. Upload a file and/or paste text.
3. Translate: the source language is detected automatically, the text is
   translated, and a passive glossary check flags any official term that
   is missing.
4. Optionally upgrade: a stronger model refines the translation, applying
   any feedback the user typed, while keeping the glossary binding.
5. Download the result as a Word document (refined + base versions).

All results are kept in session_state so they survive the reruns that
Streamlit triggers on every interaction.
"""

import streamlit as st

import doc_export
import file_readers
import language
import translation_service
from session import select_translation_model


# Target languages offered as named options, mapped to their ISO codes.
# "Other" lets the user type any ISO code.
_NAMED_TARGET_LANGUAGES = {"Dutch": "nl", "French": "fr", "English": "en"}


def _select_target_language() -> str:
    """Render the target-language picker and return the chosen ISO code.

    Shows a dropdown of named languages plus an "Other" option that
    reveals a free-text field for any ISO code.

    Returns:
        The target language ISO code (e.g. "nl"). May be an empty string
        if "Other" is selected but no code has been typed yet.
    """
    options = list(_NAMED_TARGET_LANGUAGES) + ["Other"]
    # Default to French (index 1), the most common target in practice.
    choice = st.selectbox(
        "Target language", options, index=1, key="translation_target_choice"
    )
    if choice == "Other":
        return st.text_input(
            "Target language ISO code (e.g. 'de')",
            key="translation_target_other",
        ).strip().lower()
    return _NAMED_TARGET_LANGUAGES[choice]


def _gather_input() -> str:
    """Render the file uploader and text area, and combine their content.

    Extracts text from an uploaded file (showing an error if the file is
    too large or unsupported) and appends any manually entered text.

    Returns:
        The combined source text, or an empty string if nothing was
        provided.
    """
    uploaded_file = st.file_uploader(
        "Upload a file (PDF, PPTX, XLSX, DOCX)",
        type=["pdf", "pptx", "xlsx", "docx"],
        key="translation_file",
    )

    file_text = ""
    if uploaded_file is not None:
        try:
            file_text = file_readers.extract_text(uploaded_file)
        except (file_readers.FileTooLargeError, file_readers.UnsupportedFileError) as exc:
            st.error(str(exc))
        if file_text:
            st.info(
                "Text extracted from PDFs or slides can be dense or out of "
                "order. You can use the upgrade feedback box to request "
                "layout improvements."
            )

    manual_text = st.text_area(
        "Or enter text to translate", height=150, key="translation_manual"
    )

    parts = [part for part in (file_text, manual_text) if part and part.strip()]
    return "\n".join(parts)


def _render_compliance(discrepancies: list[dict]) -> None:
    """Show a glossary-compliance warning, if there are any discrepancies.

    Args:
        discrepancies: The list returned by
            translation_service.check_compliance(). Each item has
            "glossary_term" and "official_translation". An empty list
            renders nothing.
    """
    if not discrepancies:
        return
    lines = [
        f"- The official term **{item['official_translation']}** "
        f"(for '{item['glossary_term']}') appears to be missing or replaced."
        for item in discrepancies
    ]
    st.warning(
        "Glossary check — please review the following terms:\n\n"
        + "\n".join(lines)
    )


def _run_translation(combined_text: str, target_code: str, role: str) -> None:
    """Translate the input and store the results in session_state.

    Detects the source language, calls the translation service, and runs
    the passive glossary compliance check. Any TranslationError is shown
    to the user. Clears any previous refined translation so the view does
    not show a stale upgrade.

    Args:
        combined_text: The source text to translate.
        target_code: Target language ISO code.
        role: The model role to translate with.
    """
    source_code, source_name = language.detect_language(combined_text)
    if not source_code:
        st.error(
            "The source language could not be detected. Please add more "
            "text or check the input."
        )
        return

    try:
        with st.spinner("Translating..."):
            result = translation_service.translate_text(
                combined_text, source_code, target_code, role
            )
        with st.spinner("Checking terminology..."):
            compliance = translation_service.check_compliance(
                combined_text, result.text, source_code, target_code
            )
    except translation_service.TranslationError as exc:
        st.error(str(exc))
        return

    st.session_state.translation_raw = result.text
    st.session_state.translation_glossary_instructions = result.glossary_instructions
    st.session_state.translation_source_text = combined_text
    st.session_state.translation_source_code = result.source_language_code
    st.session_state.translation_source_name = result.source_language_name
    st.session_state.translation_target_code = target_code
    st.session_state.translation_compliance = compliance
    # A new base translation invalidates any earlier refined version.
    st.session_state.translation_refined = ""
    st.session_state.translation_refined_compliance = []


def _run_upgrade(user_feedback: str) -> None:
    """Upgrade the stored base translation and store the refined result.

    Uses the premium model and reuses the glossary instructions from the
    base translation to avoid a second detection call. Any
    TranslationError is shown to the user.

    Args:
        user_feedback: Free-text guidance from the user. May be empty.
    """
    try:
        with st.spinner("Upgrading with the premium model..."):
            refined = translation_service.upgrade_translation(
                source_text=st.session_state.translation_source_text,
                current_translation=st.session_state.translation_raw,
                user_feedback=user_feedback,
                source_language_code=st.session_state.translation_source_code,
                target_language_code=st.session_state.translation_target_code,
                glossary_instructions=st.session_state.translation_glossary_instructions,
            )
        with st.spinner("Checking terminology..."):
            compliance = translation_service.check_compliance(
                st.session_state.translation_source_text,
                refined,
                st.session_state.translation_source_code,
                st.session_state.translation_target_code,
            )
    except translation_service.TranslationError as exc:
        st.error(str(exc))
        return

    st.session_state.translation_refined = refined
    st.session_state.translation_refined_compliance = compliance


def render() -> None:
    """Render the full Translation view.

    This is the entry point the app's navigation calls for this page.
    """
    role = select_translation_model()

    st.header("Translate your text")

    target_code = _select_target_language()
    combined_text = _gather_input()

    # Advisory cost estimate for large inputs (never blocks translation).
    if combined_text.strip():
        estimated_tokens = translation_service.estimate_tokens(combined_text)
        if estimated_tokens >= translation_service.TOKEN_WARNING_THRESHOLD:
            st.warning(
                f"This text is long (~{estimated_tokens:,} tokens). "
                "Translating it will take longer and cost more."
            )
        else:
            st.caption(f"~{estimated_tokens:,} tokens")

    if st.button("Translate", type="primary", key="translation_translate"):
        if not combined_text.strip():
            st.error("Please upload a file or enter some text to translate.")
        elif not target_code:
            st.error("Please specify a target language (ISO code).")
        else:
            _run_translation(combined_text, target_code, role)

    # --- Base translation output ---
    if st.session_state.translation_raw:
        st.divider()
        st.caption(
            f"Detected source language: {st.session_state.translation_source_name}"
        )
        with st.container(border=True):
            st.write(st.session_state.translation_raw)
        _render_compliance(st.session_state.translation_compliance)

        # --- Upgrade ---
        st.markdown("**Upgrade this translation** ✨")
        st.caption(
            "The upgrade always uses Claude Opus 4.8 (premium), whichever "
            "model is selected on the left."
        )
        feedback = st.text_input(
            "Your feedback or guidelines (optional)", key="translation_feedback"
        )
        if st.button("Upgrade 🚀", type="primary", key="translation_upgrade"):
            _run_upgrade(feedback)

        if st.session_state.translation_refined:
            with st.container(border=True):
                st.write(st.session_state.translation_refined)
            _render_compliance(st.session_state.translation_refined_compliance)

        # --- Download ---
        document_bytes = doc_export.build_translation_docx(
            st.session_state.translation_refined,
            st.session_state.translation_raw,
        )
        st.download_button(
            "Download translation (.docx)",
            data=document_bytes,
            file_name="translation.docx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            key="translation_download",
        )
