"""Translation view: upload or paste content, translate, upgrade, download.

This module renders the Translation interface and wires together the
input widgets, the translation service, the glossary compliance check,
and the Word export. It holds the UI only; all translation logic lives in
translation_service.

TWO INPUT PATHS
===============
PDF or image upload -> NATIVE path
    The file is sent to the model as-is, so it sees the layout. The model
    drops artefacts (running headers/footers, page numbers), follows the
    correct reading order, and translates text found inside visuals. This
    yields cleaner text than extracting it first. The source language is
    not detected; the glossary is applied as full prompt guidance.

Office file or pasted text -> TEXT path
    Text is extracted (Office) or taken as typed, the source language is
    detected, glossary terms are detected in the source, and the text is
    translated chunk by chunk. Office files often carry footnotes, page
    numbers and other clutter, so the UI suggests converting them to PDF
    for a cleaner result.

Both paths produce plain text and share the same downstream steps:
display, optional upgrade, and Word export. The passive glossary
compliance check runs for the text path; for the native path there is no
extracted source text to compare against, so it naturally no-ops.

All results are kept in session_state so they survive Streamlit reruns.
"""

import streamlit as st

import attachments
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

    Returns:
        The target language ISO code (e.g. "nl"). May be an empty string
        if "Other" is selected but no code has been typed yet.
    """
    options = list(_NAMED_TARGET_LANGUAGES) + ["Other"]
    choice = st.selectbox(
        "Target language", options, index=1, key="translation_target_choice"
    )
    if choice == "Other":
        return st.text_input(
            "Target language ISO code (e.g. 'de')",
            key="translation_target_other",
        ).strip().lower()
    return _NAMED_TARGET_LANGUAGES[choice]


def _extract_office_and_text(uploaded_file, manual_text: str) -> str:
    """Combine extracted Office text and pasted text for the text path.

    Args:
        uploaded_file: The uploaded Office file, or None.
        manual_text: Text typed by the user.

    Returns:
        The combined source text, or an empty string if nothing usable.
    """
    file_text = ""
    if uploaded_file is not None:
        try:
            file_text = file_readers.extract_text(uploaded_file)
        except (file_readers.FileTooLargeError, file_readers.UnsupportedFileError) as exc:
            st.error(str(exc))
    parts = [part for part in (file_text, manual_text) if part and part.strip()]
    return "\n".join(parts)


def _show_estimate(estimated_tokens: int) -> None:
    """Show an advisory token estimate, warning when the input is large.

    Args:
        estimated_tokens: The estimated token count for the input.
    """
    if estimated_tokens >= translation_service.TOKEN_WARNING_THRESHOLD:
        st.warning(
            f"This is large (~{estimated_tokens:,} tokens). Translating it "
            "will take longer and cost more."
        )
    else:
        st.caption(f"~{estimated_tokens:,} tokens")


def _render_compliance(discrepancies: list[dict]) -> None:
    """Show a glossary-compliance warning, if there are any discrepancies.

    Args:
        discrepancies: The list returned by
            translation_service.check_compliance(). An empty list renders
            nothing.
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


def _store_result(result, source_text: str, target_code: str, compliance: list) -> None:
    """Write a fresh translation result into session_state.

    Clears any previous refined translation so the view never shows a
    stale upgrade.

    Args:
        result: The TranslationResult from the service.
        source_text: The plain-text source ("" for native documents).
        target_code: Target language ISO code used.
        compliance: The compliance discrepancies for the base translation.
    """
    st.session_state.translation_raw = result.text
    st.session_state.translation_glossary_instructions = result.glossary_instructions
    st.session_state.translation_source_text = source_text
    st.session_state.translation_source_code = result.source_language_code
    st.session_state.translation_source_name = result.source_language_name
    st.session_state.translation_target_code = target_code
    st.session_state.translation_compliance = compliance
    st.session_state.translation_refined = ""
    st.session_state.translation_refined_compliance = []


def _run_text_translation(combined_text: str, target_code: str, role: str) -> None:
    """Translate pasted/Office text and store the results.

    Args:
        combined_text: The source text to translate.
        target_code: Target language ISO code.
        role: The model role to translate with.
    """
    source_code, _name = language.detect_language(combined_text)
    if not source_code:
        st.error(
            "The source language could not be detected. Please add more text."
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

    _store_result(result, combined_text, target_code, compliance)


def _run_native_translation(uploaded_file, target_code: str, role: str) -> None:
    """Translate a PDF/image document natively and store the results.

    Args:
        uploaded_file: The uploaded PDF or image.
        target_code: Target language ISO code.
        role: The model role to translate with.
    """
    try:
        with st.spinner("Reading and translating the document..."):
            result = translation_service.translate_document(
                [uploaded_file], target_code, role
            )
    except translation_service.TranslationError as exc:
        st.error(str(exc))
        return

    # No plain-text source for native documents, so the compliance check
    # (which derives expected terms from the source) does not apply.
    _store_result(result, "", target_code, [])


def _run_upgrade(user_feedback: str) -> None:
    """Upgrade the stored base translation and store the refined result.

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
        # Compliance only applies when there is a plain-text source.
        compliance = []
        if st.session_state.translation_source_text:
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


def _render_results() -> None:
    """Render the base translation, compliance, upgrade and download.

    Reads everything from session_state, so it shows the latest stored
    results across reruns. Renders nothing until a translation exists.
    """
    if not st.session_state.translation_raw:
        return

    st.divider()
    st.caption(f"Source: {st.session_state.translation_source_name}")
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


def render() -> None:
    """Render the full Translation view.

    This is the entry point the app's navigation calls for this page.
    """
    role = select_translation_model()

    st.header("Translate your text")

    target_code = _select_target_language()

    uploaded_file = st.file_uploader(
        "Upload a file (PDF, image, or Office)",
        type=attachments.ALLOWED_EXTENSIONS,
        key="translation_file",
    )
    manual_text = st.text_area(
        "Or paste text to translate", height=150, key="translation_manual"
    )

    # Decide which path the input takes.
    is_native = (
        uploaded_file is not None
        and attachments.is_native_type(uploaded_file.type)
    )
    is_office = uploaded_file is not None and not is_native

    if is_office:
        st.info(
            "For a cleaner result — especially with images, footnotes or rich "
            "layout — export this document to PDF and upload the PDF instead."
        )
    if is_native and manual_text.strip():
        st.caption("The pasted text is ignored when a PDF or image is uploaded.")

    # Advisory cost estimate, per path.
    if is_native:
        _show_estimate(translation_service.estimate_document_tokens([uploaded_file]))
    else:
        combined_text = _extract_office_and_text(uploaded_file, manual_text)
        if combined_text.strip():
            _show_estimate(translation_service.estimate_tokens(combined_text))

    if st.button("Translate", type="primary", key="translation_translate"):
        if not target_code:
            st.error("Please specify a target language (ISO code).")
        elif is_native:
            _run_native_translation(uploaded_file, target_code, role)
        else:
            combined_text = _extract_office_and_text(uploaded_file, manual_text)
            if not combined_text.strip():
                st.error("Please upload a file or paste some text to translate.")
            else:
                _run_text_translation(combined_text, target_code, role)

    _render_results()
