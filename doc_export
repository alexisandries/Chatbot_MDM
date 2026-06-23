"""Word document export for translations.

Builds a downloadable .docx file containing a translation. The document
holds up to two sections: the refined (upgraded) translation first, then
the base translation without refinement. Each section has its own
heading, so the user can compare both versions in one file.

The single public function returns the document as raw bytes, ready to be
handed to Streamlit's st.download_button. Nothing is written to disk.

TEXT FORMATTING
===============
Translations are plain text in which:
- a blank line (two consecutive newlines) separates paragraphs, and
- a single newline is a line break within a paragraph.
The builder reproduces both, so the Word document keeps the visual
structure the model produced.
"""

from io import BytesIO

from docx import Document


# Headings shown above each section of the exported document.
_REFINED_HEADING = "Refined translation"
_BASE_HEADING = "Translation (without refinement)"

# Placeholder text used when a section has no content.
_NO_REFINED_TEXT = "No refined translation was produced."
_NO_BASE_TEXT = "No base translation is available."


def _add_text_section(document: Document, heading: str, text: str, placeholder: str) -> None:
    """Add a titled section of text to a Word document.

    Writes the heading, then the text split into paragraphs and line
    breaks. If the text is empty, a placeholder sentence is written
    instead so the section is never blank.

    Args:
        document: The python-docx Document being built.
        heading: The section title.
        text: The section body. May be empty or None.
        placeholder: Sentence to insert when text is empty.
    """
    document.add_heading(heading, level=1)

    if not text or not text.strip():
        document.add_paragraph(placeholder)
        return

    # Split into paragraphs on blank lines; within a paragraph, keep
    # single newlines as soft line breaks.
    paragraphs = text.split("\n\n")
    for paragraph_text in paragraphs:
        if not paragraph_text.strip():
            continue
        paragraph = document.add_paragraph()
        lines = paragraph_text.split("\n")
        for index, line in enumerate(lines):
            run = paragraph.add_run(line)
            if index < len(lines) - 1:
                run.add_break()  # soft line break inside the paragraph


def build_translation_docx(refined_text: str, base_text: str) -> bytes:
    """Build a Word document containing a translation, as bytes.

    Produces a .docx with two sections: the refined translation followed
    by the base translation. Either section may be empty, in which case a
    placeholder sentence is shown for it; the document is always valid and
    downloadable.

    Args:
        refined_text: The upgraded/refined translation. May be empty if
            the user has not run an upgrade.
        base_text: The base translation produced before any refinement.
            May be empty in the unusual case where only a refined version
            exists.

    Returns:
        The complete .docx file as a bytes object, suitable for
        st.download_button(data=...).
    """
    document = Document()

    _add_text_section(document, _REFINED_HEADING, refined_text, _NO_REFINED_TEXT)
    _add_text_section(document, _BASE_HEADING, base_text, _NO_BASE_TEXT)

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
