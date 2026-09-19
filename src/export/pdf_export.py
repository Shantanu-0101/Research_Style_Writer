from fpdf import FPDF


def _sanitize(text: str) -> str:
    """
    fpdf2's built-in fonts (Helvetica) only support latin-1 characters.
    LLM output often includes curly quotes, em-dashes, etc. that aren't
    in that range — replace them rather than crash. A proper fix later
    is bundling a Unicode TTF font (e.g. DejaVu Sans) via pdf.add_font(),
    but this keeps things simple for now.
    """
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf(title: str, sections: list[tuple[str, str]]) -> bytes:
    """sections: list of (section_name, text) tuples, in order."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, _sanitize(title))
    pdf.ln(4)

    for section_name, text in sections:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, _sanitize(section_name))
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _sanitize(text))
        pdf.ln(6)

    return bytes(pdf.output())