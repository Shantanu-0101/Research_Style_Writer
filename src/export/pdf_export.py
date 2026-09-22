from fpdf import FPDF
import os

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def generate_pdf(title: str, sections: list[tuple[str, str]]) -> bytes:
    pdf = FPDF()
    pdf.add_font("DejaVu", "", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"))
    pdf.add_page()

    pdf.set_font("DejaVu", "B", 16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(4)

    for section_name, text in sections:
        pdf.set_font("DejaVu", "B", 13)
        pdf.multi_cell(0, 8, section_name)
        pdf.ln(2)
        pdf.set_font("DejaVu", "", 11)
        pdf.multi_cell(0, 6, text)
        pdf.ln(6)

    return bytes(pdf.output())