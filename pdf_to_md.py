"""
Step 1: PDF → Markdown
Uses pymupdf4llm for high-quality PDF-to-Markdown conversion.
Preserves headings, lists, tables, and code blocks.

Install: pip install pymupdf4llm
"""

import sys
import argparse
from pathlib import Path


def pdf_to_markdown(pdf_path: str, output_path: str | None = None) -> Path:
    try:
        import pymupdf4llm
    except ImportError:
        sys.exit("Missing dependency. Run: pip install pymupdf4llm")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        sys.exit(f"File not found: {pdf_path}")

    print(f"Converting: {pdf_path.name}")
    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    else:
        output_path = Path(output_path)

    output_path.write_text(md_text, encoding="utf-8")
    print(f"Saved:      {output_path}  ({len(md_text):,} chars)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown")
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument("-o", "--output", help="Output .md path (default: same name as PDF)")
    args = parser.parse_args()

    pdf_to_markdown(args.pdf, args.output)
