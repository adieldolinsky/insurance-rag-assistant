import os
from PyPDF2 import PdfReader, PdfWriter

# השורה הזו חובה לפני כל ייבוא אחר כדי למנוע את השגיאה בווינדוס
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from docling.document_converter import DocumentConverter


def process_large_pdf(input_pdf_path, output_md_path, pages_per_chunk=10):
    """
    Splits a large PDF file into chunks, converts each chunk to Markdown using Docling,
    and merges everything into a single complete file.
    """
    if not os.path.exists(input_pdf_path):
        print(f"❌ The file {input_pdf_path} was not found.")
        return

    print(f"Starting processing of {input_pdf_path}...")

    # Initialize the Docling document converter
    converter = DocumentConverter()

    # Read the original PDF
    reader = PdfReader(input_pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages in document: {total_pages}")

    final_markdown = ""

    # Loop through pages in steps of pages_per_chunk
    for start_page in range(0, total_pages, pages_per_chunk):
        end_page = min(start_page + pages_per_chunk, total_pages)
        print(f"Processing pages {start_page + 1} to {end_page}...")

        # Create a temporary PDF file for the current chunk
        writer = PdfWriter()
        for i in range(start_page, end_page):
            writer.add_page(reader.pages[i])

        temp_pdf_path = f"temp_chunk_{start_page}_{end_page}.pdf"
        with open(temp_pdf_path, "wb") as f:
            writer.write(f)

        try:
            # Run Docling on the temporary file
            result = converter.convert(temp_pdf_path)
            markdown_content = result.document.export_to_markdown()

            # Concatenate the result to the final text
            final_markdown += markdown_content + "\n\n"

        except Exception as e:
            print(f"❌ Error processing pages {start_page + 1}-{end_page}: {e}")

        finally:
            # Delete the temporary file to avoid cluttering the disk
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    # Save the final result to the Markdown file
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    print(f"\n✅ Processing complete! The full file was saved as: {output_md_path}")


if __name__ == "__main__":
    # Run the function on the Maccabi policy
    process_large_pdf(
        input_pdf_path="maccabi_sheli_regulations.pdf",
        output_md_path="maccabi_sheli_regulations.md",
        pages_per_chunk=10
    )
