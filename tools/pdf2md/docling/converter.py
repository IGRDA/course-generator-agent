from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
import os
from pathlib import Path

def convert_pdf_to_markdown(pdf_path: str | Path, return_string: bool = False):
    """
    Convert PDF to Markdown with optional image extraction.
    
    Args:
        pdf_path: Path to the PDF file
        return_string: If True, return markdown as string; if False, save to file and return path
    
    Returns:
        str: Markdown content if return_string=True, otherwise path to saved markdown file
    """
    # Setup paths
    pdf_path = Path(pdf_path).resolve()
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Extract document name from path
    doc_name = pdf_path.stem
    
    # Create output folders: OUTPUT/docling/doc_name
    # Using project root or CWD for output
    # Assuming the script is run from project root
    output_base = Path.cwd() / "OUTPUT" / "docling" / doc_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {pdf_path}")
    print(f"Output directory: {output_base}")

    # Step 1: Configure pipeline - no image extraction, tables as text
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0  # Higher quality (2x DPI)
    pipeline_options.generate_picture_images = False  # No pictures/figures extraction
    pipeline_options.generate_table_images = False  # Tables as text, not images
    
    # Step 2: Convert document with image extraction
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    try:
        result = converter.convert(pdf_path)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        raise

    # Step 3: Get markdown content
    # If return_string=True, return as string; otherwise save to file
    if return_string:
        # Export markdown to string
        markdown_content = result.document.export_to_markdown()
        print(f"✓ Markdown extracted from PDF (text only, no images)")
        return markdown_content
    else:
        # Step 4: Export markdown to file
        markdown_path = output_base / f"{doc_name}.md"
        result.document.save_as_markdown(
            markdown_path,
            image_mode=ImageRefMode.EMBEDDED  # Embed or skip images
        )
        
        print(f"✓ Markdown saved to: {markdown_path}")
        return str(markdown_path)

if __name__ == "__main__":    
    # Default test file if none provided
    pdf_file = "test.pdf"
    
    # Test file mode (saves to disk)
    result_path = convert_pdf_to_markdown(pdf_file, return_string=False)
    print(f"Saved markdown to: {result_path}")
    
    # Test string mode (returns content)
    # markdown_content = convert_pdf_to_markdown(pdf_file, return_string=True)
    # print(f"Extracted {len(markdown_content)} characters")
