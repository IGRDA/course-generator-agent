from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
import os
from pathlib import Path

def convert_pdf_to_markdown(pdf_path: str | Path):
    """
    Convert PDF to Markdown with images extracted to a separate folder.
    
    Args:
        pdf_path: Path to the PDF file
    """
    # Setup paths
    pdf_path = Path(pdf_path).resolve()
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Extract document name from path
    doc_name = pdf_path.stem
    
    # Create output folders: OUTPUT/docling/doc_name/images
    # Using project root or CWD for output
    # Assuming the script is run from project root
    output_base = Path.cwd() / "OUTPUT" / "docling" / doc_name
    images_dir = output_base / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {pdf_path}")
    print(f"Output directory: {output_base}")

    # Step 1: Configure pipeline to extract images
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0  # Higher quality (2x DPI)
    pipeline_options.generate_picture_images = False  # Extract pictures/figures
    pipeline_options.generate_table_images = True  # Extract tables as images
    
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

    # Step 3: Save images to separate folder
    picture_counter = 0
    table_counter = 0
    
    # Iterate through the document to find pictures and tables
    # Note: iterate_items() might need adjustment based on exact docling version structure
    # The user code used result.document.iterate_items()
    
    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem) and pipeline_options.generate_picture_images:
            picture_counter += 1
            image_filename = f"{doc_name}_picture_{picture_counter}.png"
            image_path = images_dir / image_filename
            
            try:
                image = element.get_image(result.document)
                if image:
                    with image_path.open("wb") as fp:
                        image.save(fp, "PNG")
            except Exception as e:
                print(f"Warning: Failed to save picture {picture_counter}: {e}")
        
        elif isinstance(element, TableItem) and pipeline_options.generate_table_images:
            table_counter += 1
            image_filename = f"{doc_name}_table_{table_counter}.png"
            image_path = images_dir / image_filename
            
            try:
                image = element.get_image(result.document)
                if image:
                    with image_path.open("wb") as fp:
                        image.save(fp, "PNG")
            except Exception as e:
                print(f"Warning: Failed to save table {table_counter}: {e}")
    
    # Step 4: Export markdown with image references
    markdown_path = output_base / f"{doc_name}.md"
    result.document.save_as_markdown(
        markdown_path,
        image_mode=ImageRefMode.REFERENCED  # Links to saved images
    )
    
    print(f"✓ Markdown saved to: {markdown_path}")
    print(f"✓ Images saved to: {images_dir}")
    print(f"✓ Extracted {picture_counter} pictures and {table_counter} tables")
    
    return markdown_path

if __name__ == "__main__":    
    # Default test file if none provided
    pdf_file =  "test.pdf"
    convert_pdf_to_markdown(pdf_file)
