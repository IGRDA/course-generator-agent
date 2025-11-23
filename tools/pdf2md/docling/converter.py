from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    EasyOcrOptions,
    TesseractOcrOptions,
    OcrMacOptions,
)
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
import os
from pathlib import Path
import logging

# Configure logging to see conversion details
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_pdf_to_markdown(
    pdf_path: str | Path, 
    return_string: bool = False,
    ocr_engine: str = "auto",
    images_scale: float = 2.0,
    extract_tables: bool = True,
    table_mode: str = "accurate"
):
    """
    Convert PDF to Markdown with configurable OCR and extraction options.
    
    Args:
        pdf_path: Path to the PDF file
        return_string: If True, return markdown as string; if False, save to file and return path
        ocr_engine: OCR engine to use. Options: 'auto', 'ocrmac', 'rapidocr', 'tesseract', 'easyocr'
                    Default 'auto' selects best for your platform (ocrmac on macOS)
        images_scale: DPI scale for images (default 2.0 = 2x quality, higher = better quality but slower)
        extract_tables: Whether to extract tables with structure (default True)
        table_mode: Table extraction mode - 'fast' or 'accurate' (default 'accurate')
    
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
    output_base = Path.cwd() / "OUTPUT" / "docling" / doc_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {pdf_path}")
    logger.info(f"Output directory: {output_base}")
    logger.info(f"OCR Engine: {ocr_engine}")
    logger.info(f"Table extraction: {extract_tables} (mode: {table_mode})")

    # Step 1: Configure pipeline with optimized settings
    pipeline_options = PdfPipelineOptions()
    
    # Image quality settings
    pipeline_options.images_scale = images_scale
    pipeline_options.generate_picture_images = False  # No picture extraction
    pipeline_options.generate_table_images = False    # Tables as text, not images
    
    # Table extraction settings
    if extract_tables:
        pipeline_options.do_table_structure = True
        # TableFormerMode.ACCURATE is slower but more accurate
        # TableFormerMode.FAST is faster but may miss complex tables
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST
        )
    else:
        pipeline_options.do_table_structure = False
    
    # OCR engine configuration
    pipeline_options.do_ocr = True
    ocr_options = _get_ocr_options(ocr_engine)
    if ocr_options is not None:
        pipeline_options.ocr_options = ocr_options
    
    # Step 2: Convert document with optimized settings
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    try:
        import time
        start_time = time.time()
        result = converter.convert(pdf_path)
        elapsed = time.time() - start_time
        logger.info(f"Conversion completed in {elapsed:.2f} seconds")
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        raise

    # Step 3: Get markdown content
    if return_string:
        # Export markdown to string
        markdown_content = result.document.export_to_markdown()
        logger.info(f"✓ Markdown extracted ({len(markdown_content)} characters)")
        return markdown_content
    else:
        # Step 4: Export markdown to file
        markdown_path = output_base / f"{doc_name}.md"
        result.document.save_as_markdown(
            markdown_path,
            image_mode=ImageRefMode.EMBEDDED
        )
        
        logger.info(f"✓ Markdown saved to: {markdown_path}")
        return str(markdown_path)


def _get_ocr_options(ocr_engine: str):
    """
    Configure OCR options based on selected engine.
    
    Args:
        ocr_engine: OCR engine name
    
    Returns:
        OCR options object or None for auto
    """
    if ocr_engine == "auto":
        # Let docling choose the best engine for the platform
        return None
    
    elif ocr_engine == "ocrmac":
        # macOS native OCR - fastest and most efficient on Mac
        # Best choice for macOS systems
        ocr_options = OcrMacOptions()
        return ocr_options
    
    elif ocr_engine == "tesseract":
        # Google's Tesseract OCR - good accuracy, widely supported
        ocr_options = TesseractOcrOptions()
        return ocr_options
    
    elif ocr_engine == "easyocr":
        # Deep learning based - high accuracy but slower
        ocr_options = EasyOcrOptions()
        return ocr_options
    
    else:
        # For rapidocr and others, use default
        logger.warning(f"Using default options for OCR engine: {ocr_engine}")
        return None

if __name__ == "__main__":
    
    print("\n=== Example 4: High quality mode ===")
    result_path = convert_pdf_to_markdown(pdf_file,
    ocr_engine="easyocr",  # Deep learning OCR (slower but more accurate)
    images_scale=3.0,       # 3x DPI for highest quality
    table_mode="accurate"   # Most accurate table extraction
    )
    print(f"Saved markdown to: {result_path}")
