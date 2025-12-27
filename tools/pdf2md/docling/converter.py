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
import platform
from pathlib import Path
import logging

# Configure logging to see conversion details
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language mappings for different OCR engines
# Each engine uses different language codes
LANGUAGE_MAP = {
    "ocrmac": {
        "es": ["es-ES"],
        "en": ["en-US"],
        "fr": ["fr-FR"],
        "de": ["de-DE"],
        "pt": ["pt-PT"],
        "it": ["it-IT"],
    },
    "easyocr": {
        "es": ["es"],
        "en": ["en"],
        "fr": ["fr"],
        "de": ["de"],
        "pt": ["pt"],
        "it": ["it"],
    },
    "tesseract": {
        "es": ["spa"],
        "en": ["eng"],
        "fr": ["fra"],
        "de": ["deu"],
        "pt": ["por"],
        "it": ["ita"],
    },
}


def convert_pdf_to_markdown(
    pdf_path: str | Path, 
    return_string: bool = False,
    ocr_engine: str = "ocrmac",
    language: str = "es",
    images_scale: float = 2.0,
    extract_tables: bool = True,
    table_mode: str = "accurate",
    force_ocr: bool = True,
    table_text_handling: str = "hybrid"
):
    """
    Convert PDF to Markdown with configurable OCR and extraction options.
    
    Optimized for Spanish documents by default. Uses force_full_page_ocr to bypass
    potentially corrupted PDF text layers and extract text via OCR instead.
    
    Args:
        pdf_path: Path to the PDF file
        return_string: If True, return markdown as string; if False, save to file and return path
        ocr_engine: OCR engine to use. Options:
                    - 'ocrmac': macOS native OCR (fastest on Mac, default)
                    - 'easyocr': Deep learning based (high accuracy, slower)
                    - 'tesseract': Google's Tesseract OCR (good accuracy, widely supported)
                    - 'auto': Let docling choose best for platform
        language: Language code for OCR. Options: 'es' (Spanish, default), 'en', 'fr', 'de', 'pt', 'it'
                  This helps OCR engines better recognize language-specific characters.
        images_scale: DPI scale for images (default 2.0 = 2x quality, higher = better quality but slower)
        extract_tables: Whether to extract tables with structure (default True)
        table_mode: Table extraction mode - 'fast' or 'accurate' (default 'accurate')
        force_ocr: If True (default), forces full-page OCR to bypass potentially corrupted
                   PDF text layers. Set to False only if you trust the PDF's embedded text.
        table_text_handling: How to handle text inside tables:
                            - 'structure': Keep markdown table formatting (may have encoding issues in cells)
                            - 'ocr': Disable table detection, full OCR (correct text, no table formatting)
                            - 'hybrid': (default) Tables enabled with force_ocr for best balance
    
    Returns:
        str: Markdown content if return_string=True, otherwise path to saved markdown file
    
    Examples:
        # Default: Spanish document with OCR
        >>> md = convert_pdf_to_markdown("document.pdf", return_string=True)
        
        # English document
        >>> md = convert_pdf_to_markdown("document.pdf", language="en")
        
        # Prioritize correct text over table formatting
        >>> md = convert_pdf_to_markdown("document.pdf", table_text_handling="ocr")
        
        # Trust PDF's embedded text (no OCR)
        >>> md = convert_pdf_to_markdown("document.pdf", force_ocr=False)
    """
    # Setup paths
    pdf_path = Path(pdf_path).resolve()
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Extract document name from path
    doc_name = pdf_path.stem
    
    # Create output folders: OUTPUT/docling/doc_name
    output_base = Path.cwd() / "output" / "docling" / doc_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {pdf_path}")
    logger.info(f"Output directory: {output_base}")
    logger.info(f"OCR Engine: {ocr_engine}, Language: {language}, Force OCR: {force_ocr}")
    logger.info(f"Table extraction: {extract_tables} (mode: {table_mode}, handling: {table_text_handling})")

    # Step 1: Configure pipeline with optimized settings
    pipeline_options = PdfPipelineOptions()
    
    # Image quality settings
    pipeline_options.images_scale = images_scale
    pipeline_options.generate_picture_images = False  # No picture extraction
    pipeline_options.generate_table_images = False    # Tables as text, not images
    
    # Table extraction settings based on table_text_handling strategy
    if table_text_handling == "ocr":
        # Disable table structure to let OCR handle all text (including table content)
        # This produces correct text but loses table markdown formatting
        pipeline_options.do_table_structure = False
        logger.info("Table structure disabled - OCR will handle all text including tables")
    elif extract_tables:
        pipeline_options.do_table_structure = True
        # TableFormerMode.ACCURATE is slower but more accurate
        # TableFormerMode.FAST is faster but may miss complex tables
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST
        )
        pipeline_options.table_structure_options.do_cell_matching = True
    else:
        pipeline_options.do_table_structure = False
    
    # OCR engine configuration
    pipeline_options.do_ocr = True
    ocr_options = _get_ocr_options(ocr_engine, language, force_ocr)
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


def _get_ocr_options(ocr_engine: str, language: str = "es", force_ocr: bool = True):
    """
    Configure OCR options based on selected engine, language, and force_ocr setting.
    
    Args:
        ocr_engine: OCR engine name ('ocrmac', 'easyocr', 'tesseract', 'auto')
        language: Language code ('es', 'en', 'fr', 'de', 'pt', 'it')
        force_ocr: If True, enables force_full_page_ocr to bypass PDF text layer
    
    Returns:
        OCR options object configured for the specified engine, or None for auto
    """
    if ocr_engine == "auto":
        # Let docling choose the best engine for the platform
        # Note: auto mode doesn't support custom language or force_ocr settings
        logger.warning("Using 'auto' OCR engine - language and force_ocr settings will be ignored")
        return None
    
    elif ocr_engine == "ocrmac":
        # macOS native OCR - fastest and most efficient on Mac
        # Best choice for macOS systems
        lang_codes = LANGUAGE_MAP["ocrmac"].get(language, [f"{language}-{language.upper()}"])
        ocr_options = OcrMacOptions(
            force_full_page_ocr=force_ocr,
            lang=lang_codes
        )
        logger.debug(f"OcrMac configured with lang={lang_codes}, force_full_page_ocr={force_ocr}")
        return ocr_options
    
    elif ocr_engine == "tesseract":
        # Google's Tesseract OCR - good accuracy, widely supported
        lang_codes = LANGUAGE_MAP["tesseract"].get(language, [language])
        ocr_options = TesseractOcrOptions(
            force_full_page_ocr=force_ocr,
            lang=lang_codes
        )
        logger.debug(f"Tesseract configured with lang={lang_codes}, force_full_page_ocr={force_ocr}")
        return ocr_options
    
    elif ocr_engine == "easyocr":
        # Deep learning based - high accuracy but slower
        lang_codes = LANGUAGE_MAP["easyocr"].get(language, [language])
        ocr_options = EasyOcrOptions(
            force_full_page_ocr=force_ocr,
            lang=lang_codes
        )
        logger.debug(f"EasyOCR configured with lang={lang_codes}, force_full_page_ocr={force_ocr}")
        return ocr_options
    
    else:
        # For unknown engines, use default
        logger.warning(f"Unknown OCR engine '{ocr_engine}', using default options")
        return None


def get_recommended_ocr_engine() -> str:
    """
    Get the recommended OCR engine for the current platform.
    
    Returns:
        str: 'ocrmac' on macOS, 'easyocr' on other platforms
    """
    if platform.system() == "Darwin":
        return "ocrmac"
    else:
        return "easyocr"


if __name__ == "__main__":
    
    print("\n=== Spanish PDF conversion with optimized settings ===")
    result_path = convert_pdf_to_markdown(
        "/Users/inaki/Documents/Personal/course-generator-agent/test.pdf",
        ocr_engine="ocrmac",      # Fast macOS native OCR
        language="es",            # Spanish language
        force_ocr=True,           # Bypass corrupted PDF text layer
        images_scale=2.0,         # Good quality
        table_mode="accurate",    # Accurate table extraction
        table_text_handling="hybrid"  # Balance of tables and correct text
    )
    print(f"Saved markdown to: {result_path}")
