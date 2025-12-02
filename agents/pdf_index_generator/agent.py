from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
from main.state import CourseState, CourseConfig, Module
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from LLMs.text2text import create_text_llm, resolve_text_model_name
from tools.pdf2md.docling.converter import convert_pdf_to_markdown
from .prompts import syllabus_analysis_prompt, retry_prompt
from .utils import compute_layout


# Content-only model for LLM output (no config fields)
class CourseContent(BaseModel):
    """Course content without configuration - used for skeleton generation from PDF"""
    title: str = Field(..., description="Title of the course")
    modules: List[Module] = Field(..., description="Full course structure with all modules")


course_parser = PydanticOutputParser(pydantic_object=CourseContent)


# -------------------------------------------------------
# Chain: Generate course index/state from PDF syllabus
# -------------------------------------------------------
def generate_course_state_from_pdf(
    pdf_path: str,
    total_pages: int,
    language: str = "English",
    max_retries: int = 3,
    words_per_page: int = 400,
    provider: str = "mistral",
) -> CourseState:
    """
    Generate course skeleton from PDF syllabus and return CourseState with embedded config.
    
    Args:
        pdf_path: Path to the PDF syllabus file
        total_pages: Total pages for the course content
        language: Language for the course content
        max_retries: Maximum number of retries for LLM parsing
        words_per_page: Words per page for calculating section word budget
        provider: LLM provider to use
    
    Returns:
        CourseState with course structure extracted from PDF
    """
    # Validate PDF path
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF syllabus not found: {pdf_path}")
    
    print(f"📄 Extracting content from PDF: {pdf_path}")
    
    # Step 1: Extract markdown from PDF
    
    try:
        syllabus_markdown = convert_pdf_to_markdown(pdf_path, return_string=True)
        print(f"✓ Extracted {len(syllabus_markdown)} characters from PDF")
    except Exception as e:
        raise RuntimeError(f"Failed to extract content from PDF: {e}")
   
    
    # Step 2: Compute layout based on total pages
    n_modules, n_submodules, n_sections = compute_layout(total_pages)
    
    # Compute per-section word budget based on total pages
    total_sections = n_modules * n_submodules * n_sections
    total_course_words = total_pages * words_per_page
    n_words = max(1, total_course_words // total_sections)
    
    print(f"📊 Computed layout: {n_modules} modules × {n_submodules} submodules × {n_sections} sections")
    print(f"📝 Target: ~{n_words} words per section")
    
    # Step 3: Create LLM with specified provider
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.1}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Step 4: Create LCEL chain with retry logic
    chain = syllabus_analysis_prompt | llm | StrOutputParser()
    
    # Create fix_parser for fallback
    fix_parser = RetryWithErrorOutputParser.from_llm(
        llm=llm,
        parser=course_parser,
        max_retries=max_retries,
    )
    
    print(f"🤖 Analyzing syllabus with {provider} LLM...")
    
    # Step 5: Invoke the chain
    raw = chain.invoke({
        "syllabus_markdown": syllabus_markdown,
        "language": language,
        "n_modules": n_modules,
        "n_submodules": n_submodules,
        "n_sections": n_sections,
        "n_words": n_words,
        "format_instructions": course_parser.get_format_instructions(),
    })
    
    # Step 6: Try to parse, with fallback to retry parser
    try:
        course_content = course_parser.parse(raw)
    except Exception:
        print("⚠ Initial parse failed, retrying with error correction...")
        course_content = fix_parser.parse_with_prompt(
            completion=raw,
            prompt_value=retry_prompt.format_prompt(
                schema=course_parser.get_format_instructions(),
                language=language
            ),
        )
    
    print(f"✅ Successfully generated course structure from PDF syllabus")
    print(f"   Title: {course_content.title}")
    print(f"   Modules: {len(course_content.modules)}")
    
    # Step 7: Return CourseState with config and generated content
    return CourseState(
        config=CourseConfig(),
        title=course_content.title,
        modules=course_content.modules
    )


if __name__ == "__main__":
    # Ensure you have set: export MISTRAL_API_KEY=your_key
    # Test with a sample PDF syllabus
    pdf_syllabus = "test.pdf"  # Replace with your PDF path
    pages = 50  # total pages desired across the whole course
    
    course_state: CourseState = generate_course_state_from_pdf(
        pdf_path=pdf_syllabus,
        total_pages=pages,
        language="English",
        max_retries=5,
        words_per_page=400,
        provider="mistral",
    )
    
    print("\n" + "="*80)
    print("GENERATED COURSE STATE:")
    print("="*80)
    print(course_state.model_dump_json(indent=2))

