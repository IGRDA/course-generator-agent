import os
from typing import List
from pydantic import BaseModel, Field
from main.state import CourseState, CourseConfig, Module
from langchain_mistralai import ChatMistralAI
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from .prompts import gen_prompt, retry_prompt
from .utils import compute_layout

MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")

# Content-only model for LLM output (no config fields)
class CourseContent(BaseModel):
    """Course content without configuration - used for skeleton generation"""
    title: str = Field(..., description="Title of the course")
    modules: List[Module] = Field(..., description="Full course structure with all modules")

llm = ChatMistralAI(
    model=MODEL_NAME,
    temperature=0.5,
)

course_parser = PydanticOutputParser(pydantic_object=CourseContent)

# -------------------------------------------------------
# LCEL Chain: generate with retry-on-parse-failure
# -------------------------------------------------------
@traceable(name="generate_course_state")
def generate_course_state(
    title: str,
    total_pages: int,
    description: str | None = None,
    language: str = "English",
    max_retries: int = 3,
    words_per_page: int = 400,
) -> CourseState:
    """Generate course skeleton and return CourseState with embedded config"""
    n_modules, n_submodules, n_sections = compute_layout(total_pages)

    # Compute per-section word budget based on total pages
    total_sections = n_modules * n_submodules * n_sections
    total_course_words = total_pages * words_per_page
    n_words = max(1, total_course_words // total_sections)

    # Create LCEL chain with retry logic
    # First try with standard parser
    chain = gen_prompt | llm | StrOutputParser()
    
    # Create fix_parser for fallback
    fix_parser = RetryWithErrorOutputParser.from_llm(
        llm=llm,
        parser=course_parser,
        max_retries=max_retries,
    )

    # Invoke the chain
    raw = chain.invoke({
        "course_title": title,
        "course_description": description or "",
        "language": language,
        "n_modules": n_modules,
        "n_submodules": n_submodules,
        "n_sections": n_sections,
        "n_words": n_words,
        "format_instructions": course_parser.get_format_instructions(),
    })

    # Try to parse, with fallback to retry parser
    try:
        course_content = course_parser.parse(raw)
    except Exception:
        course_content = fix_parser.parse_with_prompt(
            completion=raw,
            prompt_value=retry_prompt.format_prompt(
                schema=course_parser.get_format_instructions(),
                language=language
            ),
        )

    
    # Return CourseState with config and generated content
    return CourseState(
        config= CourseConfig(),
        title=course_content.title,
        modules=course_content.modules
    )


if __name__ == "__main__":
    # Ensure you have set: export MISTRAL_API_KEY=your_key
    course_title = "Introduction to Modern Data Engineering"
    pages = 100  # total pages desired across the whole course
    desc = "A practical course covering data ingestion, warehousing, orchestration, and observability."

    course_state: CourseState = generate_course_state(
        title=course_title,
        total_pages=pages,
        description=desc,
        max_retries=5,
        words_per_page=400,
    )

    print(course_state.model_dump_json(indent=2))