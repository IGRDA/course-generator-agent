import os
from main.state import CourseState
from langchain_mistralai import ChatMistralAI
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from .prompts import gen_prompt, retry_prompt
from .utils import compute_layout

MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")
# Approximate words per page for course content
WORDS_PER_PAGE = 400

llm = ChatMistralAI(
    model=MODEL_NAME,
    temperature=0.5,
)

course_parser = PydanticOutputParser(pydantic_object=CourseState)

# -------------------------------------------------------
# Agennt function: generate with retry-on-parse-failure
# -------------------------------------------------------
def generate_course_state(
    title: str,
    total_pages: int,
    description: str | None = None,
    language: str = "English",
    max_retries: int = 3,
):
    n_modules, n_submodules, n_sections = compute_layout(total_pages)

    # Compute per-section word budget based on total pages
    total_sections = n_modules * n_submodules * n_sections
    total_course_words = total_pages * WORDS_PER_PAGE
    n_words = max(1, total_course_words // total_sections)

    # Create fix_parser with configurable max_retries
    fix_parser = RetryWithErrorOutputParser.from_llm(
        llm=llm,
        parser=course_parser,
        max_retries=max_retries,
    )

    payload = gen_prompt.format(
        course_title=title,
        course_description=description or "",
        language=language,
        n_modules=n_modules,
        n_submodules=n_submodules,
        n_sections=n_sections,
        n_words=n_words,
        format_instructions=course_parser.get_format_instructions(),
    )

    raw = llm.invoke(payload).content

    try:
        return course_parser.parse(raw)
    except Exception:
        fixed = fix_parser.parse_with_prompt(
            completion=raw,
            prompt_value=retry_prompt.format_prompt(
                schema=course_parser.get_format_instructions(),
                language=language
            ),
        )
        return fixed


if __name__ == "__main__":
    # Ensure you have set: export MISTRAL_API_KEY=your_key
    course_title = "Introduction to Modern Data Engineering"
    pages = 100  # total sections desired across the whole course
    desc = "A practical course covering data ingestion, warehousing, orchestration, and observability."

    course_state: CourseState = generate_course_state(
        title=course_title,
        total_pages=pages,
        description=desc,
        max_retries=5,  # Configure the number of retries
    )

    print(course_state.model_dump_json(indent=2))