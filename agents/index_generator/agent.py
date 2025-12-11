from typing import List
from pydantic import BaseModel, Field
from main.state import CourseState, CourseConfig, Module, Submodule, Section
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from LLMs.text2text import create_text_llm, resolve_text_model_name
from tools.imagesearch import create_image_search  # Import for testing
from .prompts import gen_prompt, retry_prompt
from .utils import compute_layout

# -------------------------------------------------------
# Simplified Skeleton Models (for LLM output parsing)
# -------------------------------------------------------
class SkeletonSection(BaseModel):
    """Lightweight section model with only structural fields"""
    title: str = Field(..., description="Title of the section")
    index: int = Field(default=0, description="Section index within submodule")
    description: str = Field(default="", description="Description of the section")

class SkeletonSubmodule(BaseModel):
    """Lightweight submodule model with only structural fields"""
    title: str = Field(..., description="Title of the submodule")
    index: int = Field(default=0, description="Submodule index within module")
    description: str = Field(default="", description="Description of the submodule")
    sections: List[SkeletonSection] = Field(..., description="List of sections in this submodule")

class SkeletonModule(BaseModel):
    """Lightweight module model with only structural fields"""
    title: str = Field(..., description="Title of the module")
    index: int = Field(default=0, description="Module index in course")
    description: str = Field(default="", description="Description of the module")
    submodules: List[SkeletonSubmodule] = Field(..., description="List of submodules in this module")

class SkeletonCourse(BaseModel):
    """Lightweight course model for skeleton generation"""
    title: str = Field(..., description="Title of the course")
    modules: List[SkeletonModule] = Field(..., description="Course structure with all modules")

# Parser for skeleton course (lightweight structure)
skeleton_parser = PydanticOutputParser(pydantic_object=SkeletonCourse)

# -------------------------------------------------------
# Conversion function: Skeleton -> Full CourseState
# -------------------------------------------------------
def convert_skeleton_to_course_state(skeleton: SkeletonCourse) -> CourseState:
    """Convert lightweight skeleton models to full CourseState with empty content fields"""
    
    full_modules = []
    for skel_module in skeleton.modules:
        full_submodules = []
        for skel_submodule in skel_module.submodules:
            full_sections = []
            for skel_section in skel_submodule.sections:
                # Create full Section with empty content fields
                full_section = Section(
                    title=skel_section.title,
                    index=skel_section.index,
                    description=skel_section.description,
                    theory="",  # Empty, to be filled by section_theory_generator
                    html=None,  # None, to be filled by html_formatter
                    meta_elements=None,  # None, to be filled later
                    activities=None,  # None, to be filled by activities_generator
                )
                full_sections.append(full_section)
            
            # Create full Submodule
            full_submodule = Submodule(
                title=skel_submodule.title,
                index=skel_submodule.index,
                description=skel_submodule.description,
                duration=0.0,  # Will be calculated later in workflow
                sections=full_sections,
            )
            full_submodules.append(full_submodule)
        
        # Create full Module
        full_module = Module(
            title=skel_module.title,
            id="",  # Will be set later in workflow
            index=skel_module.index,
            description=skel_module.description,
            duration=0.0,  # Will be calculated later in workflow
            type="module",
            submodules=full_submodules,
        )
        full_modules.append(full_module)
    
    # Return CourseState with empty config (will be merged in workflow)
    return CourseState(
        config=CourseConfig(),
        title=skeleton.title,
        modules=full_modules
    )

# -------------------------------------------------------
# Chain: Generate course index/state with retry-on-parse-failure
# -------------------------------------------------------
def generate_course_state(
    title: str,
    total_pages: int,
    description: str | None = None,
    language: str = "English",
    max_retries: int = 3,
    words_per_page: int = 400,
    provider: str = "mistral",
) -> CourseState:
    """Generate course skeleton and return CourseState with embedded config"""
    n_modules, n_submodules, n_sections = compute_layout(total_pages)

    # Compute per-section word budget based on total pages
    total_sections = n_modules * n_submodules * n_sections
    total_course_words = total_pages * words_per_page
    n_words = max(1, total_course_words // total_sections)

    # Create LLM with specified provider
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)

    # Create LCEL chain with retry logic
    # First try with standard parser
    chain = gen_prompt | llm | StrOutputParser()
    
    # Create fix_parser for fallback (using skeleton parser)
    fix_parser = RetryWithErrorOutputParser.from_llm(
        llm=llm,
        parser=skeleton_parser,
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
        "format_instructions": skeleton_parser.get_format_instructions(),
    })

    # Try to parse skeleton, with fallback to retry parser
    try:
        skeleton_course = skeleton_parser.parse(raw)
    except Exception:
        skeleton_course = fix_parser.parse_with_prompt(
            completion=raw,
            prompt_value=retry_prompt.format_prompt(
                schema=skeleton_parser.get_format_instructions(),
                language=language
            ),
        )

    # Convert skeleton to full CourseState
    course_state = convert_skeleton_to_course_state(skeleton_course)
    
    # Return CourseState with generated content
    return course_state


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
        provider="mistral",
    )

    print(course_state.model_dump_json(indent=2))