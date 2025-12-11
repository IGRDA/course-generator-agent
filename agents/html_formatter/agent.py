from typing import Annotated, List
from operator import add
from pydantic import BaseModel, Field
from main.state import CourseState, HtmlElement
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from LLMs.text2text import create_text_llm, resolve_text_model_name
from .prompts import html_generation_prompt, correction_prompt


# ---- Icon Mapping ----
ICON_MAPPING = {
    "history": "mdi-book-clock",
    "historia": "mdi-book-clock",
    "science": "mdi-flask",
    "ciencia": "mdi-flask",
    "technology": "mdi-laptop",
    "tecnología": "mdi-laptop",
    "math": "mdi-calculator",
    "matemática": "mdi-calculator",
    "art": "mdi-palette",
    "arte": "mdi-palette",
    "business": "mdi-briefcase",
    "negocio": "mdi-briefcase",
    "health": "mdi-heart-pulse",
    "salud": "mdi-heart-pulse",
    "education": "mdi-school",
    "educación": "mdi-school",
    "politics": "mdi-account-group",
    "política": "mdi-account-group",
    "military": "mdi-shield-sword",
    "militar": "mdi-shield-sword",
    "war": "mdi-sword-cross",
    "guerra": "mdi-sword-cross",
    "law": "mdi-gavel",
    "derecho": "mdi-gavel",
    "philosophy": "mdi-head-lightbulb",
    "filosofía": "mdi-head-lightbulb",
    "default": "mdi-information"
}


def select_icon(section_title: str) -> str:
    """
    Select an appropriate Material Design Icon based on section title keywords.
    
    Args:
        section_title: Title of the section
        
    Returns:
        Material Design Icon class name
    """
    title_lower = section_title.lower()
    
    # Check for keyword matches
    for keyword, icon in ICON_MAPPING.items():
        if keyword in title_lower:
            return icon
    
    # Default icon if no match
    return ICON_MAPPING["default"]


# ---- State for individual section task ----
class SectionHtmlTask(BaseModel):
    """State for processing a single section's HTML structure"""
    course_state: CourseState
    module_idx: int
    submodule_idx: int
    section_idx: int
    section_title: str
    theory: str
    include_quotes: bool
    include_tables: bool


# ---- State for aggregating results ----
class HtmlFormattingState(BaseModel):
    """State for the HTML formatting subgraph"""
    course_state: CourseState
    completed_html: Annotated[list[dict], add] = Field(default_factory=list)
    total_sections: int = 0


def plan_html_formatting(state: HtmlFormattingState) -> dict:
    """Update the total_sections count in the state."""
    section_count = 0
    
    for module in state.course_state.modules:
        for submodule in module.submodules:
            section_count += len(submodule.sections)
    
    print(f"📋 Planning {section_count} sections for HTML formatting")
    
    return {"total_sections": section_count}


def continue_to_html(state: HtmlFormattingState) -> list[Send]:
    """Fan-out: Create a Send for each section to process in parallel."""
    sends = []
    
    include_quotes = state.course_state.config.include_quotes_in_html
    include_tables = state.course_state.config.include_tables_in_html
    
    for m_idx, module in enumerate(state.course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                # Create a task for each section
                task = SectionHtmlTask(
                    course_state=state.course_state,
                    module_idx=m_idx,
                    submodule_idx=sm_idx,
                    section_idx=s_idx,
                    section_title=section.title,
                    theory=section.theory,
                    include_quotes=include_quotes,
                    include_tables=include_tables
                )
                # Send to the generate_section_html node
                sends.append(Send("generate_section_html", task))
    
    return sends


def generate_section_html(state: SectionHtmlTask) -> dict:
    """
    Generate HTML structure for a single section as a direct array.
    Uses structured output with RetryWithErrorOutputParser for validation.
    """
    # Extract context
    provider = state.course_state.config.text_llm_provider
    max_retries = state.course_state.config.max_retries
    
    # Create LLM
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.0}
    if model_name:
        llm_kwargs["model_name"] = "mistral-medium-latest"
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Create parser for List[HtmlElement]
    from pydantic import create_model
    HtmlElementList = create_model('HtmlElementList', elements=(List[HtmlElement], ...))
    parser = PydanticOutputParser(pydantic_object=HtmlElementList)
    
    # Create fix parser for retry
    fix_parser = RetryWithErrorOutputParser.from_llm(
        llm=llm,
        parser=parser,
        max_retries=max_retries,
    )
    
    # Generate using LCEL chain
    chain = html_generation_prompt | llm | StrOutputParser()
    
    # Build optional instructions
    quote_instruction = ""
    if state.include_quotes:
        quote_instruction = """- Optionally include quote elements with author and quote text
  Format: {{"type": "quote", "content": {{"author": "...", "quote": "..."}}}}"""
    
    table_instruction = ""
    if state.include_tables:
        table_instruction = """- Optionally include table elements with title, headers, and rows
  Format: {{"type": "table", "content": {{"title": "...", "headers": [...], "rows": [[...], [...]]}}}}"""
    
    # Select icon for this section
    suggested_icon = select_icon(state.section_title)
    
    raw = chain.invoke({
        "theory": state.theory,
        "section_title": state.section_title,
        "language": state.course_state.language,
        "quote_instruction": quote_instruction,
        "table_instruction": table_instruction,
        "suggested_icon": suggested_icon,
        "format_instructions": parser.get_format_instructions(),
    })
    
    # Try to parse, with fallback to retry parser
    try:
        result = parser.parse(raw)
        html_array = result.elements
    except Exception as e:
        print(f"  ⚠ Initial parse failed for section {state.section_idx}, attempting correction...")
        result = fix_parser.parse_with_prompt(
            completion=raw,
            prompt_value=correction_prompt.format_prompt(
                error=str(e),
                completion=raw,
                format_instructions=parser.get_format_instructions(),
            ),
        )
        html_array = result.elements
    
    print(f"✓ Generated HTML for Module {state.module_idx+1}, "
          f"Submodule {state.submodule_idx+1}, Section {state.section_idx+1}")
    
    # Return the completed section info with direct array
    return {
        "completed_html": [{
            "module_idx": state.module_idx,
            "submodule_idx": state.submodule_idx,
            "section_idx": state.section_idx,
            "html_structure": html_array
        }]
    }


def reduce_html(state: HtmlFormattingState) -> dict:
    """Fan-in: Aggregate all generated HTML structures back into the course state."""
    print(f"📦 Reducing {len(state.completed_html)} completed HTML structures")
    
    # Update course state with all generated HTML structures
    for html_info in state.completed_html:
        m_idx = html_info["module_idx"]
        sm_idx = html_info["submodule_idx"]
        s_idx = html_info["section_idx"]
        
        section = state.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx]
        section.html = html_info["html_structure"]
    
    print(f"✅ All {state.total_sections} section HTML structures generated successfully!")
    
    return {"course_state": state.course_state}


def build_html_generation_graph(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Build the HTML generation subgraph using Send for dynamic parallelization.
    
    Args:
        max_retries: Number of retries for each section generation
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
    """
    graph = StateGraph(HtmlFormattingState)
    
    # Configure retry policy with exponential backoff
    retry_policy = RetryPolicy(
        max_attempts=max_retries,
        initial_interval=initial_delay,
        backoff_factor=backoff_factor,
        max_interval=60.0
    )
    
    # Add nodes
    graph.add_node("plan_html_formatting", plan_html_formatting)
    graph.add_node("generate_section_html", generate_section_html, retry=retry_policy)
    graph.add_node("reduce_html", reduce_html)
    
    # Add edges
    graph.add_edge(START, "plan_html_formatting")
    graph.add_conditional_edges("plan_html_formatting", continue_to_html, ["generate_section_html"])
    graph.add_edge("generate_section_html", "reduce_html")
    graph.add_edge("reduce_html", END)
    
    return graph.compile()


def generate_all_section_html(
    course_state: CourseState,
    concurrency: int = 8,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> CourseState:
    """
    Main function to generate all section HTML structures using LangGraph Send pattern.
    
    Args:
        course_state: CourseState with theories filled
        concurrency: Number of concurrent requests
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
        
    Returns:
        Updated CourseState with all HTML structures filled
    """
    print(f"🚀 Starting parallel HTML generation with max_retries={max_retries}, "
          f"initial_delay={initial_delay}s, backoff_factor={backoff_factor}")
    
    # Build the graph with retry configuration
    graph = build_html_generation_graph(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )
    
    # Initialize state
    initial_state = HtmlFormattingState(course_state=course_state)
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    return result["course_state"]
