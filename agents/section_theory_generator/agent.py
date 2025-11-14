import os
from typing import Annotated
from operator import add
from pydantic import BaseModel, Field
from main.state import CourseState
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from .prompts import section_theory_prompt


MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")

# Initialize the LLM for section theory generation
llm = ChatMistralAI(
    model=MODEL_NAME,
    temperature=0.7,  # Slightly higher temperature for more creative content
)


# ---- State for individual section task ----
class SectionTask(BaseModel):
    """State for processing a single section"""
    course_state: CourseState
    module_idx: int
    submodule_idx: int
    section_idx: int
    section_title: str
    theory: str = ""


# ---- State for aggregating results ----
class TheoryGenerationState(BaseModel):
    """State for the theory generation subgraph"""
    course_state: CourseState
    completed_sections: Annotated[list[dict], add] = Field(default_factory=list)
    total_sections: int = 0


def plan_sections(state: TheoryGenerationState) -> dict:
    """
    Update the total_sections count in the state.
    The actual Send objects are created by continue_to_sections().
    """
    section_count = 0
    
    for module in state.course_state.modules:
        for submodule in module.submodules:
            section_count += len(submodule.sections)
    
    print(f"📋 Planning {section_count} sections for parallel generation")
    
    return {"total_sections": section_count}


def continue_to_sections(state: TheoryGenerationState) -> list[Send]:
    """
    Fan-out: Create a Send for each section to process in parallel.
    This is used as a conditional edge function.
    """
    sends = []
    
    for m_idx, module in enumerate(state.course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                # Create a task for each section
                task = SectionTask(
                    course_state=state.course_state,
                    module_idx=m_idx,
                    submodule_idx=sm_idx,
                    section_idx=s_idx,
                    section_title=section.title
                )
                # Send to the generate_section node
                sends.append(Send("generate_section", task))
    
    return sends


def generate_section(state: SectionTask) -> dict:
    """
    Generate theory for a single section using LCEL chain.
    LangGraph's built-in retry mechanism handles failures automatically.
    """
    # Extract context from course state
    module = state.course_state.modules[state.module_idx]
    submodule = module.submodules[state.submodule_idx]
    
    # Calculate target word count
    total_sections = sum(
        len(submodule.sections) 
        for module in state.course_state.modules 
        for submodule in module.submodules
    )
    n_words = state.course_state.config.total_pages * state.course_state.config.words_per_page // total_sections
    
    # Generate content using LCEL chain (retry handled by LangGraph)
    # LCEL chain for section theory generation
    section_chain = section_theory_prompt | llm | StrOutputParser()
    theory = section_chain.invoke({
        "course_title": state.course_state.title,
        "module_title": module.title,
        "submodule_title": submodule.title,
        "section_title": state.section_title,
        "language": state.course_state.language,
        "n_words": n_words
    }).strip()
    
    print(f"✓ Generated theory for Module {state.module_idx+1}, "
          f"Submodule {state.submodule_idx+1}, Section {state.section_idx+1}")
    
    # Return the completed section info
    return {
        "completed_sections": [{
            "module_idx": state.module_idx,
            "submodule_idx": state.submodule_idx,
            "section_idx": state.section_idx,
            "theory": theory
        }]
    }


def reduce_sections(state: TheoryGenerationState) -> dict:
    """
    Fan-in: Aggregate all generated theories back into the course state.
    """
    print(f"📦 Reducing {len(state.completed_sections)} completed sections")
    
    # Update course state with all generated theories
    for section_info in state.completed_sections:
        m_idx = section_info["module_idx"]
        sm_idx = section_info["submodule_idx"]
        s_idx = section_info["section_idx"]
        theory = section_info["theory"]
        
        state.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx].theory = theory
    
    print(f"✅ All {state.total_sections} section theories generated successfully!")
    
    return {"course_state": state.course_state}


def build_theory_generation_graph(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Build the theory generation subgraph using Send for dynamic parallelization.
    
    Args:
        max_retries: Number of retries for each section generation
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff (e.g., 2.0 means delays of 1s, 2s, 4s, 8s...)
    """
    graph = StateGraph(TheoryGenerationState)
    
    # Configure retry policy with exponential backoff
    retry_policy = RetryPolicy(
        max_attempts=max_retries,
        initial_interval=initial_delay,
        backoff_factor=backoff_factor,
        max_interval=60.0  # Cap maximum delay at 60 seconds
    )
    
    # Add nodes
    graph.add_node("plan_sections", plan_sections)
    graph.add_node("generate_section", generate_section, retry=retry_policy)
    graph.add_node("reduce_sections", reduce_sections)
    
    # Add edges
    graph.add_edge(START, "plan_sections")
    # Use conditional edges to send tasks dynamically to generate_section
    graph.add_conditional_edges("plan_sections", continue_to_sections, ["generate_section"])
    # All generate_section tasks feed into reduce_sections
    graph.add_edge("generate_section", "reduce_sections")
    graph.add_edge("reduce_sections", END)
    
    return graph.compile()


def generate_all_section_theories(
    course_state: CourseState, 
    concurrency: int = 8, 
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> CourseState:
    """
    Main function to generate all section theories using LangGraph Send pattern.
    
    Args:
        course_state: CourseState with skeleton structure (empty theories)
        concurrency: Number of concurrent requests (not used directly, LangGraph handles this)
        max_retries: Maximum number of retry attempts for each LLM call
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)
        
    Returns:
        Updated CourseState with all theories filled
    """
    print(f"🚀 Starting parallel generation with max_retries={max_retries}, "
          f"initial_delay={initial_delay}s, backoff_factor={backoff_factor}")
    
    # Build the graph with retry configuration
    graph = build_theory_generation_graph(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )
    
    # Initialize state
    initial_state = TheoryGenerationState(course_state=course_state)
    
    # Execute the graph (LangGraph handles parallelization automatically)
    # Note: concurrency is controlled by LangGraph's executor settings
    result = graph.invoke(initial_state)
    
    return result["course_state"]

