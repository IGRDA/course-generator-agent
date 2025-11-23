from typing import Annotated, List
from operator import add
from pydantic import BaseModel, Field
from main.state import CourseState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from LLMs.text2text import create_text_llm, resolve_text_model_name
from tools.websearch import create_web_search
from tools.imagesearch import create_image_search  # Import for testing
from .prompts import (
    section_theory_prompt,
    query_generation_prompt,
    reflection_prompt,
    regeneration_prompt,
    STYLE_COURSE_INTRO,
    STYLE_MODULE_START,
    STYLE_SUBMODULE_START,
    STYLE_CONTINUATION,
    STYLE_DEEP_DIVE
)


# ---- Pydantic models for structured outputs ----
class QueryList(BaseModel):
    """List of search queries for fact verification"""
    queries: List[str] = Field(
        description="List of search queries to verify facts, formulas, laws, dates, and concepts"
    )


class Reflection(BaseModel):
    """Reflection and critique of section content"""
    critique: str = Field(
        description="Detailed critique identifying factual errors, outdated info, or improvements needed"
    )
    needs_revision: bool = Field(
        description="Whether the content needs to be regenerated based on findings"
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
    use_reflection: bool = True
    num_queries: int = 3


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
                    section_title=section.title,
                    use_reflection=state.course_state.config.use_reflection,
                    num_queries=state.course_state.config.num_reflection_queries
                )
                # Send to the generate_section node
                sends.append(Send("generate_section", task))
    
    return sends


def select_style_guidelines(
    module_idx: int,
    submodule_idx: int,
    section_idx: int,
    total_sections_in_submodule: int
) -> str:
    """
    Select appropriate style guidelines based on section position in the course structure.
    
    Args:
        module_idx: Index of the current module
        submodule_idx: Index of the current submodule
        section_idx: Index of the current section
        total_sections_in_submodule: Total number of sections in the current submodule
    
    Returns:
        Style guidelines string to inject into the prompt
    """
    # Course introduction - very first section
    if module_idx == 0 and submodule_idx == 0 and section_idx == 0:
        return STYLE_COURSE_INTRO
    
    # Module start - first section of a new module (but not the first module)
    if submodule_idx == 0 and section_idx == 0:
        return STYLE_MODULE_START
    
    # Submodule start - first section of a new submodule (but not first of module)
    if section_idx == 0:
        return STYLE_SUBMODULE_START
    
    # Deep dive - later sections in a submodule (last half)
    if section_idx >= total_sections_in_submodule // 2 and total_sections_in_submodule > 2:
        return STYLE_DEEP_DIVE
    
    # Default continuation - middle sections
    return STYLE_CONTINUATION


def reflect_and_improve(
    theory: str,
    section_title: str,
    module_title: str,
    submodule_title: str,
    language: str,
    n_words: int,
    num_queries: int,
    provider: str,
    web_search_provider: str
) -> str:
    """
    Apply reflection pattern: generate queries → parallel search → reflect → regenerate if needed
    """
    try:
        # Create LLM with specified provider
        model_name = resolve_text_model_name(provider)
        llm_kwargs = {"temperature": 0.3}
        if model_name:
            llm_kwargs["model_name"] = model_name
        llm = create_text_llm(provider=provider, **llm_kwargs)
        
        # Create web search function with specified provider
        web_search = create_web_search(web_search_provider)
        
        # Step 1: Generate verification queries
        query_chain = query_generation_prompt | llm.with_structured_output(QueryList)
        query_list = query_chain.invoke({
            "theory": theory,
            "section_title": section_title,
            "module_title": module_title,
            "submodule_title": submodule_title,
            "k": num_queries
        })
        
        # Step 2: Execute all web searches in parallel
        def safe_search(query: str, idx: int) -> str:
            try:
                result = web_search(query, max_results=num_queries)
                return f"Query {idx}: {query}\nResult: {result}\n"
            except Exception as e:
                return f"Query {idx}: {query}\nResult: Search failed - {str(e)}\n"
        
        parallel_searches = {
            f"search_{i}": RunnableLambda(lambda x, q=q, idx=i+1: safe_search(q, idx))
            for i, q in enumerate(query_list.queries)
        }
        
        search_results = RunnableParallel(**parallel_searches).invoke({})
        all_search_results = "\n".join(search_results.values())
        
        # Step 3: Reflect on content quality
        reflection_chain = reflection_prompt | llm.with_structured_output(Reflection)
        reflection_result = reflection_chain.invoke({
            "theory": theory,
            "section_title": section_title,
            "module_title": module_title,
            "submodule_title": submodule_title,
            "search_results": all_search_results
        })
        
        # Step 4: Regenerate if needed
        if reflection_result.needs_revision:
            regeneration_chain = regeneration_prompt | llm | StrOutputParser()
            improved_theory = regeneration_chain.invoke({
                "theory": theory,
                "section_title": section_title,
                "module_title": module_title,
                "submodule_title": submodule_title,
                "reflection": reflection_result.critique,
                "search_results": all_search_results,
                "language": language,
                "n_words": n_words
            }).strip()
            
            print(f"  ↻ Reflection suggested improvements, regenerated content")
            return improved_theory
        else:
            print(f"  ✓ Reflection: content is accurate")
            return theory
            
    except Exception as e:
        print(f"  ⚠ Reflection failed: {str(e)}, using original content")
        return theory

def generate_section(state: SectionTask) -> dict:
    """
    Generate theory for a single section.
    Optionally applies reflection pattern for fact verification and improvement.
    LangGraph's built-in retry mechanism handles failures automatically.
    """
    # Extract context from course state
    module = state.course_state.modules[state.module_idx]
    submodule = module.submodules[state.submodule_idx]
    provider = state.course_state.config.text_llm_provider
    
    # Calculate target word count
    total_sections = sum(
        len(submodule.sections) 
        for module in state.course_state.modules 
        for submodule in module.submodules
    )
    n_words = state.course_state.config.total_pages * state.course_state.config.words_per_page // total_sections
    
    # Create LLM with specified provider
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.2}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Determine style guidelines based on position
    style_guidelines = select_style_guidelines(
        module_idx=state.module_idx,
        submodule_idx=state.submodule_idx,
        section_idx=state.section_idx,
        total_sections_in_submodule=len(submodule.sections)
    )
    
    # Generate initial content using LCEL chain (retry handled by LangGraph)
    section_chain = section_theory_prompt | llm | StrOutputParser()
    theory = section_chain.invoke({
        "course_title": state.course_state.title,
        "module_title": module.title,
        "submodule_title": submodule.title,
        "section_title": state.section_title,
        "language": state.course_state.language,
        "n_words": n_words,
        "style_guidelines": style_guidelines
    }).strip()
    
    print(f"✓ Generated theory for Module {state.module_idx+1}, "
          f"Submodule {state.submodule_idx+1}, Section {state.section_idx+1}")
    
    # Apply reflection pattern if enabled
    if state.use_reflection:
        theory = reflect_and_improve(
            theory=theory,
            section_title=state.section_title,
            module_title=module.title,
            submodule_title=submodule.title,
            language=state.course_state.language,
            n_words=n_words,
            num_queries=state.num_queries,
            provider=provider,
            web_search_provider=state.course_state.config.web_search_provider
        )
    
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
    backoff_factor: float = 2.0,
    use_reflection: bool = False,
    num_reflection_queries: int = 3
) -> CourseState:
    """
    Main function to generate all section theories using LangGraph Send pattern.
    
    Args:
        course_state: CourseState with skeleton structure (empty theories)
        concurrency: Number of concurrent requests (not used directly, LangGraph handles this)
        max_retries: Maximum number of retry attempts for each LLM call
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)
        use_reflection: Whether to use reflection pattern for fact verification (default: False)
        num_reflection_queries: Number of verification queries to generate (default: 3)
        
    Returns:
        Updated CourseState with all theories filled
    """
    reflection_msg = f", reflection={'ON' if use_reflection else 'OFF'}"
    if use_reflection:
        reflection_msg += f" (queries={num_reflection_queries})"
    
    print(f"🚀 Starting parallel generation with max_retries={max_retries}, "
          f"initial_delay={initial_delay}s, backoff_factor={backoff_factor}{reflection_msg}")
    
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

