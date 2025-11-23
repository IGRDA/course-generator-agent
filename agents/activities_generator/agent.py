from typing import Annotated, List
import random
from operator import add
from pydantic import BaseModel, Field
from main.state import CourseState, GlossaryTerm, Activity, FinalActivity
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from LLMs.text2text import create_text_llm, resolve_text_model_name
from .prompts import activities_generation_prompt, correction_prompt


# ---- Activity Selection ----
def select_activity_types(mode: str, num: int, section_idx: int) -> tuple[List[str], List[str]]:
    """
    Select activity types based on mode (random or deterministic).
    
    Args:
        mode: "random" or "deterministic"
        num: Number of quiz activities to select
        section_idx: Section index for deterministic cycling
        
    Returns:
        Tuple of (quiz_types, final_types)
    """
    quiz_types = ["order_list", "fill_gaps", "swipper", "linking_terms"]
    final_types = ["group_activity", "discussion_forum", "individual_project", "open_ended_quiz"]
    
    if mode == "random":
        # Random selection
        selected_quiz = random.sample(quiz_types, min(num, len(quiz_types)))
        selected_final = [random.choice(final_types)]
    else:  # deterministic - cycle through types
        # Cycle through quiz types based on section index
        selected_quiz = []
        for i in range(num):
            type_idx = (section_idx + i) % len(quiz_types)
            selected_quiz.append(quiz_types[type_idx])
        # Cycle through final types
        final_idx = section_idx % len(final_types)
        selected_final = [final_types[final_idx]]
    
    # Always add multiple_choice and multi_selection
    selected_quiz.extend(["multiple_choice", "multi_selection"])
    
    return selected_quiz, selected_final


# ---- Output Model ----
class ActivitiesOutput(BaseModel):
    """Complete activities output for a section"""
    glossary: List[GlossaryTerm] = Field(..., min_length=1, max_length=4, description="1-4 glossary terms")
    key_concept: str = Field(..., description="One sentence key concept summary")
    activities: List[Activity] = Field(..., min_length=1, description="List of activities")
    final_activities: List[FinalActivity] = Field(..., min_length=1, description="List of final activities")


# ---- State for individual section task ----
class SectionActivitiesTask(BaseModel):
    """State for processing a single section's activities"""
    course_state: CourseState
    module_idx: int
    submodule_idx: int
    section_idx: int
    section_title: str
    theory: str
    activity_types: List[str]
    final_activity_types: List[str]


# ---- State for aggregating results ----
class ActivitiesGenerationState(BaseModel):
    """State for the activities generation subgraph"""
    course_state: CourseState
    completed_activities: Annotated[list[dict], add] = Field(default_factory=list)
    total_sections: int = 0


def plan_activities(state: ActivitiesGenerationState) -> dict:
    """Update the total_sections count in the state."""
    section_count = 0
    
    for module in state.course_state.modules:
        for submodule in module.submodules:
            section_count += len(submodule.sections)
    
    print(f"📋 Planning {section_count} sections for activities generation")
    
    return {"total_sections": section_count}


def continue_to_activities(state: ActivitiesGenerationState) -> list[Send]:
    """Fan-out: Create a Send for each section to process in parallel."""
    sends = []
    
    mode = state.course_state.config.activity_selection_mode
    num = state.course_state.config.num_activities_per_section
    global_section_idx = 0
    
    for m_idx, module in enumerate(state.course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                # Select activity types for this section
                quiz_types, final_types = select_activity_types(mode, num, global_section_idx)
                
                # Create a task for each section
                task = SectionActivitiesTask(
                    course_state=state.course_state,
                    module_idx=m_idx,
                    submodule_idx=sm_idx,
                    section_idx=s_idx,
                    section_title=section.title,
                    theory=section.theory,
                    activity_types=quiz_types,
                    final_activity_types=final_types
                )
                # Send to the generate_section_activities node
                sends.append(Send("generate_section_activities", task))
                global_section_idx += 1
    
    return sends


def generate_section_activities(state: SectionActivitiesTask) -> dict:
    """
    Generate activities for a single section.
    Uses structured output with RetryWithErrorOutputParser for validation.
    """
    # Extract context
    provider = state.course_state.config.text_llm_provider
    max_retries = state.course_state.config.max_retries
    
    # Create LLM
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.5}  # Slightly higher temp for creative activities
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=ActivitiesOutput)
    
    # Create fix parser for retry
    fix_parser = RetryWithErrorOutputParser.from_llm(
        llm=llm,
        parser=parser,
        max_retries=max_retries,
    )
    
    # Generate using LCEL chain
    chain = activities_generation_prompt | llm | StrOutputParser()
    
    # Build activity types description
    activity_desc = ", ".join(state.activity_types)
    final_desc = ", ".join(state.final_activity_types)
    
    raw = chain.invoke({
        "theory": state.theory,
        "section_title": state.section_title,
        "language": state.course_state.language,
        "activity_types": activity_desc,
        "final_activity_types": final_desc,
        "format_instructions": parser.get_format_instructions(),
    })
    
    # Try to parse, with fallback to retry parser
    try:
        result = parser.parse(raw)
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
    
    print(f"✓ Generated activities for Module {state.module_idx+1}, "
          f"Submodule {state.submodule_idx+1}, Section {state.section_idx+1}")
    
    # Return the completed section info
    return {
        "completed_activities": [{
            "module_idx": state.module_idx,
            "submodule_idx": state.submodule_idx,
            "section_idx": state.section_idx,
            "glossary": result.glossary,
            "key_concept": result.key_concept,
            "activities": result.activities,
            "final_activities": result.final_activities
        }]
    }


def reduce_activities(state: ActivitiesGenerationState) -> dict:
    """Fan-in: Aggregate all generated activities back into the course state."""
    print(f"📦 Reducing {len(state.completed_activities)} completed activities")
    
    # Update course state with all generated activities
    for activity_info in state.completed_activities:
        m_idx = activity_info["module_idx"]
        sm_idx = activity_info["submodule_idx"]
        s_idx = activity_info["section_idx"]
        
        section = state.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx]
        section.glossary = activity_info["glossary"]
        section.key_concept = activity_info["key_concept"]
        section.activities = activity_info["activities"]
        section.final_activities = activity_info["final_activities"]
    
    print(f"✅ All {state.total_sections} section activities generated successfully!")
    
    return {"course_state": state.course_state}


def build_activities_generation_graph(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Build the activities generation subgraph using Send for dynamic parallelization.
    
    Args:
        max_retries: Number of retries for each section generation
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
    """
    graph = StateGraph(ActivitiesGenerationState)
    
    # Configure retry policy with exponential backoff
    retry_policy = RetryPolicy(
        max_attempts=max_retries,
        initial_interval=initial_delay,
        backoff_factor=backoff_factor,
        max_interval=60.0
    )
    
    # Add nodes
    graph.add_node("plan_activities", plan_activities)
    graph.add_node("generate_section_activities", generate_section_activities, retry=retry_policy)
    graph.add_node("reduce_activities", reduce_activities)
    
    # Add edges
    graph.add_edge(START, "plan_activities")
    graph.add_conditional_edges("plan_activities", continue_to_activities, ["generate_section_activities"])
    graph.add_edge("generate_section_activities", "reduce_activities")
    graph.add_edge("reduce_activities", END)
    
    return graph.compile()


def generate_all_section_activities(
    course_state: CourseState,
    concurrency: int = 8,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> CourseState:
    """
    Main function to generate all section activities using LangGraph Send pattern.
    
    Args:
        course_state: CourseState with theories filled
        concurrency: Number of concurrent requests
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
        
    Returns:
        Updated CourseState with all activities filled
    """
    print(f"🚀 Starting parallel activities generation with max_retries={max_retries}, "
          f"initial_delay={initial_delay}s, backoff_factor={backoff_factor}")
    
    # Build the graph with retry configuration
    graph = build_activities_generation_graph(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )
    
    # Initialize state
    initial_state = ActivitiesGenerationState(course_state=course_state)
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    return result["course_state"]

