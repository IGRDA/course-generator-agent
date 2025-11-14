import asyncio
import json
from main.state import CourseState, CourseConfig
from agents.index_generator.agent import generate_course_state
from agents.section_theory_generator.agent import generate_all_section_theories
from agents.section_html_generator.agent import generate_all_section_html
from langgraph.graph import StateGraph, START, END

def generate_skeleton_node(state: CourseState) -> CourseState:
    """Generate the course skeleton with empty theories while preserving config"""
    print("Generating course skeleton...")
    
    # Use config from the existing state
    config = state.config
    
    # Generate new course content skeleton (with empty theories)
    content_skeleton = generate_course_state(
        title=state.title,
        total_pages=config.total_pages,
        description=config.description,
        language=config.language,
        max_retries=config.max_retries,
        words_per_page=config.words_per_page
    )
    
    # Preserve original config, update only content fields
    state.title = content_skeleton.title
    state.modules = content_skeleton.modules
    
    print("Course skeleton generated successfully!")
    return state


def generate_theories_node(state: CourseState) -> CourseState:
    """Generate all section theories in parallel using LangGraph Send"""
    print("Generating section theories in parallel...")
    
    # Use config settings
    concurrency = state.config.concurrency
    max_retries = state.config.max_retries
    
    # Run theory generation (no longer async, uses LangGraph subgraph)
    updated_state = generate_all_section_theories(
        state, 
        concurrency=concurrency,
        max_retries=max_retries
    )
    
    print("All section theories generated successfully!")
    return updated_state


def generate_html_node(state: CourseState) -> CourseState:
    """Generate HTML for all section theories in parallel"""
    print("Generating HTML for all sections in parallel...")
    
    # Use concurrency setting from config
    concurrency = state.config.concurrency
    
    # Run the async HTML generation
    updated_state = asyncio.run(
        generate_all_section_html(state, concurrency)
    )
    
    print("All section HTML generated successfully!")
    return updated_state


# Build the graph
def build_course_generation_graph():
    """Build and return the course generation graph"""
    graph = StateGraph(CourseState)
    
    # Add nodes for three-step generation
    graph.add_node("generate_skeleton", generate_skeleton_node)
    graph.add_node("generate_theories", generate_theories_node)
    graph.add_node("generate_html", generate_html_node)
    
    # Add edges for sequential execution
    graph.add_edge(START, "generate_skeleton")
    graph.add_edge("generate_skeleton", "generate_theories")
    graph.add_edge("generate_theories", "generate_html")
    graph.add_edge("generate_html", END)
    
    return graph.compile()


if __name__ == "__main__":
    # Ensure you have set: export MISTRAL_API_KEY=your_key
    
    # Build the graph
    app = build_course_generation_graph()
    
    
    # Save the graph visualization
    graph_viz = app.get_graph().draw_mermaid_png()
        
    # Save to file
    with open("workflow_graph.png", "wb") as f:
        f.write(graph_viz)
    
    # Create initial CourseState with config and minimal content
    config = CourseConfig(
        total_pages=5,  # Total pages for the course
        words_per_page=400,  # Target words per page
        description="A practical course covering data ingestion, warehousing, orchestration, and observability.",
        language="Euskera",  # Can be changed to any language (e.g., "Spanish", "French", "German", etc.)
        max_retries=3,
        concurrency=4  # Number of concurrent section theory generations
    )
    
    initial_state = CourseState(
        config=config,
        title="Introduction to Modern Data Engineering",
        modules=[]  # Will be populated by skeleton generation
    )
    
    # Run the graph
    result = app.invoke(initial_state)
    
    # Print the final course state
    print("Workflow completed successfully!")
    
    # Extract the final state from LangGraph result (which is a dictionary)
    final_state = result if isinstance(result, CourseState) else CourseState.model_validate(result)
    
    # Pretty print the final course state
    print(final_state.model_dump_json(indent=2))