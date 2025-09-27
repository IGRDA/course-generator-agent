import asyncio
import json
from main.state import CourseState
from agents.index_generator.agent import generate_course_state
from agents.section_theory_generator.agent import generate_all_section_theories
from langgraph.graph import StateGraph, START, END

def generate_skeleton_node(state: CourseState) -> CourseState:
    """Generate the course skeleton with empty theories"""
    print("Generating course skeleton...")
    
    # Extract generation parameters from state (assuming they exist or use defaults)
    title = getattr(state, 'title', 'Default Course Title')
    total_pages = getattr(state, 'total_pages', 100)
    description = getattr(state, 'description', '')
    language = getattr(state, 'language', 'English')
    max_retries = getattr(state, 'max_retries', 3)
    
    # Generate new course state skeleton (with empty theories as per existing prompt)
    skeleton_state = generate_course_state(
        title=title,
        total_pages=total_pages,
        description=description,
        language=language,
        max_retries=max_retries
    )
    
    print("Course skeleton generated successfully!")
    return skeleton_state


def generate_theories_node(state: CourseState) -> CourseState:
    """Generate all section theories in parallel"""
    print("Generating section theories in parallel...")
    
    # Extract concurrency setting from state, default to 8
    concurrency = getattr(state, 'concurrency', 8)
    
    # Run the async theory generation
    updated_state = asyncio.run(
        generate_all_section_theories(state, concurrency)
    )
    
    print("All section theories generated successfully!")
    return updated_state


# Build the graph
def build_course_generation_graph():
    """Build and return the course generation graph"""
    graph = StateGraph(CourseState)
    
    # Add nodes for two-step generation
    graph.add_node("generate_skeleton", generate_skeleton_node)
    graph.add_node("generate_theories", generate_theories_node)
    
    # Add edges for sequential execution
    graph.add_edge(START, "generate_skeleton")
    graph.add_edge("generate_skeleton", "generate_theories")
    graph.add_edge("generate_theories", END)
    
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
    
    # Create a minimal initial CourseState that will be populated during generation
    # We only need to set the generation parameters as attributes
    initial_state = CourseState(
        title="Introduction to Modern Data Engineering",
        n_modules=1,  # Will be overridden by skeleton generation
        n_submodules=1,  # Will be overridden by skeleton generation
        n_sections=1,  # Will be overridden by skeleton generation
        n_words=400,  # Will be overridden by skeleton generation
        modules=[]  # Will be populated by skeleton generation
    )
    
    # Set generation parameters as attributes for the workflow nodes
    initial_state.total_pages = 5  # Increased to match the parallel example
    initial_state.description = "A practical course covering data ingestion, warehousing, orchestration, and observability."
    initial_state.language = "Euskera"  # Can be changed to any language (e.g., "Spanish", "French", "German", etc.)
    initial_state.max_retries = 3
    initial_state.concurrency = 1  # Number of concurrent section theory generations
    
    # Run the graph
    result = app.invoke(initial_state)
    
    # Print the final course state
    print(result)