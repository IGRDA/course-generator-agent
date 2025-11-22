from main.state import CourseState, CourseConfig
from agents.index_generator.agent import generate_course_state
from agents.section_theory_generator.agent import generate_all_section_theories
from langgraph.graph import StateGraph, START, END

def generate_index_node(state: CourseState) -> CourseState:
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
        words_per_page=config.words_per_page,
        provider=config.text_llm_provider
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
    use_reflection = state.config.use_reflection
    num_reflection_queries = state.config.num_reflection_queries
    
    # Run theory generation
    updated_state = generate_all_section_theories(
        state, 
        concurrency=concurrency,
        max_retries=max_retries,
        use_reflection=use_reflection,
        num_reflection_queries=num_reflection_queries
    )
    
    print("All section theories generated successfully!")
    return updated_state


# Build the graph
def build_course_generation_graph():
    """Build and return the course generation graph"""
    graph = StateGraph(CourseState)
    
    # Add nodes for two-step generation
    graph.add_node("generate_index", generate_index_node)
    graph.add_node("generate_theories", generate_theories_node)
    
    # Add edges for sequential execution
    graph.add_edge(START, "generate_index")
    graph.add_edge("generate_index", "generate_theories")
    graph.add_edge("generate_theories", END)
    
    return graph.compile()


if __name__ == "__main__":
    # Ensure you have set: export MISTRAL_API_KEY=your_key
    
    # Build the graph
    app = build_course_generation_graph()
    
    # Create initial CourseState with config and minimal content
    config = CourseConfig(
        title="Reforma legal de España 2024",
        text_llm_provider="mistral",  # LLM provider: mistral | gemini | groq | openai
        web_search_provider="ddg",  # Web search provider: ddg | tavily | wikipedia
        total_pages=5,  # Total pages for the course
        words_per_page=400,  # Target words per page
        description="",
        language="Español",  # Can be changed to any language (e.g., "Spanish", "French", "German", etc.)
        max_retries=3,
        concurrency=4,  # Number of concurrent section theory generations
        use_reflection=True,  # Enable reflection pattern for fact verification (default: False)
        num_reflection_queries=3  # Number of verification queries per section (default: 3)
    )
    
    initial_state = CourseState(
        config=config,
        title=config.title,  # Initialize from config, can be refined during generation
        modules=[]  # Will be populated by skeleton generation
    )
    
    # Run the graph with custom trace name
    result = app.invoke(
        initial_state,
        config={"run_name": f"{initial_state.title}"}
    )
    
    # Print the final course state
    print("Workflow completed successfully!")
    
    # Extract the final state from LangGraph result (which is a dictionary)
    final_state = result if isinstance(result, CourseState) else CourseState.model_validate(result)
    
    # Pretty print the final course state
    print(final_state.model_dump_json(indent=2))