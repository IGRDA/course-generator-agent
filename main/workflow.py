from main.state import CourseState, CourseConfig
from agents.index_generator.agent import generate_course_state
from agents.section_theory_generator.agent import generate_all_section_theories
from agents.activities_generator.agent import generate_all_section_activities
from agents.html_formatter.agent import generate_all_section_html
from agents.html_formatter.exporter import export_to_html
from agents.image_generator.agent import generate_all_section_images
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


def generate_activities_node(state: CourseState) -> CourseState:
    """Generate activities for all sections in parallel"""
    print("Generating section activities in parallel...")
    
    updated_state = generate_all_section_activities(
        state,
        concurrency=state.config.activities_concurrency,
        max_retries=state.config.max_retries
    )
    
    print("All section activities generated successfully!")
    return updated_state


def calculate_metadata_node(state: CourseState) -> CourseState:
    """Calculate IDs, indexes and durations for all course elements"""
    print("Calculating course metadata (IDs, Indexes, Durations)...")
    
    for m_idx, module in enumerate(state.modules):
        # Simple string ID matching index
        module.id = str(m_idx + 1)
        module.index = m_idx + 1
        
        if not module.description:
            module.description = module.title
            
        for sm_idx, submodule in enumerate(module.submodules):
            # Submodules only have index, no id
            submodule.index = sm_idx + 1
            
            if not submodule.description:
                submodule.description = submodule.title
            
            for s_idx, section in enumerate(submodule.sections):
                # Sections only have index, no id
                section.index = s_idx + 1
                
                if not section.description:
                    section.description = section.title
            
            # Calculate submodule duration: 0.1 hours per section
            submodule.duration = len(submodule.sections) * 0.1
            
        # Calculate module duration
        module.duration = sum(sm.duration for sm in module.submodules)
        
    print("Metadata calculation completed!")
    return state


def generate_html_node(state: CourseState) -> CourseState:
    """Generate HTML structure for all sections in parallel"""
    print("Generating HTML structures in parallel...")
    
    updated_state = generate_all_section_html(
        state,
        concurrency=state.config.html_concurrency,
        max_retries=state.config.max_retries
    )
    
    print("All HTML structures generated successfully!")
    return updated_state


def generate_images_node(state: CourseState) -> CourseState:
    """Generate images for all HTML blocks using configured image search provider"""
    print(f"Generating images for HTML blocks using {state.config.image_search_provider}...")
    if state.config.use_vision_ranking:
        print(f"   Vision ranking enabled: fetching {state.config.num_images_to_fetch} images, ranking with {state.config.vision_llm_provider}")
    else:
        print("   Vision ranking disabled: picking first image result")
    
    updated_state = generate_all_section_images(
        state,
        max_retries=state.config.max_retries,
        use_vision_ranking=state.config.use_vision_ranking,
        num_images_to_fetch=state.config.num_images_to_fetch,
        vision_provider=state.config.vision_llm_provider,
    )
    
    print("All images generated successfully!")
    return updated_state


# Build the graph
def build_course_generation_graph():
    """Build and return the course generation graph"""
    graph = StateGraph(CourseState)
    
    # Add nodes for complete course generation pipeline
    graph.add_node("generate_index", generate_index_node)
    graph.add_node("generate_theories", generate_theories_node)
    graph.add_node("generate_activities", generate_activities_node)
    graph.add_node("calculate_metadata", calculate_metadata_node)
    graph.add_node("generate_html", generate_html_node)
    graph.add_node("generate_images", generate_images_node)
    
    # Add edges for sequential execution
    graph.add_edge(START, "generate_index")
    graph.add_edge("generate_index", "generate_theories")
    graph.add_edge("generate_theories", "generate_activities")
    graph.add_edge("generate_activities", "calculate_metadata")
    graph.add_edge("calculate_metadata", "generate_html")
    graph.add_edge("generate_html", "generate_images")
    graph.add_edge("generate_images", END)
    
    return graph.compile()


if __name__ == "__main__":
    import os
    import json
    import argparse
    from datetime import datetime
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a course")
    parser.add_argument("--total-pages", type=int, default=2, help="Total pages for the course (default: 2)")
    args = parser.parse_args()
    
    # Ensure you have set: export MISTRAL_API_KEY=your_key
    
    # Build the graph
    app = build_course_generation_graph()
    
    # Create initial CourseState with config and minimal content
    config = CourseConfig(
        title="Chess masterclass",
        text_llm_provider="mistral",  # LLM provider: mistral | gemini | groq | openai
        web_search_provider="tavily",  # Web search provider: ddg | tavily | wikipedia
        total_pages=args.total_pages,  # Total pages for the course
        words_per_page=400,  # Target words per page
        language="Español",        
        description="",
        max_retries=8,
        concurrency=10,  # Number of concurrent section theory generations
        use_reflection=True,  # Enable reflection pattern for fact verification (default: False)
        num_reflection_queries=7,  # Number of verification queries per section (default: 3)
        # Activities configuration
        activities_concurrency=30,  # Number of concurrent activity generations
        activity_selection_mode="deterministic",  # "random" or "deterministic"
        num_activities_per_section=1,  # Number of quiz activities (+ multiple_choice + multi_selection)
        # HTML configuration
        html_concurrency=15,  # Number of concurrent HTML generations
        select_html="random",  # "LLM" | "random"
        html_formats="paragraphs|accordion|tabs|carousel|flip|timeline|conversation",  # Pipe-separated list of available formats
        html_random_seed=42,  # Seed for deterministic random selection
        include_quotes_in_html=True,  # Include quote elements
        include_tables_in_html=True,  # Include table elements
        # Image generation configuration
        image_search_provider="freepik",  # Image search provider: bing | freepik | ddg
        use_vision_ranking=False,  # Use vision LLM to rank images (slower but better quality)
        num_images_to_fetch=5,  # Number of images to fetch for ranking
        vision_llm_provider="pixtral",  # Vision LLM provider for image ranking
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
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in final_state.title)
    safe_title = safe_title.replace(' ', '_')
    base_filename = f"{safe_title}_{timestamp}"
    
    # Save JSON file
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        # Use by_alias=True to ensure keyConcept is serialized as keyConcept
        f.write(final_state.model_dump_json(indent=2, by_alias=True))
    print(f"\n✅ JSON saved to: {json_path}")
    
    # Save HTML file
    html_path = os.path.join(output_dir, f"{base_filename}.html")
    export_to_html(final_state, html_path)
    print(f"✅ HTML saved to: {html_path}")
    
    # Print summary
    total_sections = sum(len(s.sections) for m in final_state.modules for s in m.submodules)
    print(f"\n📊 Course Summary:")
    print(f"   Title: {final_state.title}")
    print(f"   Modules: {len(final_state.modules)}")
    print(f"   Total Sections: {total_sections}")
    print(f"   Language: {final_state.language}")
    print(f"   HTML Selection: {final_state.config.select_html}")
    print(f"   Available Formats: {final_state.config.html_formats}")
