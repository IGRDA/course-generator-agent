from typing import Optional
from langchain_core.runnables import RunnableConfig

from main.state import CourseState, CourseConfig
from main.output_manager import OutputManager
from agents.index_generator.agent import generate_course_state
from agents.section_theory_generator.agent import generate_all_section_theories
from agents.activities_generator.agent import generate_all_section_activities
from agents.html_formatter.agent import generate_all_section_html
from agents.image_generator.agent import generate_all_section_images
from agents.bibliography_generator.agent import generate_course_bibliography
from langgraph.graph import StateGraph, START, END


def _get_output_manager(config: Optional[RunnableConfig]) -> Optional[OutputManager]:
    """Extract OutputManager from LangGraph config if present."""
    if config is None:
        return None
    return config.get("configurable", {}).get("output_manager")

def generate_index_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate the course skeleton with empty theories while preserving config"""
    print("Generating course skeleton...")
    
    # Use config from the existing state
    course_config = state.config
    
    # Generate new course content skeleton (with empty theories)
    # Now includes research phase if enabled
    content_skeleton = generate_course_state(
        title=state.title,
        total_pages=course_config.total_pages,
        description=course_config.description,
        language=course_config.language,
        max_retries=course_config.max_retries,
        words_per_page=course_config.words_per_page,
        provider=course_config.text_llm_provider,
        # Research configuration
        enable_research=course_config.enable_research,
        web_search_provider=course_config.web_search_provider,
        research_max_queries=course_config.research_max_queries,
        research_max_results_per_query=course_config.research_max_results_per_query,
        # Audience configuration
        target_audience=course_config.target_audience,
    )
    
    # Transfer generated content to state
    state.modules = content_skeleton.modules
    
    # Preserve research output for downstream agents
    state.research = content_skeleton.research
    
    print("Course skeleton generated successfully!")
    if state.research:
        print(f"   Research summary: {state.research.course_summary[:100]}...")
        print(f"   Learning objectives: {len(state.research.learning_objectives)}")
        print(f"   Key topics: {len(state.research.key_topics)}")
    
    # Save step snapshot if OutputManager is available
    output_mgr = _get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("index", state)
    
    return state


def generate_theories_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
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
    
    # Save step snapshot if OutputManager is available
    output_mgr = _get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("theories", updated_state)
    
    return updated_state


def generate_activities_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate activities for all sections in parallel"""
    print("Generating section activities in parallel...")
    
    updated_state = generate_all_section_activities(
        state,
        concurrency=state.config.activities_concurrency,
        max_retries=state.config.max_retries
    )
    
    print("All section activities generated successfully!")
    
    # Save step snapshot if OutputManager is available
    output_mgr = _get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("activities", updated_state)
    
    return updated_state


def calculate_metadata_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
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
            submodule.duration = round(len(submodule.sections) * 0.1, 1)
            
        # Calculate module duration
        module.duration = round(sum(sm.duration for sm in module.submodules), 1)
        
    print("Metadata calculation completed!")
    
    # Save step snapshot if OutputManager is available
    output_mgr = _get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("metadata", state)
    
    return state


def generate_html_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate HTML structure for all sections in parallel"""
    print("Generating HTML structures in parallel...")
    
    updated_state = generate_all_section_html(
        state,
        concurrency=state.config.html_concurrency,
        max_retries=state.config.max_retries
    )
    
    print("All HTML structures generated successfully!")
    
    # Save step snapshot if OutputManager is available
    output_mgr = _get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("html", updated_state)
    
    return updated_state


def generate_images_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
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
    
    # Save step snapshot if OutputManager is available
    output_mgr = _get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("images", updated_state)
    
    return updated_state


def generate_bibliography_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate book bibliography for the course (optional, controlled by config)"""
    if not state.config.generate_bibliography:
        print("📚 Bibliography generation disabled, skipping...")
        return state
    
    print("📚 Generating course bibliography...")
    
    bibliography = generate_course_bibliography(state)
    state.bibliography = bibliography
    
    print("Bibliography generation completed!")
    
    # Save step snapshot if OutputManager is available
    output_mgr = _get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("bibliography", state)
    
    return state


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
    graph.add_node("generate_bibliography", generate_bibliography_node)
    
    # Add edges for sequential execution
    graph.add_edge(START, "generate_index")
    graph.add_edge("generate_index", "generate_theories")
    graph.add_edge("generate_theories", "generate_activities")
    graph.add_edge("generate_activities", "calculate_metadata")
    graph.add_edge("calculate_metadata", "generate_html")
    graph.add_edge("generate_html", "generate_images")
    graph.add_edge("generate_images", "generate_bibliography")
    graph.add_edge("generate_bibliography", END)
    
    return graph.compile()


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a course")
    parser.add_argument("--total-pages", type=int, default=2, help="Total pages for the course (default: 2)")
    args = parser.parse_args()
    
    # Ensure you have set: export MISTRAL_API_KEY=your_key
    
    # Build the graph
    app = build_course_generation_graph()
    
    # Create initial CourseState with config and minimal content
    course_config = CourseConfig(
        title="Quantum Theory",
        text_llm_provider="mistral",  # LLM provider: mistral | gemini | groq | openai | deepseek
        web_search_provider="ddg",  # Web search provider: ddg | tavily | wikipedia
        total_pages=args.total_pages,  # Total pages for the course
        words_per_page=400,  # Target words per page
        language="Español",        
        description="",
        max_retries=8,
        concurrency=10,  # Number of concurrent section theory generations
        use_reflection=True,  # Enable reflection pattern for fact verification (default: False)
        num_reflection_queries=7,  # Number of verification queries per section (default: 3)
        # Research configuration
        enable_research=True,  # Enable research phase before index generation
        research_max_queries= 7,  # Number of search queries to generate
        research_max_results_per_query=5,  # Results per search query
        # Activities configuration
        activities_concurrency=30,  # Number of concurrent activity generations
        activity_selection_mode="deterministic",  # "random" or "deterministic"
        num_activities_per_section=1,  # Number of quiz activities (+ multiple_choice + multi_selection)
        # HTML configuration
        html_concurrency=15,  # Number of concurrent HTML generations
        select_html="LLM",  # "LLM" | "random"
        html_formats="paragraphs|accordion|tabs|carousel|flip|timeline|conversation",  # Pipe-separated list of available formats
        html_random_seed=42,  # Seed for deterministic random selection
        include_quotes_in_html=True,  # Include quote elements
        include_tables_in_html=True,  # Include table elements
        # Image generation configuration
        image_search_provider="bing",  # Image search provider: bing | freepik | ddg | google
        use_vision_ranking=False,  # Use vision LLM to rank images (slower but better quality)
        num_images_to_fetch=8,  # Number of images to fetch for ranking
        vision_llm_provider="pixtral",  # Vision LLM provider for image ranking
        image_concurrency=10,  # Number of image blocks to process in parallel
        imagetext2text_concurrency=5,  # Number of Pixtral vision LLM calls in parallel for image scoring
        vision_ranking_batch_size=8,  # Number of images per batch for Pixtral ranking calls
        target_audience=None,  # Target audience: None | "kids" | "general" | "advanced"
    )
    
    initial_state = CourseState(
        config=course_config,
        title=course_config.title,  # Initialize from config, can be refined during generation
        modules=[]  # Will be populated by skeleton generation
    )
    
    # Create OutputManager for this run (creates timestamped folder)
    output_mgr = OutputManager(title=course_config.title)
    print(f"📁 Output folder: {output_mgr.get_run_folder()}")
    
    # Run the graph with OutputManager passed via configurable
    result = app.invoke(
        initial_state,
        config={
            "run_name": f"{initial_state.title}",
            "configurable": {"output_manager": output_mgr}
        }
    )
    
    # Print the final course state
    print("\nWorkflow completed successfully!")
    
    # Extract the final state from LangGraph result (which is a dictionary)
    final_state = result if isinstance(result, CourseState) else CourseState.model_validate(result)
    
    # Save final outputs: course.json, course.html, and individual modules
    output_mgr.save_final(final_state)
    output_mgr.save_modules(final_state)
    
    # Print summary
    total_sections = sum(len(s.sections) for m in final_state.modules for s in m.submodules)
    print(f"\n📊 Course Summary:")
    print(f"   Title: {final_state.title}")
    print(f"   Modules: {len(final_state.modules)}")
    print(f"   Total Sections: {total_sections}")
    print(f"   Language: {final_state.language}")
    print(f"   HTML Selection: {final_state.config.select_html}")
    print(f"   Available Formats: {final_state.config.html_formats}")
    print(f"\n✅ All outputs saved to: {output_mgr.get_run_folder()}")
