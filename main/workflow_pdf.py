"""
PDF Course Generation Workflow

Generates a complete course from a PDF syllabus using:
1. PDF extraction (modules, submodules, sections)
2. Research for enrichment
3. Theory generation
4. Activities generation
5. HTML formatting
"""

from typing import Optional
from langchain_core.runnables import RunnableConfig

from main.state import CourseState, CourseConfig
from main.output_manager import OutputManager
from agents.pdf_index_generator.agent import generate_course_state_from_pdf
from agents.section_theory_generator.agent import generate_all_section_theories
from agents.activities_generator.agent import generate_all_section_activities
from agents.html_formatter.agent import generate_all_section_html
from agents.image_generator.agent import generate_all_section_images
from langgraph.graph import StateGraph, START, END


def _get_output_manager(config: Optional[RunnableConfig]) -> Optional[OutputManager]:
    """Extract OutputManager from LangGraph config if present."""
    if config is None:
        return None
    return config.get("configurable", {}).get("output_manager")


def generate_index_from_pdf_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate the course skeleton from PDF syllabus with empty theories while preserving config"""
    print("Generating course skeleton from PDF syllabus...")
    
    # Use config from the existing state
    course_config = state.config
    
    # Validate PDF path
    if not course_config.pdf_syllabus_path:
        raise ValueError("pdf_syllabus_path must be specified in CourseConfig")
    
    # Generate new course content skeleton from PDF (with empty theories)
    content_skeleton = generate_course_state_from_pdf(
        pdf_path=course_config.pdf_syllabus_path,
        total_pages=course_config.total_pages,
        language=course_config.language,
        max_retries=course_config.max_retries,
        words_per_page=course_config.words_per_page,
        provider=course_config.text_llm_provider,
        # Research configuration
        enable_research=course_config.enable_research,
        web_search_provider=course_config.web_search_provider,
        research_max_queries=course_config.research_max_queries,
        research_max_results_per_query=course_config.research_max_results_per_query,
    )
    
    # Transfer generated content to state
    state.title = content_skeleton.title
    state.modules = content_skeleton.modules
    
    # Preserve research output for downstream agents
    state.research = content_skeleton.research
    
    print("Course skeleton generated successfully from PDF!")
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
            
        # If module has duration from PDF, keep it; otherwise calculate
        if module.duration == 0.0:
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


# Build the graph
def build_course_generation_graph_from_pdf():
    """Build and return the course generation graph that uses PDF syllabus"""
    graph = StateGraph(CourseState)
    
    # Add nodes for complete course generation pipeline
    graph.add_node("generate_index_from_pdf", generate_index_from_pdf_node)
    graph.add_node("generate_theories", generate_theories_node)
    graph.add_node("generate_activities", generate_activities_node)
    graph.add_node("calculate_metadata", calculate_metadata_node)
    graph.add_node("generate_html", generate_html_node)
    graph.add_node("generate_images", generate_images_node)
    
    # Add edges for sequential execution
    graph.add_edge(START, "generate_index_from_pdf")
    graph.add_edge("generate_index_from_pdf", "generate_theories")
    graph.add_edge("generate_theories", "generate_activities")
    graph.add_edge("generate_activities", "calculate_metadata")
    graph.add_edge("calculate_metadata", "generate_html")
    graph.add_edge("generate_html", "generate_images")
    graph.add_edge("generate_images", END)
    
    return graph.compile()


if __name__ == "__main__":
    import os
    from datetime import datetime
    
    # Build the graph
    app = build_course_generation_graph_from_pdf()
    
    # Create initial CourseState with config and PDF path
    # Note: title will be extracted from the PDF
    config = CourseConfig(
        pdf_syllabus_path="test.pdf",  # Path to your PDF syllabus
        text_llm_provider="mistral",  # LLM provider: mistral | gemini | groq | openai
        web_search_provider="ddg",  # Web search provider: ddg | tavily | wikipedia
        total_pages=200,  # Total pages for the course
        words_per_page=400,  # Target words per page
        language="Español",
        max_retries=8,
        concurrency=5,  # Number of concurrent section theory generations
        use_reflection=True,  # Enable reflection pattern for fact verification
        num_reflection_queries=5,  # Number of verification queries per section
        # Research configuration
        enable_research=True,  # Enable research phase for enrichment
        research_max_queries=5,  # Number of search queries to generate
        research_max_results_per_query=3,  # Results per search query
        # Activities configuration
        activities_concurrency=4,  # Number of concurrent activity generations
        activity_selection_mode="deterministic",  # "random" or "deterministic"
        num_activities_per_section=2,  # Number of quiz activities
        # HTML configuration
        html_concurrency=4,  # Number of concurrent HTML generations
        select_html="LLM",  # "LLM" | "random"
        html_formats="paragraphs|accordion|tabs|carousel|flip|timeline|conversation",
        include_quotes_in_html=True,  # Include quote elements
        include_tables_in_html=True,  # Include table elements
        # Image generation configuration
        image_search_provider="bing",  # Image search provider
        use_vision_ranking=False,  # Use vision LLM to rank images
        num_images_to_fetch=8,  # Number of images to fetch for ranking
        vision_llm_provider="pixtral",  # Vision LLM provider
        image_concurrency=10,  # Number of image blocks to process in parallel
    )
    
    initial_state = CourseState(
        config=config,
        title="",  # Will be extracted from PDF
        modules=[]  # Will be populated by PDF skeleton generation
    )
    
    # Create OutputManager for this run
    output_mgr = OutputManager(title="PDF_Course")
    print(f"📁 Output folder: {output_mgr.get_run_folder()}")
    
    # Run the graph with OutputManager passed via configurable
    result = app.invoke(
        initial_state,
        config={
            "run_name": "PDF Course Generation",
            "configurable": {"output_manager": output_mgr}
        }
    )
    
    # Print the final course state
    print("\nWorkflow completed successfully!")
    
    # Extract the final state from LangGraph result
    final_state = result if isinstance(result, CourseState) else CourseState.model_validate(result)
    
    # Save final outputs
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
    print(f"   Source: {final_state.config.pdf_syllabus_path}")
    print(f"\n✅ All outputs saved to: {output_mgr.get_run_folder()}")
