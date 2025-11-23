from main.state import CourseState, CourseConfig
from agents.pdf_index_generator.agent import generate_course_state_from_pdf
from agents.section_theory_generator.agent import generate_all_section_theories
from agents.activities_generator.agent import generate_all_section_activities
from agents.html_formatter.agent import generate_all_section_html
from agents.html_formatter.exporter import export_to_html
from langgraph.graph import StateGraph, START, END

def generate_index_from_pdf_node(state: CourseState) -> CourseState:
    """Generate the course skeleton from PDF syllabus with empty theories while preserving config"""
    print("Generating course skeleton from PDF syllabus...")
    
    # Use config from the existing state
    config = state.config
    
    # Validate PDF path
    if not config.pdf_syllabus_path:
        raise ValueError("pdf_syllabus_path must be specified in CourseConfig")
    
    # Generate new course content skeleton from PDF (with empty theories)
    content_skeleton = generate_course_state_from_pdf(
        pdf_path=config.pdf_syllabus_path,
        total_pages=config.total_pages,
        language=config.language,
        max_retries=config.max_retries,
        words_per_page=config.words_per_page,
        provider=config.text_llm_provider
    )
    
    # Preserve original config, update only content fields
    state.title = content_skeleton.title
    state.modules = content_skeleton.modules
    
    print("Course skeleton generated successfully from PDF!")
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
    """Calculate indices, IDs, and durations for all course elements"""
    print("Calculating course metadata (IDs, Indices, Durations)...")
    
    for m_idx, module in enumerate(state.modules):
        module.index = m_idx + 1
        module.id = str(module.index)
        if not module.description:
            module.description = module.title
            
        for sm_idx, submodule in enumerate(module.submodules):
            submodule.index = sm_idx + 1
            submodule.id = f"{module.id}.{submodule.index}"
            if not submodule.description:
                submodule.description = submodule.title
            
            for s_idx, section in enumerate(submodule.sections):
                section.index = s_idx + 1
                section.id = f"{submodule.id}.{section.index}"
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
    
    # Add edges for sequential execution
    graph.add_edge(START, "generate_index_from_pdf")
    graph.add_edge("generate_index_from_pdf", "generate_theories")
    graph.add_edge("generate_theories", "generate_activities")
    graph.add_edge("generate_activities", "calculate_metadata")
    graph.add_edge("calculate_metadata", "generate_html")
    graph.add_edge("generate_html", END)
    
    return graph.compile()


if __name__ == "__main__":
    import os
    import json
    from datetime import datetime
    
    # Ensure you have set: export MISTRAL_API_KEY=your_key
    
    # Build the graph
    app = build_course_generation_graph_from_pdf()
    
    # Create initial CourseState with config and PDF path
    # Note: title and description are NOT hardcoded - they will be extracted from the PDF
    config = CourseConfig(
        pdf_syllabus_path="test.pdf",  # Path to your PDF syllabus
        text_llm_provider="mistral",  # LLM provider: mistral | gemini | groq | openai
        web_search_provider="ddg",  # Web search provider: ddg | tavily | wikipedia
        total_pages=200,  # Total pages for the course
        words_per_page=400,  # Target words per page
        language="Español",        
        max_retries=8,
        concurrency=5,  # Number of concurrent section theory generations
        use_reflection=True,  # Enable reflection pattern for fact verification (default: False)
        num_reflection_queries=5,  # Number of verification queries per section (default: 3)
        # Activities configuration
        activities_concurrency=4,  # Number of concurrent activity generations
        activity_selection_mode="deterministic",  # "random" or "deterministic"
        num_activities_per_section=2,  # Number of quiz activities (+ multiple_choice + multi_selection)
        # HTML configuration
        html_concurrency=4,  # Number of concurrent HTML generations
        html_format="tabs",  # "tabs" | "accordion" | "timeline" | "cards"
        include_quotes_in_html=True,  # Include quote elements
        include_tables_in_html=True  # Include table elements
    )
    
    initial_state = CourseState(
        config=config,
        title="",  # Will be extracted from PDF
        modules=[]  # Will be populated by PDF skeleton generation
    )
    
    # Run the graph
    result = app.invoke(
        initial_state,
        config={"run_name": "PDF Course Generation"}
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
    print(f"   Format: {final_state.config.html_format}")
    print(f"   Source: {final_state.config.pdf_syllabus_path}")

