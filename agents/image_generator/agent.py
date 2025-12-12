from typing import Annotated, List
from operator import add
import os
from pydantic import BaseModel, Field
from main.state import CourseState, HtmlElement, ParagraphBlock
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from LLMs.text2text import create_text_llm, resolve_text_model_name
from tools.imagesearch.factory import create_image_search
from .prompts import image_query_prompt


# ---- State for individual section task ----
class ImageGenerationTask(BaseModel):
    """State for processing a single section's images"""
    course_state: CourseState
    module_idx: int
    submodule_idx: int
    section_idx: int
    section_title: str
    html_elements: List[HtmlElement]


# ---- State for aggregating results ----
class ImageGenerationState(BaseModel):
    """State for the image generation subgraph"""
    course_state: CourseState
    completed_sections: Annotated[list[dict], add] = Field(default_factory=list)
    total_sections: int = 0


def plan_image_generation(state: ImageGenerationState) -> dict:
    """Count the total sections that need image generation."""
    section_count = 0
    
    for module in state.course_state.modules:
        for submodule in module.submodules:
            section_count += len(submodule.sections)
    
    print(f"📋 Planning image generation for {section_count} sections")
    
    return {"total_sections": section_count}


def continue_to_images(state: ImageGenerationState) -> list[Send]:
    """Fan-out: Create a Send for each section to process in parallel."""
    sends = []
    
    for m_idx, module in enumerate(state.course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                # Only process sections that have HTML
                if section.html:
                    task = ImageGenerationTask(
                        course_state=state.course_state,
                        module_idx=m_idx,
                        submodule_idx=sm_idx,
                        section_idx=s_idx,
                        section_title=section.title,
                        html_elements=section.html
                    )
                    sends.append(Send("generate_section_images", task))
    
    return sends


def generate_section_images(state: ImageGenerationTask) -> dict:
    """
    Generate images for all interactive blocks in a single section.
    Uses LLM to create image queries and fetches images from configured provider.
    """
    # Extract context
    llm_provider = state.course_state.config.text_llm_provider
    image_provider = state.course_state.config.image_search_provider
    course_title = state.course_state.title
    
    # Create LLM
    model_name = resolve_text_model_name(llm_provider)
    llm_kwargs = {"temperature": 0.3}  # Slightly creative for better queries
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=llm_provider, **llm_kwargs)
    
    # Create query generation chain
    query_chain = image_query_prompt | llm | StrOutputParser()
    
    # Get the image search function from factory
    search_images = create_image_search(image_provider)
    
    # Interactive formats that need images
    interactive_formats = ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]
    
    images_added = 0
    
    # Process each HTML element
    for element in state.html_elements:
        # Check if this is an interactive format
        if element.type in interactive_formats:
            # Process each block in the content array
            if isinstance(element.content, list):
                for block in element.content:
                    # Ensure it's a ParagraphBlock (has title, icon, elements)
                    if isinstance(block, (dict, ParagraphBlock)):
                        # Convert dict to ParagraphBlock if needed
                        if isinstance(block, dict):
                            block_obj = ParagraphBlock(**block)
                        else:
                            block_obj = block
                        
                        # Extract first paragraph content for context
                        content_preview = ""
                        for elem in block_obj.elements:
                            if elem.type == "p":
                                content_preview = elem.content[:200]  # First 200 chars
                                break
                        
                        # Generate image query using LLM
                        try:
                            query = query_chain.invoke({
                                "course_title": course_title,
                                "block_title": block_obj.title,
                                "content_preview": content_preview
                            })
                            query = query.strip().replace('*', '').replace('"', '').strip()
                            
                            # Search for image using configured provider
                            results = search_images(query, max_results=1)
                            
                            # Check for errors
                            if not results or "error" in results[0]:
                                error_msg = results[0].get("error", "Unknown error") if results else "No results"
                                raise Exception(f"Image search ({image_provider}) failed for query '{query}': {error_msg}")
                            
                            # Get image URL
                            image_url = results[0]["url"]
                            
                            # Add image to block
                            block_obj.image = {
                                "type": "img",
                                "query": query,
                                "content": image_url
                            }
                            
                            images_added += 1
                            
                        except Exception as e:
                            # Fail loudly as specified
                            print(f"❌ Error generating image for block '{block_obj.title}': {str(e)}")
                            raise
    
    print(f"✓ Generated {images_added} images for Module {state.module_idx+1}, "
          f"Submodule {state.submodule_idx+1}, Section {state.section_idx+1}")
    
    # Return the completed section info
    return {
        "completed_sections": [{
            "module_idx": state.module_idx,
            "submodule_idx": state.submodule_idx,
            "section_idx": state.section_idx,
            "html_elements": state.html_elements
        }]
    }


def reduce_images(state: ImageGenerationState) -> dict:
    """Fan-in: Aggregate all generated images back into the course state."""
    print(f"📦 Reducing {len(state.completed_sections)} sections with images")
    
    # Update course state with all generated images
    for section_info in state.completed_sections:
        m_idx = section_info["module_idx"]
        sm_idx = section_info["submodule_idx"]
        s_idx = section_info["section_idx"]
        
        section = state.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx]
        section.html = section_info["html_elements"]
    
    print(f"✅ All {state.total_sections} section images generated successfully!")
    
    return {"course_state": state.course_state}


def build_image_generation_graph(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Build the image generation subgraph using Send for dynamic parallelization.
    
    Args:
        max_retries: Number of retries for each section generation
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
    """
    graph = StateGraph(ImageGenerationState)
    
    # Configure retry policy with exponential backoff
    retry_policy = RetryPolicy(
        max_attempts=max_retries,
        initial_interval=initial_delay,
        backoff_factor=backoff_factor,
        max_interval=60.0
    )
    
    # Add nodes
    graph.add_node("plan_image_generation", plan_image_generation)
    graph.add_node("generate_section_images", generate_section_images, retry=retry_policy)
    graph.add_node("reduce_images", reduce_images)
    
    # Add edges
    graph.add_edge(START, "plan_image_generation")
    graph.add_conditional_edges("plan_image_generation", continue_to_images, ["generate_section_images"])
    graph.add_edge("generate_section_images", "reduce_images")
    graph.add_edge("reduce_images", END)
    
    return graph.compile()


def generate_all_section_images(
    course_state: CourseState,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> CourseState:
    """
    Main function to generate images for all section HTML blocks using LangGraph Send pattern.
    
    Args:
        course_state: CourseState with HTML structures filled
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
        
    Returns:
        Updated CourseState with all images added to HTML blocks
    """
    image_provider = course_state.config.image_search_provider
    print(f"🚀 Starting parallel image generation with provider={image_provider}, "
          f"max_retries={max_retries}, initial_delay={initial_delay}s, backoff_factor={backoff_factor}")
    
    # Build the graph with retry configuration
    graph = build_image_generation_graph(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )
    
    # Initialize state
    initial_state = ImageGenerationState(course_state=course_state)
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    return result["course_state"]


if __name__ == "__main__":
    import json
    from main.state import CourseConfig
    
    INPUT_FILE = "/Users/inaki/Documents/Personal/course-generator-agent/output/Chess_masterclass_20251211_172927.json"
    # Generate OUTPUT_FILE from INPUT_FILE with _images suffix
    base, ext = os.path.splitext(INPUT_FILE)
    OUTPUT_FILE = f"{base}_images{ext}"
    
    # Hardcoded defaults - change these to override config values
    DEFAULT_LLM_PROVIDER = "mistral"
    DEFAULT_LANGUAGE = "English"
    DEFAULT_IMAGE_PROVIDER = "freepik"  # Options: bing | freepik | ddg
    
    print("="*60)
    print("Image Generator - Standalone Mode")
    print("="*60)
    print(f"Reading from: {INPUT_FILE}")
    
    # Load input JSON
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found")
        print(f"   Please create an {INPUT_FILE} file with CourseState JSON structure")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {INPUT_FILE}: {str(e)}")
        exit(1)
    
    # Parse as CourseState
    try:
        course_state = CourseState.model_validate(data)
        
        # Apply defaults if config is incomplete
        if not course_state.config:
            course_state.config = CourseConfig(
                title=course_state.title or "Untitled Course",
                text_llm_provider=DEFAULT_LLM_PROVIDER,
                image_search_provider=DEFAULT_IMAGE_PROVIDER,
                total_pages=10
            )
        else:
            # Override image provider with hardcoded default
            course_state.config.image_search_provider = DEFAULT_IMAGE_PROVIDER
        
        if not course_state.language:
            course_state.language = DEFAULT_LANGUAGE
            
    except Exception as e:
        print(f"❌ Error: Failed to parse CourseState: {str(e)}")
        exit(1)
    
    # Count sections with HTML
    html_sections = 0
    total_sections = 0
    for module in course_state.modules:
        for submodule in module.submodules:
            for section in submodule.sections:
                total_sections += 1
                if section.html:
                    html_sections += 1
    
    print(f"📊 Course: {course_state.title}")
    print(f"   Language: {course_state.language}")
    print(f"   Modules: {len(course_state.modules)}")
    print(f"   Sections: {total_sections}")
    print(f"   Sections with HTML: {html_sections}")
    
    if html_sections == 0:
        print("⚠️  Warning: No sections have HTML content to add images to")
        print("   Exiting without changes")
        exit(0)
    
    # Generate images
    print(f"\n🚀 Starting image generation...")
    print(f"   Using LLM: {course_state.config.text_llm_provider}")
    print(f"   Image provider: {course_state.config.image_search_provider}")
    
    try:
        updated_state = generate_all_section_images(
            course_state,
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0
        )
    except Exception as e:
        print(f"\n❌ Error during image generation: {str(e)}")
        print("   No output file created")
        exit(1)
    
    # Save output
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(updated_state.model_dump_json(indent=2, by_alias=True))
        print(f"✅ Success! Images added to {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error: Failed to save output: {str(e)}")
        exit(1)
    
    # Print all image queries
    print("\n" + "="*60)
    print("📝 All Generated Image Queries:")
    print("="*60)
    
    query_count = 0
    for m_idx, module in enumerate(updated_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                if section.html:
                    section_has_images = False
                    for element in section.html:
                        if element.type in ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]:
                            if isinstance(element.content, list):
                                for block in element.content:
                                    if isinstance(block, (dict, ParagraphBlock)):
                                        block_obj = block if isinstance(block, ParagraphBlock) else ParagraphBlock(**block)
                                        if block_obj.image and "query" in block_obj.image:
                                            if not section_has_images:
                                                print(f"\n📍 Module {m_idx+1}.{sm_idx+1}.{s_idx+1}: {section.title}")
                                                section_has_images = True
                                            query_count += 1
                                            print(f"   {query_count}. [{block_obj.title}] → \"{block_obj.image['query']}\"")
    
    print(f"\n✨ Total queries generated: {query_count}")
    print("="*60)
