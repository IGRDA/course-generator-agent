"""
People Generator Agent.

Generates relevant people for course modules using:
1. LLM suggests notable people based on module content
2. Wikipedia validates people and fetches images
3. Results are embedded directly in each Module
"""

import logging
from typing import Optional

from langchain_core.runnables import RunnableConfig

from main.state import (
    CourseState,
    PersonReference,
    Module,
)
from tools.peoplesearch import search_relevant_people, PersonResult

logger = logging.getLogger(__name__)


def _person_result_to_reference(result: PersonResult) -> PersonReference:
    """
    Convert PersonResult from tools to PersonReference for state embedding.
    
    Args:
        result: PersonResult from peoplesearch tool
        
    Returns:
        PersonReference for embedding in Module
    """
    return PersonReference(
        name=result.name,
        description=result.description,
        wikiUrl=result.wikiUrl,
        image=result.image,
    )


def _get_language_code(language: str) -> str:
    """
    Convert full language name to language code.
    
    Args:
        language: Full language name (e.g., "Español", "English")
        
    Returns:
        Language code (e.g., "es", "en")
    """
    language_map = {
        "english": "en",
        "spanish": "es",
        "español": "es",
        "french": "fr",
        "français": "fr",
        "german": "de",
        "deutsch": "de",
        "italian": "it",
        "italiano": "it",
        "portuguese": "pt",
        "português": "pt",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "russian": "ru",
        "arabic": "ar",
    }
    return language_map.get(language.lower(), "en")


def generate_module_people(
    module: Module,
    course_title: str,
    language: str,
    provider: str = "mistral",
    num_people: int = 3,
    concurrency: int = 5,
) -> list[PersonReference]:
    """
    Generate relevant people for a single module.
    
    Uses the module title and description to create a topic string,
    then searches for relevant notable people via Wikipedia.
    
    Args:
        module: Module to generate people for
        course_title: Course title for additional context
        language: Course language for descriptions
        provider: LLM provider for suggestions
        num_people: Number of people to find
        concurrency: Number of parallel API calls
        
    Returns:
        List of PersonReference to embed in module
    """
    # Create topic from module info
    topic = f"{module.title}"
    if module.description:
        topic += f": {module.description}"
    
    # Get language code
    lang_code = _get_language_code(language)
    
    print(f"      🔍 Topic: {topic[:60]}...")
    
    try:
        results = search_relevant_people(
            topic=topic,
            max_results=num_people,
            language=lang_code,
            llm_provider=provider,
            concurrency=concurrency,
        )
    except Exception as e:
        logger.error(f"People search failed: {e}")
        results = []
    
    # Convert to PersonReference objects
    people = [_person_result_to_reference(r) for r in results]
    
    return people


def generate_course_people(
    state: CourseState,
    provider: str | None = None,
    people_per_module: int | None = None,
    concurrency: int | None = None,
) -> CourseState:
    """
    Generate relevant people for entire course.
    
    Processes modules sequentially, embedding people directly in each module.
    
    Args:
        state: CourseState with modules
        provider: LLM provider (defaults to state.config.text_llm_provider)
        people_per_module: People per module (defaults to state.config.people_per_module)
        concurrency: Number of parallel API calls (defaults to state.config.people_concurrency)
        
    Returns:
        Updated CourseState with people embedded in modules
    """
    provider = provider or state.config.people_llm_provider or state.config.text_llm_provider
    people_per_module = people_per_module or state.config.people_per_module
    concurrency = concurrency or state.config.people_concurrency
    
    print(f"👥 Generating relevant people for {len(state.modules)} modules...")
    print(f"   Target: {people_per_module} people per module")
    print(f"   Provider: {provider}")
    
    for idx, module in enumerate(state.modules):
        print(f"\n   👤 Module {idx + 1}/{len(state.modules)}: {module.title}")
        
        people = generate_module_people(
            module=module,
            course_title=state.title,
            language=state.config.language,
            provider=provider,
            num_people=people_per_module,
            concurrency=concurrency,
        )
        
        # Embed in module
        module.relevant_people = people if people else None
        
        print(f"      ✓ Found {len(people)} people")
    
    total_people = sum(
        len(m.relevant_people) for m in state.modules if m.relevant_people
    )
    print(f"\n✅ People generation complete!")
    print(f"   Total people: {total_people}")
    
    return state


def generate_people_node(
    state: CourseState,
    config: Optional[RunnableConfig] = None,
) -> CourseState:
    """
    LangGraph node for people generation.
    
    Generates relevant people for all modules and embeds in state.
    Only runs if state.config.generate_people is True.
    
    Args:
        state: CourseState with modules
        config: LangGraph runtime config
        
    Returns:
        Updated CourseState with people embedded in modules
    """
    if not state.config.generate_people:
        print("👥 People generation disabled, skipping...")
        return state
    
    return generate_course_people(state)

