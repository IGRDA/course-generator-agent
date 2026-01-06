"""
Bibliography Generator Agent.

Generates book bibliographies for course modules using a hybrid approach:
1. LLM suggests relevant books based on module content
2. Open Library API validates and enriches book metadata
3. Citations are formatted in APA 7 style
4. Deduplication prevents repeated citations across modules
"""

import json
import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from main.state import (
    CourseState,
    BookReference,
    ModuleBibliography,
    CourseBibliography,
    Module,
)
from LLMs.text2text import create_text_llm, resolve_text_model_name
from tools.booksearch.openlibrary.client import (
    validate_book,
    search_books_by_title_author,
    BookResult,
)
from .prompts import book_suggestion_prompt, apa_formatting_prompt

logger = logging.getLogger(__name__)


def _format_apa_citation(book: BookReference) -> str:
    """
    Format a book reference in APA 7 style.
    
    Args:
        book: BookReference with book metadata
        
    Returns:
        Formatted APA 7 citation string
    """
    # Format authors
    if not book.authors:
        authors_str = "Unknown Author"
    elif len(book.authors) == 1:
        authors_str = book.authors[0]
    elif len(book.authors) == 2:
        authors_str = f"{book.authors[0]}, & {book.authors[1]}"
    else:
        # More than 2 authors: First author, ..., & Last author
        authors_str = f"{book.authors[0]}, "
        authors_str += ", ".join(book.authors[1:-1])
        authors_str += f", & {book.authors[-1]}"
    
    # Format year
    year_str = f"({book.year})" if book.year else "(n.d.)"
    
    # Format title (italicized in actual rendering, plain text here)
    title_str = book.title
    
    # Add edition if present
    if book.edition:
        title_str += f" ({book.edition})"
    
    # Format publisher
    publisher_str = book.publisher if book.publisher else ""
    
    # Build citation
    citation_parts = [authors_str, year_str, f"{title_str}."]
    if publisher_str:
        citation_parts.append(f"{publisher_str}.")
    
    # Add DOI or URL if present
    if book.doi:
        citation_parts.append(f"https://doi.org/{book.doi}")
    elif book.url:
        citation_parts.append(book.url)
    
    return " ".join(citation_parts)


def _book_result_to_reference(
    result: BookResult,
    relevance: str = "",
) -> BookReference:
    """
    Convert Open Library BookResult to BookReference model.
    
    Args:
        result: BookResult from Open Library search
        relevance: Optional relevance explanation
        
    Returns:
        BookReference with APA citation
    """
    book = BookReference(
        title=result["title"],
        authors=result["authors"],
        year=result["year"],
        publisher=result["publisher"],
        isbn=result["isbn"],
        isbn_13=result["isbn_13"],
        url=result["openlibrary_url"],
    )
    
    # Generate APA citation
    book.apa_citation = _format_apa_citation(book)
    
    return book


def _extract_module_topics(module: Module) -> str:
    """
    Extract topic list from module content for LLM context.
    
    Args:
        module: Module with submodules and sections
        
    Returns:
        Formatted string of topics
    """
    topics = []
    
    for submodule in module.submodules:
        topics.append(f"- {submodule.title}")
        for section in submodule.sections:
            topics.append(f"  - {section.title}")
            if section.description:
                topics.append(f"    ({section.description})")
    
    return "\n".join(topics)


def generate_module_bibliography(
    module: Module,
    course_title: str,
    language: str,
    provider: str = "mistral",
    num_books: int = 5,
    existing_keys: set[str] | None = None,
) -> tuple[ModuleBibliography, set[str]]:
    """
    Generate bibliography for a single module.
    
    Uses LLM to suggest books, then validates via Open Library API.
    Skips books that are already in existing_keys (deduplication).
    
    Args:
        module: Module to generate bibliography for
        course_title: Course title for context
        language: Course language
        provider: LLM provider
        num_books: Target number of books to include
        existing_keys: Set of dedup keys for books already cited
        
    Returns:
        Tuple of (ModuleBibliography, updated existing_keys set)
    """
    if existing_keys is None:
        existing_keys = set()
    
    # Create LLM
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.3}  # Some creativity for book suggestions
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Extract module topics
    module_topics = _extract_module_topics(module)
    
    # Build exclusion note if we have existing books
    exclusion_note = ""
    if existing_keys:
        exclusion_note = f"NOTE: Avoid suggesting books already recommended in previous modules. We have {len(existing_keys)} books already."
    
    # Get LLM suggestions
    chain = book_suggestion_prompt | llm | StrOutputParser()
    
    # Request more books than needed to account for validation failures and deduplication
    request_count = num_books * 2
    
    raw_suggestions = chain.invoke({
        "course_title": course_title,
        "module_title": module.title,
        "module_description": module.description,
        "module_topics": module_topics,
        "language": language,
        "num_books": request_count,
        "exclusion_note": exclusion_note,
    })
    
    # Parse LLM response
    try:
        # Clean markdown fences if present
        clean_response = raw_suggestions.strip()
        if clean_response.startswith("```"):
            clean_response = clean_response.split("```")[1]
            if clean_response.startswith("json"):
                clean_response = clean_response[4:]
        if clean_response.endswith("```"):
            clean_response = clean_response.rsplit("```", 1)[0]
        
        suggestions = json.loads(clean_response.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM book suggestions: {e}")
        suggestions = []
    
    # Validate books and build references
    validated_books: list[BookReference] = []
    new_keys: set[str] = set()
    
    for suggestion in suggestions:
        if len(validated_books) >= num_books:
            break
        
        title = suggestion.get("title", "")
        authors = suggestion.get("authors", [])
        year = suggestion.get("year")
        publisher = suggestion.get("publisher")
        relevance = suggestion.get("relevance", "")
        
        if not title:
            continue
        
        # Try to validate via Open Library
        author_query = authors[0] if authors else None
        validated = validate_book(title, author_query)
        
        if validated:
            book = _book_result_to_reference(validated, relevance)
        else:
            # Use LLM suggestion data if validation fails
            # Still include it but mark as unvalidated
            book = BookReference(
                title=title,
                authors=[_format_author_for_apa(a) for a in authors],
                year=year,
                publisher=publisher,
            )
            book.apa_citation = _format_apa_citation(book)
        
        # Check for deduplication
        dedup_key = book.get_dedup_key()
        if dedup_key in existing_keys or dedup_key in new_keys:
            logger.debug(f"Skipping duplicate book: {book.title}")
            continue
        
        validated_books.append(book)
        new_keys.add(dedup_key)
    
    # Create module bibliography
    module_bib = ModuleBibliography(
        module_index=module.index,
        module_title=module.title,
        books=validated_books,
    )
    
    # Update existing keys with new ones
    updated_keys = existing_keys | new_keys
    
    return module_bib, updated_keys


def _format_author_for_apa(author: str) -> str:
    """
    Format author name to APA style: Last, F. M.
    
    Args:
        author: Author name in various formats
        
    Returns:
        APA-formatted author name
    """
    # Handle already formatted names (Last, First)
    if ", " in author:
        return author
    
    # Handle "First Last" or "First Middle Last" format
    parts = author.strip().split()
    if len(parts) == 1:
        return parts[0]
    
    last = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1])
    return f"{last}, {initials}"


def generate_course_bibliography(
    state: CourseState,
    provider: str | None = None,
    books_per_module: int | None = None,
) -> CourseBibliography:
    """
    Generate bibliography for entire course with deduplication.
    
    Processes modules sequentially, tracking cited books to avoid repetition.
    
    Args:
        state: CourseState with modules
        provider: LLM provider (defaults to state.config.text_llm_provider)
        books_per_module: Books per module (defaults to state.config.bibliography_books_per_module)
        
    Returns:
        CourseBibliography with per-module and deduplicated master list
    """
    provider = provider or state.config.text_llm_provider
    books_per_module = books_per_module or state.config.bibliography_books_per_module
    
    print(f"📚 Generating bibliography for {len(state.modules)} modules...")
    print(f"   Target: {books_per_module} books per module")
    print(f"   Provider: {provider}")
    
    module_bibliographies: list[ModuleBibliography] = []
    all_books: list[BookReference] = []
    existing_keys: set[str] = set()
    
    for idx, module in enumerate(state.modules):
        print(f"\n   📖 Module {idx + 1}/{len(state.modules)}: {module.title}")
        
        module_bib, existing_keys = generate_module_bibliography(
            module=module,
            course_title=state.title,
            language=state.config.language,
            provider=provider,
            num_books=books_per_module,
            existing_keys=existing_keys,
        )
        
        module_bibliographies.append(module_bib)
        all_books.extend(module_bib.books)
        
        print(f"      ✓ Added {len(module_bib.books)} books")
    
    # Sort all_books alphabetically by first author
    all_books_sorted = sorted(
        all_books,
        key=lambda b: b.authors[0].lower() if b.authors else "zzz"
    )
    
    bibliography = CourseBibliography(
        modules=module_bibliographies,
        all_books=all_books_sorted,
    )
    
    print(f"\n✅ Bibliography complete!")
    print(f"   Total unique books: {len(all_books_sorted)}")
    
    return bibliography


def generate_bibliography_node(
    state: CourseState,
    config: Optional[RunnableConfig] = None,
) -> CourseState:
    """
    LangGraph node for bibliography generation.
    
    Generates bibliography for all modules and stores in state.
    Only runs if state.config.generate_bibliography is True.
    
    Args:
        state: CourseState with modules
        config: LangGraph runtime config
        
    Returns:
        Updated CourseState with bibliography
    """
    if not state.config.generate_bibliography:
        print("📚 Bibliography generation disabled, skipping...")
        return state
    
    bibliography = generate_course_bibliography(state)
    state.bibliography = bibliography
    
    return state
