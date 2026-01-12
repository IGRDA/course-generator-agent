"""
Factory for creating book search client instances.

This module provides a unified interface for book search across
different providers (currently Open Library).
"""

from typing import Callable

from .openlibrary.client import search_books as openlibrary_search
from .openlibrary.client import BookResult

BookSearchFunc = Callable[[str, int], list[BookResult]]

BOOK_SEARCH_PROVIDERS: dict[str, BookSearchFunc] = {
    "openlibrary": openlibrary_search,
}


def available_book_search_providers() -> list[str]:
    """Return the list of registered book search providers."""
    return sorted(BOOK_SEARCH_PROVIDERS.keys())


def create_book_search(provider: str = "openlibrary") -> BookSearchFunc:
    """
    Get book search function for the specified provider.
    
    Args:
        provider: Book search provider name (openlibrary).
        
    Returns:
        A book search function that accepts (query: str, max_results: int).
        
    Raises:
        ValueError: If provider is not supported.
    """
    if not provider:
        provider = "openlibrary"
    
    key = provider.lower()
    try:
        return BOOK_SEARCH_PROVIDERS[key]
    except KeyError as exc:
        available = ", ".join(available_book_search_providers())
        raise ValueError(
            f"Unsupported book search provider '{provider}'. "
            f"Available providers: {available}"
        ) from exc
