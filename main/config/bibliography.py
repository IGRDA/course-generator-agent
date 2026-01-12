"""Bibliography generation configuration."""

from pydantic import BaseModel, Field


class BibliographyConfig(BaseModel):
    """Configuration for book bibliography generation."""
    
    enabled: bool = Field(
        default=False,
        description="Generate book bibliography for the course"
    )
    books_per_module: int = Field(
        default=5,
        description="Number of books to recommend per module"
    )
    search_provider: str = Field(
        default="openlibrary",
        description="Book search provider for validation (openlibrary)"
    )

