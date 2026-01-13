"""
Shared workflow node functions.

This module provides reusable node implementations for LangGraph workflows.
All workflow files should import nodes from here to avoid duplication.
"""

from .utils import get_output_manager
from .index import generate_index_node, generate_index_from_pdf_node
from .content import generate_theories_node, generate_activities_node
from .formatting import generate_html_node, generate_images_node
from .metadata import calculate_metadata_node
from .extras import generate_bibliography_node, generate_podcasts_node, generate_pdf_book_node

__all__ = [
    # Utils
    "get_output_manager",
    # Index generation
    "generate_index_node",
    "generate_index_from_pdf_node",
    # Content generation
    "generate_theories_node",
    "generate_activities_node",
    # Formatting
    "generate_html_node",
    "generate_images_node",
    # Metadata
    "calculate_metadata_node",
    # Extras
    "generate_bibliography_node",
    "generate_podcasts_node",
    "generate_pdf_book_node",
]

