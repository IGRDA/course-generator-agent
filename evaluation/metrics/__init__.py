"""Metrics modules for course evaluation.

Re-exports all public functions for convenient imports.
"""

from evaluation.metrics.embedding_metrics import (
    compute_section_similarity,
    compute_title_embedding_similarity,
)
from evaluation.metrics.nlp_metrics import (
    compute_readability,
    compute_repetition_metrics,
)
from evaluation.metrics.structure_metrics import (
    compute_title_uniqueness,
)

__all__ = [
    "compute_section_similarity",
    "compute_title_embedding_similarity",
    "compute_readability",
    "compute_repetition_metrics",
    "compute_title_uniqueness",
]

