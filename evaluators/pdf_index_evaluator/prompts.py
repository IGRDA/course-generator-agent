"""Prompts for PDF-based index evaluation - reuses index evaluation prompts."""

# PDF index evaluator uses the same prompts as index evaluator
# since it evaluates the same CourseState structure
from evaluators.index_evaluator.prompts import (
    INDEX_EVALUATION_PROMPT,
    CORRECTION_PROMPT
)

__all__ = ["INDEX_EVALUATION_PROMPT", "CORRECTION_PROMPT"]

