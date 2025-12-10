"""PDF Index evaluator - extends IndexEvaluator for PDF-generated indices."""

from typing import Dict, Any
from langsmith import traceable
from main.state import CourseState
from evaluators.index_evaluator.evaluator import IndexEvaluator


class PdfIndexEvaluator(IndexEvaluator):
    """
    Evaluates course indices generated from PDF syllabi.
    Inherits from IndexEvaluator and uses the same evaluation logic.
    """
    
    @traceable(name="pdf_index_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """
        Evaluate the PDF-generated course index.
        
        Args:
            course_state: The CourseState to evaluate
            
        Returns:
            Dictionary with scores and metrics
        """
        # Use parent class evaluation
        result = super().evaluate(course_state)
        
        # Update evaluator name to reflect PDF source
        result["evaluator"] = "pdf_index"
        
        return result

