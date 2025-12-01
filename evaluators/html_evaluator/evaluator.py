"""HTML evaluator for section HTML structure evaluation."""

from typing import Dict, Any, List
from langsmith import traceable
from main.state import CourseState
from evaluators.base import BaseEvaluator, HtmlScore
from .prompts import HTML_EVALUATION_PROMPT, CORRECTION_PROMPT


# Valid HTML element types from state.py
VALID_ELEMENT_TYPES = {"p", "ul", "quote", "table", "paragraphs"}


class HtmlEvaluator(BaseEvaluator):
    """
    Evaluates HTML structure formatting quality and validity.
    """
    
    @traceable(name="html_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """
        Evaluate all HTML structures in the course.
        
        Args:
            course_state: The CourseState to evaluate
            
        Returns:
            Dictionary with scores and metrics
        """
        section_results = []
        element_type_usage: Dict[str, int] = {}
        
        for m_idx, module in enumerate(course_state.modules):
            for sm_idx, submodule in enumerate(module.submodules):
                for s_idx, section in enumerate(submodule.sections):
                    section_id = f"{m_idx+1}.{sm_idx+1}.{s_idx+1}"
                    
                    # Evaluate individual section HTML
                    result = self._evaluate_section_html(
                        section=section,
                        section_id=section_id
                    )
                    section_results.append(result)
                    
                    # Track element types
                    for elem_type, count in result.get("element_counts", {}).items():
                        element_type_usage[elem_type] = element_type_usage.get(elem_type, 0) + count
        
        # Schema checks
        schema_checks = self._run_schema_checks(course_state, element_type_usage)
        
        # Compute averages
        formatting_scores = [r["formatting_score"] for r in section_results if r.get("formatting_score")]
        avg_formatting = sum(formatting_scores) / len(formatting_scores) if formatting_scores else 0.0
        
        return {
            "evaluator": "html",
            "section_results": section_results,
            "summary": {
                "total_sections_evaluated": len(section_results),
                "average_formatting_score": avg_formatting,
                "element_type_usage": element_type_usage,
            },
            "schema_checks": schema_checks
        }
    
    def _evaluate_section_html(self, section, section_id: str) -> Dict[str, Any]:
        """Evaluate HTML for a single section."""
        result = {
            "section_id": section_id,
            "section_title": section.title,
        }
        
        # Check if section has HTML
        if not section.html or not section.html.theory:
            result["formatting_score"] = None
            result["formatting_reasoning"] = "No HTML structure found"
            result["has_html"] = False
            result["element_counts"] = {}
            return result
        
        result["has_html"] = True
        
        # Analyze HTML structure
        elements = section.html.theory
        element_counts = self._count_element_types(elements)
        result["element_counts"] = element_counts
        result["total_elements"] = sum(element_counts.values())
        
        # Build HTML structure text for LLM
        html_structure = self._build_html_structure_text(elements)
        element_types = ", ".join(element_counts.keys()) if element_counts else "None"
        
        # Run LLM-as-judge evaluation
        try:
            llm_score = self.evaluate_with_rubric(
                prompt=HTML_EVALUATION_PROMPT,
                output_model=HtmlScore,
                prompt_variables={
                    "section_title": section.title,
                    "html_structure": html_structure[:2000],  # Limit length
                    "total_elements": result["total_elements"],
                    "element_types": element_types,
                },
                correction_prompt=CORRECTION_PROMPT
            )
            result["formatting_score"] = llm_score.formatting.score
            result["formatting_reasoning"] = llm_score.formatting.reasoning
        except Exception as e:
            result["formatting_score"] = None
            result["formatting_reasoning"] = f"Evaluation failed: {str(e)}"
        
        return result
    
    def _count_element_types(self, elements: list) -> Dict[str, int]:
        """Count occurrences of each element type."""
        counts = {}
        for elem in elements:
            elem_type = elem.type
            counts[elem_type] = counts.get(elem_type, 0) + 1
        return counts
    
    def _build_html_structure_text(self, elements: list) -> str:
        """Build text representation of HTML structure for LLM."""
        lines = []
        for i, elem in enumerate(elements, 1):
            content_preview = str(elem.content)[:100] if elem.content else "empty"
            lines.append(f"{i}. <{elem.type}>: {content_preview}...")
        return "\n".join(lines) if lines else "No elements"
    
    def _run_schema_checks(
        self,
        course_state: CourseState,
        element_type_usage: Dict[str, int]
    ) -> Dict[str, Any]:
        """Run schema validation checks on HTML structures."""
        sections_with_html = 0
        invalid_element_types = []
        
        for module in course_state.modules:
            for submodule in module.submodules:
                for section in submodule.sections:
                    if section.html and section.html.theory:
                        sections_with_html += 1
                        # Check for invalid element types
                        for elem in section.html.theory:
                            if elem.type not in VALID_ELEMENT_TYPES:
                                invalid_element_types.append(elem.type)
        
        total_sections = sum(
            len(sm.sections)
            for m in course_state.modules
            for sm in m.submodules
        )
        
        # Check element type variety
        used_types = set(element_type_usage.keys())
        
        return {
            "total_sections": total_sections,
            "sections_with_html": sections_with_html,
            "element_type_variety": len(used_types),
            "all_element_types_valid": len(invalid_element_types) == 0,
            "invalid_element_types": list(set(invalid_element_types)),
            "all_sections_have_html": sections_with_html == total_sections
        }

