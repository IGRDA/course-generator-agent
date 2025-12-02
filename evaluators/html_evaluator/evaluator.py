"""HTML evaluator for section HTML structure evaluation."""

from typing import Dict, Any
from langsmith import traceable
from main.state import CourseState
from evaluators.base import BaseEvaluator, SingleCriteriaScore
from .prompts import HTML_EVALUATION_PROMPT, CORRECTION_PROMPT


# Valid HTML element types from state.py
VALID_ELEMENT_TYPES = {"p", "ul", "quote", "table", "paragraphs"}


class HtmlEvaluator(BaseEvaluator):
    """Evaluates HTML structure formatting quality and validity."""
    
    @traceable(name="html_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """Evaluate all HTML structures in the course."""
        section_results = []
        element_type_usage: Dict[str, int] = {}
        
        for section_id, _, _, section in self.iter_sections(course_state):
            result = self._evaluate_section_html(section, section_id)
            section_results.append(result)
            
            for elem_type, count in result.get("element_counts", {}).items():
                element_type_usage[elem_type] = element_type_usage.get(elem_type, 0) + count
        
        formatting_scores = [r["formatting_score"] for r in section_results if r.get("formatting_score")]
        
        return {
            "evaluator": "html",
            "section_results": section_results,
            "summary": {
                "total_sections_evaluated": len(section_results),
                "average_formatting_score": sum(formatting_scores) / len(formatting_scores) if formatting_scores else 0.0,
                "element_type_usage": element_type_usage,
            },
            "schema_checks": self._run_schema_checks(course_state, element_type_usage)
        }
    
    def _evaluate_section_html(self, section, section_id: str) -> Dict[str, Any]:
        """Evaluate HTML for a single section."""
        result = {"section_id": section_id, "section_title": section.title, "element_counts": {}}
        
        if not section.html or not section.html.theory:
            result.update(formatting_score=None, formatting_reasoning="No HTML structure found", has_html=False)
            return result
        
        result["has_html"] = True
        elements = section.html.theory
        element_counts = {elem.type: 0 for elem in elements}
        for elem in elements:
            element_counts[elem.type] += 1
        
        result["element_counts"] = element_counts
        result["total_elements"] = sum(element_counts.values())
        
        html_structure = "\n".join(
            f"{i}. <{e.type}>: {str(e.content)[:100]}..." 
            for i, e in enumerate(elements, 1)
        ) or "No elements"
        
        llm_score = self.evaluate_with_rubric(
            prompt=HTML_EVALUATION_PROMPT,
            output_model=SingleCriteriaScore,
            prompt_variables={
                "section_title": section.title,
                "html_structure": html_structure[:2000],
                "total_elements": result["total_elements"],
                "element_types": ", ".join(element_counts.keys()) or "None",
            },
            correction_prompt=CORRECTION_PROMPT
        )
        result["formatting_score"] = llm_score.score.score
        result["formatting_reasoning"] = llm_score.score.reasoning
        return result
    
    def _run_schema_checks(self, course_state, element_usage: dict) -> Dict[str, Any]:
        """Run schema validation checks on HTML structures."""
        sections_with_html = 0
        invalid_types = set()
        
        for _, _, _, section in self.iter_sections(course_state):
            if section.html and section.html.theory:
                sections_with_html += 1
                for elem in section.html.theory:
                    if elem.type not in VALID_ELEMENT_TYPES:
                        invalid_types.add(elem.type)
        
        total = self.count_sections(course_state)
        return {
            "total_sections": total,
            "sections_with_html": sections_with_html,
            "element_type_variety": len(set(element_usage.keys())),
            "all_element_types_valid": len(invalid_types) == 0,
            "invalid_element_types": list(invalid_types),
            "all_sections_have_html": sections_with_html == total
        }
