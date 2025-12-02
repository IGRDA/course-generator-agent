"""Section evaluator for theory content evaluation."""

from typing import Dict, Any, List
from langsmith import traceable
from main.state import CourseState
from evaluators.base import BaseEvaluator, SingleCriteriaScore
from .prompts import SECTION_EVALUATION_PROMPT, CORRECTION_PROMPT


class SectionEvaluator(BaseEvaluator):
    """Evaluates section theory content for accuracy and quality."""
    
    @traceable(name="section_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """Evaluate all sections in the course."""
        section_results = []
        all_theories = []
        
        for section_id, module, submodule, section in self.iter_sections(course_state):
            result = self._evaluate_section(course_state, module, submodule, section, section_id)
            section_results.append(result)
            if section.theory:
                all_theories.append(section.theory)
        
        accuracy_scores = [r["accuracy_score"] for r in section_results if r.get("accuracy_score")]
        
        return {
            "evaluator": "section",
            "section_results": section_results,
            "summary": {
                "total_sections": len(section_results),
                "average_accuracy_score": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0,
            },
            "nlp_metrics": self._compute_nlp_metrics(all_theories),
            "schema_checks": self._run_schema_checks(course_state)
        }
    
    def _evaluate_section(self, course_state, module, submodule, section, section_id: str) -> Dict[str, Any]:
        """Evaluate a single section."""
        result = {"section_id": section_id, "section_title": section.title}
        
        if not section.theory or len(section.theory.strip()) < 50:
            result.update(accuracy_score=None, accuracy_reasoning="Insufficient content", has_content=False)
            return result
        
        result["has_content"] = True
        llm_score = self.evaluate_with_rubric(
            prompt=SECTION_EVALUATION_PROMPT,
            output_model=SingleCriteriaScore,
            prompt_variables={
                "course_title": course_state.title,
                "module_title": module.title,
                "submodule_title": submodule.title,
                "section_title": section.title,
                "theory": section.theory[:3000],
            },
            correction_prompt=CORRECTION_PROMPT
        )
        result["accuracy_score"] = llm_score.score.score
        result["accuracy_reasoning"] = llm_score.score.reasoning
        return result
    
    def _compute_nlp_metrics(self, theories: List[str]) -> Dict[str, Any]:
        """Compute NLP metrics for all theories combined."""
        if not theories:
            return {"error": "No content to analyze"}
        
        from evaluation.metrics.nlp_metrics import compute_readability, compute_repetition_metrics
        combined_text = "\n\n".join(theories)
        return {
            "readability": compute_readability(combined_text),
            "repetition": compute_repetition_metrics(combined_text)
        }
    
    def _run_schema_checks(self, course_state: CourseState) -> Dict[str, Any]:
        """Run schema validation checks on section content."""
        empty_theories = []
        sections_with_theory = 0
        
        for section_id, _, _, section in self.iter_sections(course_state):
            if section.theory and len(section.theory.strip()) > 0:
                sections_with_theory += 1
            else:
                empty_theories.append(section_id)
        
        total = self.count_sections(course_state)
        return {
            "total_sections": total,
            "sections_with_theory": sections_with_theory,
            "empty_theories": empty_theories,
            "all_sections_have_theory": len(empty_theories) == 0
        }
