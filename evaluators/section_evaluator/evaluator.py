"""Section evaluator for theory content evaluation."""

from typing import Dict, Any, List
from langsmith import traceable
from main.state import CourseState
from evaluators.base import BaseEvaluator, SectionScore
from .prompts import SECTION_EVALUATION_PROMPT, CORRECTION_PROMPT


class SectionEvaluator(BaseEvaluator):
    """
    Evaluates section theory content for accuracy and quality.
    Integrates NLP metrics for readability and repetition analysis.
    """
    
    @traceable(name="section_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """
        Evaluate all sections in the course.
        
        Args:
            course_state: The CourseState to evaluate
            
        Returns:
            Dictionary with scores and metrics for all sections
        """
        section_results = []
        all_theories = []
        
        for m_idx, module in enumerate(course_state.modules):
            for sm_idx, submodule in enumerate(module.submodules):
                for s_idx, section in enumerate(submodule.sections):
                    # Evaluate individual section
                    result = self._evaluate_section(
                        course_state=course_state,
                        module=module,
                        submodule=submodule,
                        section=section,
                        section_id=f"{m_idx+1}.{sm_idx+1}.{s_idx+1}"
                    )
                    section_results.append(result)
                    if section.theory:
                        all_theories.append(section.theory)
        
        # Compute NLP metrics across all sections
        nlp_metrics = self._compute_nlp_metrics(all_theories)
        
        # Compute schema checks
        schema_checks = self._run_schema_checks(course_state)
        
        # Compute averages
        accuracy_scores = [r["accuracy_score"] for r in section_results if r.get("accuracy_score")]
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        
        return {
            "evaluator": "section",
            "section_results": section_results,
            "summary": {
                "total_sections": len(section_results),
                "average_accuracy_score": avg_accuracy,
            },
            "nlp_metrics": nlp_metrics,
            "schema_checks": schema_checks
        }
    
    def _evaluate_section(
        self,
        course_state: CourseState,
        module,
        submodule,
        section,
        section_id: str
    ) -> Dict[str, Any]:
        """Evaluate a single section."""
        result = {
            "section_id": section_id,
            "section_title": section.title,
        }
        
        # Skip LLM evaluation if no theory content
        if not section.theory or len(section.theory.strip()) < 50:
            result["accuracy_score"] = None
            result["accuracy_reasoning"] = "Insufficient content for evaluation"
            result["has_content"] = False
            return result
        
        result["has_content"] = True
        
        # Run LLM-as-judge evaluation
        llm_score = self.evaluate_with_rubric(
            prompt=SECTION_EVALUATION_PROMPT,
            output_model=SectionScore,
            prompt_variables={
                "course_title": course_state.title,
                "module_title": module.title,
                "submodule_title": submodule.title,
                "section_title": section.title,
                "theory": section.theory[:3000],  # Limit content length
            },
            correction_prompt=CORRECTION_PROMPT
        )
        result["accuracy_score"] = llm_score.accuracy.score
        result["accuracy_reasoning"] = llm_score.accuracy.reasoning
        
        return result
    
    def _compute_nlp_metrics(self, theories: List[str]) -> Dict[str, Any]:
        """Compute NLP metrics for all theories combined."""
        if not theories:
            return {"error": "No content to analyze"}
        
        # Lazy import to handle optional dependency
        from evaluation.metrics.nlp_metrics import compute_readability, compute_repetition_metrics
        
        combined_text = "\n\n".join(theories)
        
        readability = compute_readability(combined_text)
        repetition = compute_repetition_metrics(combined_text)
        
        return {
            "readability": readability,
            "repetition": repetition
        }
    
    def _run_schema_checks(self, course_state: CourseState) -> Dict[str, Any]:
        """Run schema validation checks on section content."""
        sections_with_theory = 0
        empty_theories = []
        
        for m_idx, module in enumerate(course_state.modules):
            for sm_idx, submodule in enumerate(module.submodules):
                for s_idx, section in enumerate(submodule.sections):
                    if section.theory and len(section.theory.strip()) > 0:
                        sections_with_theory += 1
                    else:
                        empty_theories.append(f"{m_idx+1}.{sm_idx+1}.{s_idx+1}")
        
        total_sections = sum(
            len(sm.sections)
            for m in course_state.modules
            for sm in m.submodules
        )
        
        return {
            "total_sections": total_sections,
            "sections_with_theory": sections_with_theory,
            "empty_theories": empty_theories,
            "all_sections_have_theory": len(empty_theories) == 0
        }

