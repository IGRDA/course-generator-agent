"""Index evaluator for course structure evaluation."""

from typing import Dict, Any
from langsmith import traceable
from main.state import CourseState
from evaluators.base import BaseEvaluator, MultiCriteriaScore
from .prompts import INDEX_EVALUATION_PROMPT, CORRECTION_PROMPT


class IndexEvaluator(BaseEvaluator):
    """
    Evaluates the course index/structure for coverage, logical organization, and balance.
    """
    
    @traceable(name="index_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """
        Evaluate the course index.
        
        Args:
            course_state: The CourseState to evaluate
            
        Returns:
            Dictionary with scores and metrics
        """
        # Extract structure info
        num_modules = len(course_state.modules)
        num_submodules = sum(len(m.submodules) for m in course_state.modules)
        num_sections = sum(
            len(sm.sections) 
            for m in course_state.modules 
            for sm in m.submodules
        )
        
        # Build structure text for LLM
        course_structure = self._build_structure_text(course_state)
        
        # Run LLM-as-judge evaluation
        llm_scores = self.evaluate_with_rubric(
            prompt=INDEX_EVALUATION_PROMPT,
            output_model=MultiCriteriaScore,
            prompt_variables={
                "course_title": course_state.title,
                "num_modules": num_modules,
                "num_submodules": num_submodules,
                "num_sections": num_sections,
                "course_structure": course_structure,
            },
            correction_prompt=CORRECTION_PROMPT
        )
        
        # Schema validation checks
        schema_checks = self._run_schema_checks(course_state)
        
        # Compute average LLM score
        avg_llm_score = self.compute_average_score([
            llm_scores.coverage,
            llm_scores.structure,
            llm_scores.balance
        ])
        
        return {
            "evaluator": "index",
            "llm_scores": {
                "coverage": {
                    "score": llm_scores.coverage.score,
                    "reasoning": llm_scores.coverage.reasoning
                },
                "structure": {
                    "score": llm_scores.structure.score,
                    "reasoning": llm_scores.structure.reasoning
                },
                "balance": {
                    "score": llm_scores.balance.score,
                    "reasoning": llm_scores.balance.reasoning
                },
                "average": avg_llm_score
            },
            "schema_checks": schema_checks,
            "metadata": {
                "num_modules": num_modules,
                "num_submodules": num_submodules,
                "num_sections": num_sections
            }
        }
    
    def _build_structure_text(self, course_state: CourseState) -> str:
        """Build a text representation of the course structure for LLM evaluation."""
        lines = []
        for m_idx, module in enumerate(course_state.modules, 1):
            lines.append(f"Module {m_idx}: {module.title}")
            for sm_idx, submodule in enumerate(module.submodules, 1):
                lines.append(f"  {m_idx}.{sm_idx} {submodule.title}")
                for s_idx, section in enumerate(submodule.sections, 1):
                    lines.append(f"    {m_idx}.{sm_idx}.{s_idx} {section.title}")
        return "\n".join(lines)
    
    def _run_schema_checks(self, course_state: CourseState) -> Dict[str, Any]:
        """Run schema validation checks on the course structure."""
        checks = {
            "has_modules": len(course_state.modules) > 0,
            "has_title": bool(course_state.title),
            "all_modules_have_submodules": all(
                len(m.submodules) > 0 for m in course_state.modules
            ),
            "all_submodules_have_sections": all(
                len(sm.sections) > 0
                for m in course_state.modules
                for sm in m.submodules
            ),
            "all_sections_have_titles": all(
                bool(s.title)
                for m in course_state.modules
                for sm in m.submodules
                for s in sm.sections
            ),
        }
        checks["all_passed"] = all(checks.values())
        return checks

