"""Overall evaluator for complete course evaluation."""

from typing import Dict, Any, List
from langsmith import traceable
from main.state import CourseState
from evaluators.base import BaseEvaluator, OverallScore
from .prompts import OVERALL_EVALUATION_PROMPT, CORRECTION_PROMPT


class OverallEvaluator(BaseEvaluator):
    """
    Evaluates the complete course for coherence, completeness, and content similarity.
    """
    
    @traceable(name="overall_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """
        Evaluate the overall course.
        
        Args:
            course_state: The CourseState to evaluate
            
        Returns:
            Dictionary with scores and metrics
        """
        # Run LLM coherence evaluation
        coherence_result = self._evaluate_coherence(course_state)
        
        # Run completeness check
        completeness = self._check_completeness(course_state)
        
        # Run structure metrics (title similarity, hierarchy)
        structure_metrics = self._compute_structure_metrics(course_state)
        
        # Run embedding similarity (if available)
        embedding_metrics = self._compute_embedding_metrics(course_state)
        
        return {
            "evaluator": "overall",
            "coherence": coherence_result,
            "completeness": completeness,
            "structure_metrics": structure_metrics,
            "embedding_metrics": embedding_metrics,
        }
    
    def _evaluate_coherence(self, course_state: CourseState) -> Dict[str, Any]:
        """Evaluate course coherence using LLM."""
        # Build course overview
        course_overview = self._build_course_overview(course_state)
        
        # Build section titles sample
        section_titles = self._get_section_titles_sample(course_state)
        
        llm_score = self.evaluate_with_rubric(
            prompt=OVERALL_EVALUATION_PROMPT,
            output_model=OverallScore,
            prompt_variables={
                "course_title": course_state.title,
                "language": course_state.language,
                "course_overview": course_overview,
                "section_titles_sample": section_titles,
            },
            correction_prompt=CORRECTION_PROMPT
        )
        return {
            "score": llm_score.coherence.score,
            "reasoning": llm_score.coherence.reasoning
        }
    
    def _build_course_overview(self, course_state: CourseState) -> str:
        """Build a text overview of the course structure."""
        lines = [f"Course: {course_state.title}"]
        for m_idx, module in enumerate(course_state.modules, 1):
            lines.append(f"\nModule {m_idx}: {module.title}")
            for sm_idx, submodule in enumerate(module.submodules, 1):
                lines.append(f"  {m_idx}.{sm_idx} {submodule.title} ({len(submodule.sections)} sections)")
        return "\n".join(lines)
    
    def _get_section_titles_sample(self, course_state: CourseState, max_titles: int = 20) -> str:
        """Get a sample of section titles showing the flow."""
        titles = []
        for m_idx, module in enumerate(course_state.modules, 1):
            for sm_idx, submodule in enumerate(module.submodules, 1):
                for s_idx, section in enumerate(submodule.sections, 1):
                    titles.append(f"{m_idx}.{sm_idx}.{s_idx} {section.title}")
                    if len(titles) >= max_titles:
                        break
                if len(titles) >= max_titles:
                    break
            if len(titles) >= max_titles:
                break
        
        if len(titles) == max_titles:
            titles.append("...")
        
        return "\n".join(titles)
    
    def _check_completeness(self, course_state: CourseState) -> Dict[str, Any]:
        """Check if all required fields are populated."""
        total_sections = 0
        sections_with_theory = 0
        sections_with_activities = 0
        sections_with_html = 0
        
        for module in course_state.modules:
            for submodule in module.submodules:
                for section in submodule.sections:
                    total_sections += 1
                    if section.theory and len(section.theory.strip()) > 0:
                        sections_with_theory += 1
                    if section.other_elements and section.other_elements.activities:
                        sections_with_activities += 1
                    if section.html and section.html.theory:
                        sections_with_html += 1
        
        return {
            "total_sections": total_sections,
            "sections_with_theory": sections_with_theory,
            "sections_with_activities": sections_with_activities,
            "sections_with_html": sections_with_html,
            "theory_completeness": sections_with_theory / total_sections if total_sections > 0 else 0,
            "activities_completeness": sections_with_activities / total_sections if total_sections > 0 else 0,
            "html_completeness": sections_with_html / total_sections if total_sections > 0 else 0,
            "is_complete": (
                sections_with_theory == total_sections and
                sections_with_activities == total_sections and
                sections_with_html == total_sections
            )
        }
    
    def _compute_structure_metrics(self, course_state: CourseState) -> Dict[str, Any]:
        """Compute structure-based metrics using dedicated module."""
        from evaluation.structure_metrics import (
            compute_title_uniqueness,
            compute_hierarchy_balance,
            find_duplicate_titles
        )
        
        title_uniqueness = compute_title_uniqueness(course_state)
        hierarchy_balance = compute_hierarchy_balance(course_state)
        duplicates = find_duplicate_titles(course_state)
        
        return {
            "title_uniqueness": title_uniqueness,
            "hierarchy_balance": hierarchy_balance,
            "duplicate_titles": duplicates
        }
    
    def _compute_embedding_metrics(self, course_state: CourseState) -> Dict[str, Any]:
        """Compute embedding-based similarity metrics."""
        from evaluation.embedding_metrics import compute_section_similarity
        
        # Extract all section theories
        sections_data = []
        for m_idx, module in enumerate(course_state.modules):
            for sm_idx, submodule in enumerate(module.submodules):
                for s_idx, section in enumerate(submodule.sections):
                    if section.theory and len(section.theory.strip()) > 50:
                        sections_data.append({
                            "id": f"{m_idx+1}.{sm_idx+1}.{s_idx+1}",
                            "title": section.title,
                            "text": section.theory
                        })
        
        if len(sections_data) < 2:
            return {"error": "Not enough sections with content for similarity analysis"}
        
        return compute_section_similarity(sections_data)

