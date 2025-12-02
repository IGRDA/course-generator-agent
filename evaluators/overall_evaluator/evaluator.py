"""Overall evaluator for complete course evaluation."""

from typing import Dict, Any
from langsmith import traceable
from main.state import CourseState
from evaluators.base import BaseEvaluator, SingleCriteriaScore
from .prompts import OVERALL_EVALUATION_PROMPT, CORRECTION_PROMPT


class OverallEvaluator(BaseEvaluator):
    """Evaluates the complete course for coherence, completeness, and content similarity."""
    
    @traceable(name="overall_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """Evaluate the overall course."""
        return {
            "evaluator": "overall",
            "coherence": self._evaluate_coherence(course_state),
            "completeness": self._check_completeness(course_state),
            "structure_metrics": self._compute_structure_metrics(course_state),
            "embedding_metrics": self._compute_embedding_metrics(course_state),
        }
    
    def _evaluate_coherence(self, course_state: CourseState) -> Dict[str, Any]:
        """Evaluate course coherence using LLM."""
        llm_score = self.evaluate_with_rubric(
            prompt=OVERALL_EVALUATION_PROMPT,
            output_model=SingleCriteriaScore,
            prompt_variables={
                "course_title": course_state.title,
                "language": course_state.language,
                "course_overview": self._build_course_overview(course_state),
                "section_titles_sample": self._get_section_titles_sample(course_state),
            },
            correction_prompt=CORRECTION_PROMPT
        )
        return {"score": llm_score.score.score, "reasoning": llm_score.score.reasoning}
    
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
        for section_id, _, _, section in self.iter_sections(course_state):
            titles.append(f"{section_id} {section.title}")
            if len(titles) >= max_titles:
                titles.append("...")
                break
        return "\n".join(titles)
    
    def _check_completeness(self, course_state: CourseState) -> Dict[str, Any]:
        """Check if all required fields are populated."""
        with_theory = with_activities = with_html = 0
        
        for _, _, _, section in self.iter_sections(course_state):
            if section.theory and section.theory.strip():
                with_theory += 1
            if section.other_elements and section.other_elements.activities:
                with_activities += 1
            if section.html and section.html.theory:
                with_html += 1
        
        total = self.count_sections(course_state)
        return {
            "total_sections": total,
            "theory_completeness": with_theory / total if total else 0,
            "activities_completeness": with_activities / total if total else 0,
            "html_completeness": with_html / total if total else 0,
        }
    
    def _compute_structure_metrics(self, course_state: CourseState) -> Dict[str, Any]:
        """Compute structure-based metrics using dedicated module."""
        from evaluation.metrics.structure_metrics import compute_title_uniqueness
        return {"title_uniqueness": compute_title_uniqueness(course_state)}
    
    def _compute_embedding_metrics(self, course_state: CourseState) -> Dict[str, Any]:
        """Compute embedding-based similarity metrics."""
        from evaluation.metrics.embedding_metrics import compute_section_similarity, compute_title_embedding_similarity
        
        # Collect titles
        module_titles = [m.title for m in course_state.modules]
        submodule_titles = [sm.title for m in course_state.modules for sm in m.submodules]
        section_titles = [s.title for _, _, _, s in self.iter_sections(course_state)]
        
        title_embedding = compute_title_embedding_similarity(module_titles, submodule_titles, section_titles)
        
        # Content similarity
        sections_data = [
            {"id": sid, "title": s.title, "text": s.theory}
            for sid, _, _, s in self.iter_sections(course_state)
            if s.theory and len(s.theory.strip()) > 50
        ]
        
        content_similarity = (
            compute_section_similarity(sections_data) if len(sections_data) >= 2
            else {"error": "Not enough sections with content for similarity analysis"}
        )
        
        return {"title_embedding": title_embedding, "content_similarity": content_similarity}
