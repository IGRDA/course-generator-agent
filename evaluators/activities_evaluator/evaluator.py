"""Activities evaluator for section activities evaluation."""

from typing import Dict, Any, List, Set
from langsmith import traceable
from main.state import CourseState
from evaluators.base import BaseEvaluator, ActivityScore
from .prompts import ACTIVITIES_EVALUATION_PROMPT, CORRECTION_PROMPT


# All possible activity types from state.py
ALL_ACTIVITY_TYPES = {
    "order_list", "fill_gaps", "swipper", "linking_terms", 
    "multiple_choice", "multi_selection"
}

ALL_FINAL_ACTIVITY_TYPES = {
    "group_activity", "discussion_forum", "individual_project", "open_ended_quiz"
}


class ActivitiesEvaluator(BaseEvaluator):
    """
    Evaluates section activities for quality, relevance, and completeness.
    """
    
    @traceable(name="activities_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """
        Evaluate all activities in the course.
        
        Args:
            course_state: The CourseState to evaluate
            
        Returns:
            Dictionary with scores and metrics
        """
        section_results = []
        activity_type_usage: Dict[str, int] = {}
        final_activity_type_usage: Dict[str, int] = {}
        
        for m_idx, module in enumerate(course_state.modules):
            for sm_idx, submodule in enumerate(module.submodules):
                for s_idx, section in enumerate(submodule.sections):
                    section_id = f"{m_idx+1}.{sm_idx+1}.{s_idx+1}"
                    
                    # Track activity types
                    if section.other_elements:
                        for activity in section.other_elements.activities:
                            activity_type_usage[activity.type] = activity_type_usage.get(activity.type, 0) + 1
                        for final_act in section.other_elements.final_activities:
                            final_activity_type_usage[final_act.type] = final_activity_type_usage.get(final_act.type, 0) + 1
                    
                    # Evaluate individual section activities
                    result = self._evaluate_section_activities(
                        course_state=course_state,
                        section=section,
                        section_id=section_id
                    )
                    section_results.append(result)
        
        # Schema checks
        schema_checks = self._run_schema_checks(
            course_state, 
            activity_type_usage, 
            final_activity_type_usage
        )
        
        # Compute averages
        quality_scores = [r["quality_score"] for r in section_results if r.get("quality_score")]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {
            "evaluator": "activities",
            "section_results": section_results,
            "summary": {
                "total_sections_evaluated": len(section_results),
                "average_quality_score": avg_quality,
                "activity_type_usage": activity_type_usage,
                "final_activity_type_usage": final_activity_type_usage,
            },
            "schema_checks": schema_checks
        }
    
    def _evaluate_section_activities(
        self,
        course_state: CourseState,
        section,
        section_id: str
    ) -> Dict[str, Any]:
        """Evaluate activities for a single section."""
        result = {
            "section_id": section_id,
            "section_title": section.title,
        }
        
        # Check if section has activities
        if not section.other_elements:
            result["quality_score"] = None
            result["quality_reasoning"] = "No activities found"
            result["has_activities"] = False
            return result
        
        other_elements = section.other_elements
        
        # Build activities text for LLM
        activities_text = self._build_activities_text(other_elements.activities)
        glossary_terms = ", ".join([g.term for g in other_elements.glossary]) if other_elements.glossary else "None"
        key_concept = other_elements.key_concept or "None"
        
        if not activities_text:
            result["quality_score"] = None
            result["quality_reasoning"] = "No activities to evaluate"
            result["has_activities"] = False
            return result
        
        result["has_activities"] = True
        result["num_activities"] = len(other_elements.activities)
        result["num_final_activities"] = len(other_elements.final_activities)
        
        # Run LLM-as-judge evaluation
        try:
            theory_summary = section.theory[:500] if section.theory else "No theory content"
            
            llm_score = self.evaluate_with_rubric(
                prompt=ACTIVITIES_EVALUATION_PROMPT,
                output_model=ActivityScore,
                prompt_variables={
                    "section_title": section.title,
                    "theory_summary": theory_summary,
                    "activities_text": activities_text,
                    "glossary_terms": glossary_terms,
                    "key_concept": key_concept,
                },
                correction_prompt=CORRECTION_PROMPT
            )
            result["quality_score"] = llm_score.quality.score
            result["quality_reasoning"] = llm_score.quality.reasoning
        except Exception as e:
            result["quality_score"] = None
            result["quality_reasoning"] = f"Evaluation failed: {str(e)}"
        
        return result
    
    def _build_activities_text(self, activities) -> str:
        """Build text representation of activities for LLM."""
        lines = []
        for i, activity in enumerate(activities, 1):
            lines.append(f"{i}. Type: {activity.type}")
            if hasattr(activity.content, 'question'):
                lines.append(f"   Question: {activity.content.question}")
        return "\n".join(lines) if lines else ""
    
    def _run_schema_checks(
        self,
        course_state: CourseState,
        activity_type_usage: Dict[str, int],
        final_activity_type_usage: Dict[str, int]
    ) -> Dict[str, Any]:
        """Run schema validation checks on activities."""
        sections_with_activities = 0
        sections_with_glossary = 0
        sections_with_key_concept = 0
        
        for module in course_state.modules:
            for submodule in module.submodules:
                for section in submodule.sections:
                    if section.other_elements:
                        if section.other_elements.activities:
                            sections_with_activities += 1
                        if section.other_elements.glossary:
                            sections_with_glossary += 1
                        if section.other_elements.key_concept:
                            sections_with_key_concept += 1
        
        total_sections = sum(
            len(sm.sections)
            for m in course_state.modules
            for sm in m.submodules
        )
        
        # Check activity type coverage
        used_activity_types = set(activity_type_usage.keys())
        missing_activity_types = ALL_ACTIVITY_TYPES - used_activity_types
        
        used_final_types = set(final_activity_type_usage.keys())
        missing_final_types = ALL_FINAL_ACTIVITY_TYPES - used_final_types
        
        return {
            "total_sections": total_sections,
            "sections_with_activities": sections_with_activities,
            "sections_with_glossary": sections_with_glossary,
            "sections_with_key_concept": sections_with_key_concept,
            "activity_type_coverage": len(used_activity_types) / len(ALL_ACTIVITY_TYPES),
            "missing_activity_types": list(missing_activity_types),
            "final_activity_type_coverage": len(used_final_types) / len(ALL_FINAL_ACTIVITY_TYPES),
            "missing_final_activity_types": list(missing_final_types),
            "all_sections_have_activities": sections_with_activities == total_sections
        }

