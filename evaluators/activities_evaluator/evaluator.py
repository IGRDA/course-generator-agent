"""Activities evaluator for section activities evaluation."""

from typing import Dict, Any
from langsmith import traceable
from main.state import CourseState
from evaluators.base import BaseEvaluator, SingleCriteriaScore
from .prompts import ACTIVITIES_EVALUATION_PROMPT, CORRECTION_PROMPT


# All possible activity types from state.py
ALL_ACTIVITY_TYPES = {"order_list", "fill_gaps", "swipper", "linking_terms", "multiple_choice", "multi_selection"}
ALL_FINAL_ACTIVITY_TYPES = {"group_activity", "discussion_forum", "individual_project", "open_ended_quiz"}


class ActivitiesEvaluator(BaseEvaluator):
    """Evaluates section activities for quality, relevance, and completeness."""
    
    @traceable(name="activities_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """Evaluate all activities in the course."""
        section_results = []
        activity_type_usage: Dict[str, int] = {}
        final_activity_type_usage: Dict[str, int] = {}
        
        for section_id, _, _, section in self.iter_sections(course_state):
            # Track activity types
            if section.other_elements:
                for act in section.other_elements.activities:
                    activity_type_usage[act.type] = activity_type_usage.get(act.type, 0) + 1
                for final_act in section.other_elements.final_activities:
                    final_activity_type_usage[final_act.type] = final_activity_type_usage.get(final_act.type, 0) + 1
            
            result = self._evaluate_section_activities(course_state, section, section_id)
            section_results.append(result)
        
        quality_scores = [r["quality_score"] for r in section_results if r.get("quality_score")]
        
        return {
            "evaluator": "activities",
            "section_results": section_results,
            "summary": {
                "total_sections_evaluated": len(section_results),
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                "activity_type_usage": activity_type_usage,
                "final_activity_type_usage": final_activity_type_usage,
            },
            "schema_checks": self._run_schema_checks(course_state, activity_type_usage, final_activity_type_usage)
        }
    
    def _evaluate_section_activities(self, course_state, section, section_id: str) -> Dict[str, Any]:
        """Evaluate activities for a single section."""
        result = {"section_id": section_id, "section_title": section.title}
        
        if not section.other_elements:
            result.update(quality_score=None, quality_reasoning="No activities found", has_activities=False)
            return result
        
        other = section.other_elements
        activities_text = self._build_activities_text(other.activities)
        
        if not activities_text:
            result.update(quality_score=None, quality_reasoning="No activities to evaluate", has_activities=False)
            return result
        
        result.update(
            has_activities=True,
            num_activities=len(other.activities),
            num_final_activities=len(other.final_activities)
        )
        
        llm_score = self.evaluate_with_rubric(
            prompt=ACTIVITIES_EVALUATION_PROMPT,
            output_model=SingleCriteriaScore,
            prompt_variables={
                "section_title": section.title,
                "theory_summary": (section.theory[:500] if section.theory else "No theory content"),
                "activities_text": activities_text,
                "glossary_terms": ", ".join(g.term for g in other.glossary) if other.glossary else "None",
                "key_concept": other.key_concept or "None",
            },
            correction_prompt=CORRECTION_PROMPT
        )
        result["quality_score"] = llm_score.score.score
        result["quality_reasoning"] = llm_score.score.reasoning
        return result
    
    def _build_activities_text(self, activities) -> str:
        """Build text representation of activities for LLM."""
        lines = []
        for i, act in enumerate(activities, 1):
            lines.append(f"{i}. Type: {act.type}")
            if hasattr(act.content, 'question'):
                lines.append(f"   Question: {act.content.question}")
        return "\n".join(lines)
    
    def _run_schema_checks(self, course_state, activity_usage: dict, final_usage: dict) -> Dict[str, Any]:
        """Run schema validation checks on activities."""
        sections_with_activities = sections_with_glossary = sections_with_key_concept = 0
        
        for _, _, _, section in self.iter_sections(course_state):
            if section.other_elements:
                if section.other_elements.activities:
                    sections_with_activities += 1
                if section.other_elements.glossary:
                    sections_with_glossary += 1
                if section.other_elements.key_concept:
                    sections_with_key_concept += 1
        
        total = self.count_sections(course_state)
        used_types = set(activity_usage.keys())
        used_final = set(final_usage.keys())
        
        return {
            "total_sections": total,
            "sections_with_activities": sections_with_activities,
            "sections_with_glossary": sections_with_glossary,
            "sections_with_key_concept": sections_with_key_concept,
            "activity_type_coverage": len(used_types) / len(ALL_ACTIVITY_TYPES),
            "missing_activity_types": list(ALL_ACTIVITY_TYPES - used_types),
            "final_activity_type_coverage": len(used_final) / len(ALL_FINAL_ACTIVITY_TYPES),
            "missing_final_activity_types": list(ALL_FINAL_ACTIVITY_TYPES - used_final),
            "all_sections_have_activities": sections_with_activities == total
        }
