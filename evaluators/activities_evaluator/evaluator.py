"""Activities evaluator for section activities evaluation with parallel processing."""

from typing import Dict, Any, List, Annotated
from operator import add
from pydantic import BaseModel, Field, ConfigDict
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from main.state import CourseState
from evaluators.base import BaseEvaluator, SingleCriteriaScore
from .prompts import ACTIVITIES_EVALUATION_PROMPT, CORRECTION_PROMPT


# All possible activity types from state.py
ALL_ACTIVITY_TYPES = {"order_list", "fill_gaps", "swipper", "linking_terms", "multiple_choice", "multi_selection"}
ALL_FINAL_ACTIVITY_TYPES = {"group_activity", "discussion_forum", "individual_project", "open_ended_quiz"}


# ---- State Models for Parallel Evaluation ----

class ActivitiesEvalTask(BaseModel):
    """State for evaluating activities in a single section."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    course_state: CourseState
    module_idx: int
    submodule_idx: int
    section_idx: int
    section_id: str


class ActivitiesEvalState(BaseModel):
    """State for the activities evaluation graph."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    course_state: CourseState
    evaluator: "ActivitiesEvaluator"
    completed_evaluations: Annotated[List[dict], add] = Field(default_factory=list)


class ActivitiesEvaluator(BaseEvaluator):
    """Evaluates section activities for quality, relevance, and completeness."""
    
    def __init__(self, provider: str = "mistral", max_retries: int = 3):
        super().__init__(provider, max_retries)
        self.concurrency = 4  # default
    
    def set_concurrency(self, concurrency: int):
        """Set concurrency for parallel evaluation."""
        self.concurrency = concurrency
    
    @traceable(name="activities_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """Evaluate all activities in the course using parallel processing."""
        # Track activity types across all sections
        activity_type_usage: Dict[str, int] = {}
        final_activity_type_usage: Dict[str, int] = {}
        
        for section_id, _, _, section in self.iter_sections(course_state):
            if section.activities:
                for act in section.activities.quiz:
                    activity_type_usage[act.type] = activity_type_usage.get(act.type, 0) + 1
                for final_act in section.activities.application:
                    final_activity_type_usage[final_act.type] = final_activity_type_usage.get(final_act.type, 0) + 1
        
        # Build and run the evaluation graph
        graph = self._build_evaluation_graph()
        initial_state = ActivitiesEvalState(
            course_state=course_state,
            evaluator=self
        )
        
        result = graph.invoke(initial_state)
        section_results = result["completed_evaluations"]
        
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
    
    def _build_evaluation_graph(self):
        """Build the evaluation graph with Send pattern for parallelization."""
        graph = StateGraph(ActivitiesEvalState)
        
        # Configure retry policy
        retry_policy = RetryPolicy(
            max_attempts=self.max_retries,
            initial_interval=1.0,
            backoff_factor=2.0,
            max_interval=60.0
        )
        
        # Add nodes
        graph.add_node("plan", _plan_evaluations)
        graph.add_node("evaluate_activities", _evaluate_single_activities, retry=retry_policy)
        graph.add_node("reduce", _reduce_evaluations)
        
        # Add edges
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", _continue_to_evaluations, ["evaluate_activities"])
        graph.add_edge("evaluate_activities", "reduce")
        graph.add_edge("reduce", END)
        
        return graph.compile()
    
    def _evaluate_section_activities(self, course_state, section, section_id: str) -> Dict[str, Any]:
        """Evaluate activities for a single section."""
        result = {"section_id": section_id, "section_title": section.title}
        
        if not section.activities:
            result.update(quality_score=None, quality_reasoning="No activities found", has_activities=False)
            return result
        
        activities_text = self._build_activities_text(section.activities.quiz)
        
        if not activities_text:
            result.update(quality_score=None, quality_reasoning="No activities to evaluate", has_activities=False)
            return result
        
        result.update(
            has_activities=True,
            num_activities=len(section.activities.quiz),
            num_final_activities=len(section.activities.application)
        )
        
        # Get glossary and key_concept from meta_elements if available
        meta = section.meta_elements
        glossary_terms = ", ".join(g.term for g in meta.glossary) if meta and meta.glossary else "None"
        key_concept = meta.key_concept if meta and meta.key_concept else "None"
        
        llm_score = self.evaluate_with_rubric(
            prompt=ACTIVITIES_EVALUATION_PROMPT,
            output_model=SingleCriteriaScore,
            prompt_variables={
                "section_title": section.title,
                "theory_summary": (section.theory[:500] if section.theory else "No theory content"),
                "activities_text": activities_text,
                "glossary_terms": glossary_terms,
                "key_concept": key_concept,
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
            if section.activities:
                if section.activities.quiz:
                    sections_with_activities += 1
            if section.meta_elements:
                if section.meta_elements.glossary:
                    sections_with_glossary += 1
                if section.meta_elements.key_concept:
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


# ---- Graph Functions (module-level for LangGraph) ----

def _plan_evaluations(state: ActivitiesEvalState) -> dict:
    """Plan phase - just returns empty dict, Send handles fan-out."""
    return {}


def _continue_to_evaluations(state: ActivitiesEvalState) -> List[Send]:
    """Fan-out: Create a Send for each section to evaluate in parallel."""
    sends = []
    
    for m_idx, module in enumerate(state.course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                task = ActivitiesEvalTask(
                    course_state=state.course_state,
                    module_idx=m_idx,
                    submodule_idx=sm_idx,
                    section_idx=s_idx,
                    section_id=f"{m_idx+1}.{sm_idx+1}.{s_idx+1}"
                )
                sends.append(Send("evaluate_activities", {"task": task, "evaluator": state.evaluator}))
    
    return sends


def _evaluate_single_activities(state: dict) -> dict:
    """Evaluate activities for a single section and return result for aggregation."""
    task: ActivitiesEvalTask = state["task"]
    evaluator: ActivitiesEvaluator = state["evaluator"]
    
    module = task.course_state.modules[task.module_idx]
    submodule = module.submodules[task.submodule_idx]
    section = submodule.sections[task.section_idx]
    
    result = evaluator._evaluate_section_activities(
        task.course_state, section, task.section_id
    )
    
    return {"completed_evaluations": [result]}


def _reduce_evaluations(state: ActivitiesEvalState) -> dict:
    """Fan-in: Results are already aggregated via Annotated[list, add]."""
    return {}
