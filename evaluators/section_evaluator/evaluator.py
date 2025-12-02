"""Section evaluator for theory content evaluation with parallel processing."""

from typing import Dict, Any, List, Annotated
from operator import add
from pydantic import BaseModel, Field, ConfigDict
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from main.state import CourseState
from evaluators.base import BaseEvaluator, SingleCriteriaScore
from .prompts import SECTION_EVALUATION_PROMPT, CORRECTION_PROMPT


# ---- State Models for Parallel Evaluation ----

class SectionEvalTask(BaseModel):
    """State for evaluating a single section."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    course_state: CourseState
    module_idx: int
    submodule_idx: int
    section_idx: int
    section_id: str


class SectionEvalState(BaseModel):
    """State for the section evaluation graph."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    course_state: CourseState
    evaluator: "SectionEvaluator"
    completed_evaluations: Annotated[List[dict], add] = Field(default_factory=list)


class SectionEvaluator(BaseEvaluator):
    """Evaluates section theory content for accuracy and quality."""
    
    def __init__(self, provider: str = "mistral", max_retries: int = 3):
        super().__init__(provider, max_retries)
        self.concurrency = 4  # default
    
    def set_concurrency(self, concurrency: int):
        """Set concurrency for parallel evaluation."""
        self.concurrency = concurrency
    
    @traceable(name="section_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """Evaluate all sections in the course using parallel processing."""
        # Build and run the evaluation graph
        graph = self._build_evaluation_graph()
        initial_state = SectionEvalState(
            course_state=course_state,
            evaluator=self
        )
        
        result = graph.invoke(initial_state)
        section_results = result["completed_evaluations"]
        
        # Collect theories for NLP metrics
        all_theories = []
        for section_id, module, submodule, section in self.iter_sections(course_state):
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
    
    def _build_evaluation_graph(self):
        """Build the evaluation graph with Send pattern for parallelization."""
        graph = StateGraph(SectionEvalState)
        
        # Configure retry policy
        retry_policy = RetryPolicy(
            max_attempts=self.max_retries,
            initial_interval=1.0,
            backoff_factor=2.0,
            max_interval=60.0
        )
        
        # Add nodes
        graph.add_node("plan", _plan_evaluations)
        graph.add_node("evaluate_section", _evaluate_single_section, retry=retry_policy)
        graph.add_node("reduce", _reduce_evaluations)
        
        # Add edges
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", _continue_to_evaluations, ["evaluate_section"])
        graph.add_edge("evaluate_section", "reduce")
        graph.add_edge("reduce", END)
        
        return graph.compile()
    
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


# ---- Graph Functions (module-level for LangGraph) ----

def _plan_evaluations(state: SectionEvalState) -> dict:
    """Plan phase - just returns empty dict, Send handles fan-out."""
    return {}


def _continue_to_evaluations(state: SectionEvalState) -> List[Send]:
    """Fan-out: Create a Send for each section to evaluate in parallel."""
    sends = []
    
    for m_idx, module in enumerate(state.course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                task = SectionEvalTask(
                    course_state=state.course_state,
                    module_idx=m_idx,
                    submodule_idx=sm_idx,
                    section_idx=s_idx,
                    section_id=f"{m_idx+1}.{sm_idx+1}.{s_idx+1}"
                )
                sends.append(Send("evaluate_section", {"task": task, "evaluator": state.evaluator}))
    
    return sends


def _evaluate_single_section(state: dict) -> dict:
    """Evaluate a single section and return result for aggregation."""
    task: SectionEvalTask = state["task"]
    evaluator: SectionEvaluator = state["evaluator"]
    
    module = task.course_state.modules[task.module_idx]
    submodule = module.submodules[task.submodule_idx]
    section = submodule.sections[task.section_idx]
    
    result = evaluator._evaluate_section(
        task.course_state, module, submodule, section, task.section_id
    )
    
    return {"completed_evaluations": [result]}


def _reduce_evaluations(state: SectionEvalState) -> dict:
    """Fan-in: Results are already aggregated via Annotated[list, add]."""
    return {}
