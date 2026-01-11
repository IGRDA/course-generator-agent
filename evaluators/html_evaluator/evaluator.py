"""HTML evaluator for section HTML structure evaluation with parallel processing."""

from typing import Dict, Any, List, Annotated
from operator import add
from pydantic import BaseModel, Field, ConfigDict
from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from main.state import CourseState
from evaluators.base import BaseEvaluator, SingleCriteriaScore
from .prompts import HTML_EVALUATION_PROMPT, INFO_PRESERVATION_EVALUATION_PROMPT, CORRECTION_PROMPT


# Valid HTML element types from state.py
VALID_ELEMENT_TYPES = {"p", "ul", "quote", "table", "paragraphs"}


# ---- State Models for Parallel Evaluation ----

class HtmlEvalTask(BaseModel):
    """State for evaluating HTML in a single section."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    course_state: CourseState
    module_idx: int
    submodule_idx: int
    section_idx: int
    section_id: str


class HtmlEvalState(BaseModel):
    """State for the HTML evaluation graph."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    course_state: CourseState
    evaluator: "HtmlEvaluator"
    completed_evaluations: Annotated[List[dict], add] = Field(default_factory=list)


class HtmlEvaluator(BaseEvaluator):
    """Evaluates HTML structure formatting quality and validity."""
    
    def __init__(self, provider: str = "mistral", max_retries: int = 3):
        super().__init__(provider, max_retries)
        self.concurrency = 4  # default
    
    def set_concurrency(self, concurrency: int):
        """Set concurrency for parallel evaluation."""
        self.concurrency = concurrency
    
    @traceable(name="html_evaluator")
    def evaluate(self, course_state: CourseState) -> Dict[str, Any]:
        """Evaluate all HTML structures in the course using parallel processing."""
        # Build and run the evaluation graph
        graph = self._build_evaluation_graph()
        initial_state = HtmlEvalState(
            course_state=course_state,
            evaluator=self
        )
        
        result = graph.invoke(initial_state)
        section_results = result["completed_evaluations"]
        
        # Aggregate element type usage
        element_type_usage: Dict[str, int] = {}
        for r in section_results:
            for elem_type, count in r.get("element_counts", {}).items():
                element_type_usage[elem_type] = element_type_usage.get(elem_type, 0) + count
        
        formatting_scores = [r["formatting_score"] for r in section_results if r.get("formatting_score")]
        info_preservation_scores = [r["info_preservation_score"] for r in section_results if r.get("info_preservation_score")]
        
        return {
            "evaluator": "html",
            "section_results": section_results,
            "summary": {
                "total_sections_evaluated": len(section_results),
                "average_formatting_score": sum(formatting_scores) / len(formatting_scores) if formatting_scores else 0.0,
                "average_info_preservation_score": sum(info_preservation_scores) / len(info_preservation_scores) if info_preservation_scores else 0.0,
                "element_type_usage": element_type_usage,
            },
            "schema_checks": self._run_schema_checks(course_state, element_type_usage)
        }
    
    def _build_evaluation_graph(self):
        """Build the evaluation graph with Send pattern for parallelization."""
        graph = StateGraph(HtmlEvalState)
        
        # Configure retry policy
        retry_policy = RetryPolicy(
            max_attempts=self.max_retries,
            initial_interval=1.0,
            backoff_factor=2.0,
            max_interval=60.0
        )
        
        # Add nodes
        graph.add_node("plan", _plan_evaluations)
        graph.add_node("evaluate_html", _evaluate_single_html, retry=retry_policy)
        graph.add_node("reduce", _reduce_evaluations)
        
        # Add edges
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", _continue_to_evaluations, ["evaluate_html"])
        graph.add_edge("evaluate_html", "reduce")
        graph.add_edge("reduce", END)
        
        return graph.compile()
    
    def _evaluate_section_html(self, section, section_id: str) -> Dict[str, Any]:
        """Evaluate HTML for a single section."""
        result = {"section_id": section_id, "section_title": section.title, "element_counts": {}}
        
        if not section.html:
            result.update(formatting_score=None, formatting_reasoning="No HTML structure found", has_html=False)
            return result
        
        result["has_html"] = True
        elements = section.html
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
    
    def _evaluate_section_info_preservation(self, section, section_id: str) -> Dict[str, Any]:
        """Evaluate information preservation between original theory and HTML output."""
        result = {"section_id": section_id, "section_title": section.title}
        
        # Check if we have both theory and HTML to compare
        if not section.theory or len(section.theory.strip()) < 50:
            result.update(info_preservation_score=None, info_preservation_reasoning="No original theory content to compare")
            return result
        
        if not section.html:
            result.update(info_preservation_score=None, info_preservation_reasoning="No HTML content to compare")
            return result
        
        # Extract text from HTML elements for comparison
        html_content = "\n".join(
            f"<{e.type}>: {str(e.content)}" 
            for e in section.html
        )
        
        llm_score = self.evaluate_with_rubric(
            prompt=INFO_PRESERVATION_EVALUATION_PROMPT,
            output_model=SingleCriteriaScore,
            prompt_variables={
                "section_title": section.title,
                "section_description": section.description or "No description",
                "theory_content": section.theory[:3000],
                "html_content": html_content[:3000],
            },
            correction_prompt=CORRECTION_PROMPT
        )
        result["info_preservation_score"] = llm_score.score.score
        result["info_preservation_reasoning"] = llm_score.score.reasoning
        return result
    
    def _run_schema_checks(self, course_state, element_usage: dict) -> Dict[str, Any]:
        """Run schema validation checks on HTML structures."""
        sections_with_html = 0
        invalid_types = set()
        
        for _, _, _, section in self.iter_sections(course_state):
            if section.html:
                sections_with_html += 1
                for elem in section.html:
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


# ---- Graph Functions (module-level for LangGraph) ----

def _plan_evaluations(state: HtmlEvalState) -> dict:
    """Plan phase - just returns empty dict, Send handles fan-out."""
    return {}


def _continue_to_evaluations(state: HtmlEvalState) -> List[Send]:
    """Fan-out: Create a Send for each section to evaluate in parallel."""
    sends = []
    
    for m_idx, module in enumerate(state.course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                task = HtmlEvalTask(
                    course_state=state.course_state,
                    module_idx=m_idx,
                    submodule_idx=sm_idx,
                    section_idx=s_idx,
                    section_id=f"{m_idx+1}.{sm_idx+1}.{s_idx+1}"
                )
                sends.append(Send("evaluate_html", {"task": task, "evaluator": state.evaluator}))
    
    return sends


def _evaluate_single_html(state: dict) -> dict:
    """Evaluate HTML for a single section and return result for aggregation."""
    task: HtmlEvalTask = state["task"]
    evaluator: HtmlEvaluator = state["evaluator"]
    
    module = task.course_state.modules[task.module_idx]
    submodule = module.submodules[task.submodule_idx]
    section = submodule.sections[task.section_idx]
    
    # Evaluate HTML formatting
    result = evaluator._evaluate_section_html(section, task.section_id)
    
    # Evaluate information preservation (theory vs HTML)
    preservation_result = evaluator._evaluate_section_info_preservation(section, task.section_id)
    result["info_preservation_score"] = preservation_result.get("info_preservation_score")
    result["info_preservation_reasoning"] = preservation_result.get("info_preservation_reasoning")
    
    return {"completed_evaluations": [result]}


def _reduce_evaluations(state: HtmlEvalState) -> dict:
    """Fan-in: Results are already aggregated via Annotated[list, add]."""
    return {}
