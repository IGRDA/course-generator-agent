"""Evaluation workflow using LangSmith Evaluations API.

Uses proper LangSmith experiments, datasets, and evaluators for:
- Experiment tracking and comparison in the LangSmith UI
- Custom LLM-as-judge evaluators with rubric scoring
- Single trace per complete evaluation (via LangGraph)

Usage:
    # Run full evaluation (all evaluators)
    python -m evaluation.workflow evaluate --dataset my-courses
    
    # Run specific evaluators only
    python -m evaluation.workflow evaluate --dataset my-courses --steps index_evaluator overall_evaluator
    
    # Quick evaluation of a single file (creates temp dataset automatically)
    python -m evaluation.workflow quick --input output/course.json
"""

from typing import Dict, Any, List, Optional, Callable, NamedTuple
from dataclasses import dataclass

from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult
from langsmith.schemas import Example, Run
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from main.state import CourseState
from evaluation.dataset import load_course_state, get_client, create_dataset
from evaluators.index_evaluator import IndexEvaluator
from evaluators.section_evaluator import SectionEvaluator
from evaluators.activities_evaluator import ActivitiesEvaluator
from evaluators.html_evaluator import HtmlEvaluator
from evaluators.overall_evaluator import OverallEvaluator


# ---- Evaluation Graph State ----

class EvalGraphState(BaseModel):
    """State for the evaluation graph."""
    course_state: CourseState
    provider: str = "mistral"
    max_retries: int = 3
    steps: Optional[List[str]] = None
    
    index_result: Optional[Dict[str, Any]] = None
    section_result: Optional[Dict[str, Any]] = None
    activities_result: Optional[Dict[str, Any]] = None
    html_result: Optional[Dict[str, Any]] = None
    overall_result: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True


# ---- Evaluator Registry & Node Factory ----

EVALUATOR_REGISTRY = [
    ("index_evaluator", "index_eval", "index_result", IndexEvaluator),
    ("section_evaluator", "section_eval", "section_result", SectionEvaluator),
    ("activities_evaluator", "activities_eval", "activities_result", ActivitiesEvaluator),
    ("html_evaluator", "html_eval", "html_result", HtmlEvaluator),
    ("overall_evaluator", "overall_eval", "overall_result", OverallEvaluator),
]


def _make_eval_node(evaluator_cls, result_key: str, name: str):
    """Factory function to create evaluation graph nodes."""
    def node(state: EvalGraphState) -> Dict[str, Any]:
        print(f"   Running {name}...")
        evaluator = evaluator_cls(provider=state.provider, max_retries=state.max_retries)
        result = evaluator.evaluate(state.course_state)
        print(f"   ✓ {name} complete")
        return {result_key: result}
    return node


def get_evaluation_graph(steps: Optional[List[str]] = None):
    """Build the evaluation StateGraph with optional step selection."""
    evaluators = [
        (name, node_id, _make_eval_node(cls, result_key, name))
        for name, node_id, result_key, cls in EVALUATOR_REGISTRY
        if steps is None or name in steps
    ]
    
    graph = StateGraph(EvalGraphState)
    
    if not evaluators:
        graph.add_edge(START, END)
        return graph.compile()
    
    for _, node_id, func in evaluators:
        graph.add_node(node_id, func)
    
    graph.add_edge(START, evaluators[0][1])
    for i in range(len(evaluators) - 1):
        graph.add_edge(evaluators[i][1], evaluators[i + 1][1])
    graph.add_edge(evaluators[-1][1], END)
    
    return graph.compile()


# ---- Global Evaluator Settings ----

_eval_settings = {"provider": "mistral", "max_retries": 3, "steps": None}


def set_evaluator_settings(provider: str = "mistral", max_retries: int = 3, steps: Optional[List[str]] = None):
    """Set global evaluator settings."""
    global _eval_settings
    _eval_settings = {"provider": provider, "max_retries": max_retries, "steps": steps}


# ---- Declarative Metrics Extraction ----

@dataclass
class MetricDef:
    """Definition for a metric to extract from evaluation state."""
    key: str
    path: str  # dot-separated path like "index_result.llm_scores.coverage"
    normalizer: Callable[[Any], float]
    comment_fn: Callable[[Any], str]
    exclude_from_avg: bool = False


def _get_nested(obj, path: str):
    """Get nested value from object/dict using dot notation."""
    for part in path.split("."):
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
    return obj


def _div5(v): return v["score"] / 5.0
def _div5_direct(v): return v / 5.0
def _identity(v): return v
def _invert(v): return round(1 - v, 4)
def _normalize_readability(score: float) -> float:
    """Normalize readability score (Coleman-Liau/ARI ~5-18) to 0-1 range."""
    clamped = max(5, min(18, score))
    return round(1 - (clamped - 5) / 13, 4)


# Metric definitions registry
METRIC_DEFS: List[MetricDef] = [
    # Index metrics
    MetricDef("index_coverage", "index_result.llm_scores.coverage", _div5, lambda v: v["reasoning"]),
    MetricDef("index_structure", "index_result.llm_scores.structure", _div5, lambda v: v["reasoning"]),
    MetricDef("index_balance", "index_result.llm_scores.balance", _div5, lambda v: v["reasoning"]),
    
    # Section metrics
    MetricDef("section_accuracy", "section_result.summary.average_accuracy_score", _div5_direct,
              lambda v: f"Average accuracy score"),
    MetricDef("readability_score", "section_result.nlp_metrics.readability.readability_score",
              _normalize_readability, lambda v: f"Coleman-Liau+ARI mean: {v:.1f}"),
    MetricDef("avg_sentence_length", "section_result.nlp_metrics.readability.avg_sentence_length",
              _identity, lambda v: f"Avg words/sentence: {v:.1f}", exclude_from_avg=True),
    MetricDef("word_count", "section_result.nlp_metrics.readability.word_count",
              _identity, lambda v: f"Total words: {v}", exclude_from_avg=True),
    MetricDef("vocabulary_diversity", "section_result.nlp_metrics.repetition.type_token_ratio",
              _identity, lambda v: f"Type-token ratio: {v:.3f}"),
    MetricDef("ngram_originality", "section_result.nlp_metrics.repetition.weighted_ngram_repetition",
              _invert, lambda v: f"Weighted n-gram repetition: {v:.3f}"),
    
    # Activities metrics
    MetricDef("activities_quality", "activities_result.summary.average_quality_score", _div5_direct,
              lambda v: "Average activity quality"),
    
    # HTML metrics
    MetricDef("html_formatting", "html_result.summary.average_formatting_score", _div5_direct,
              lambda v: "Average HTML formatting score"),
    MetricDef("html_info_preservation", "html_result.summary.average_info_preservation_score", _div5_direct,
              lambda v: "Average information preservation score (theory vs HTML)"),
    
    # Overall metrics
    MetricDef("overall_coherence", "overall_result.coherence.score", _div5_direct,
              lambda v: "Course coherence score"),
]


def _extract_completeness_metrics(state: EvalGraphState) -> List[EvaluationResult]:
    """Extract completeness metrics from overall result."""
    results = []
    completeness = _get_nested(state, "overall_result.completeness")
    if not completeness:
        return results
    
    keys = ["theory_completeness", "activities_completeness", "html_completeness"]
    comp_score = sum(completeness.get(k, 0) for k in keys) / 3
    results.append(EvaluationResult(
        key="overall_completeness", score=comp_score,
        comment=f"Theory: {completeness.get('theory_completeness', 0):.0%}, "
                f"Activities: {completeness.get('activities_completeness', 0):.0%}, "
                f"HTML: {completeness.get('html_completeness', 0):.0%}"
    ))
    return results


def _extract_activities_completeness(state: EvalGraphState) -> List[EvaluationResult]:
    """Extract activities completeness metric."""
    checks = _get_nested(state, "activities_result.schema_checks")
    if not checks:
        return []
    total = checks.get("total_sections", 0)
    with_act = checks.get("sections_with_activities", 0)
    if total == 0:
        return []
    return [EvaluationResult(
        key="activities_completeness",
        score=with_act / total,
        comment=f"{with_act}/{total} sections have activities"
    )]


def _extract_html_completeness(state: EvalGraphState) -> List[EvaluationResult]:
    """Extract HTML completeness metric."""
    checks = _get_nested(state, "html_result.schema_checks")
    if not checks:
        return []
    total = checks.get("total_sections", 0)
    with_html = checks.get("sections_with_html", 0)
    if total == 0:
        return []
    return [EvaluationResult(
        key="html_completeness",
        score=with_html / total,
        comment=f"{with_html}/{total} sections have HTML"
    )]


def _extract_structure_metrics(state: EvalGraphState) -> List[EvaluationResult]:
    """Extract title uniqueness metrics from structure analysis."""
    results = []
    uniqueness = _get_nested(state, "overall_result.structure_metrics.title_uniqueness")
    if not uniqueness:
        return results
    
    metric_map = [
        ("module_uniqueness", "total_modules", "Exact match uniqueness for {n} modules"),
        ("submodule_uniqueness", "total_submodules", "Exact match uniqueness for {n} submodules"),
        ("section_uniqueness", "total_sections", "Exact match uniqueness for {n} sections"),
        ("module_ngram_uniqueness", None, "Weighted n-gram uniqueness for modules"),
        ("submodule_ngram_uniqueness", None, "Weighted n-gram uniqueness for submodules"),
        ("section_ngram_uniqueness", None, "Weighted n-gram uniqueness for sections"),
    ]
    
    for key, count_key, comment_tmpl in metric_map:
        if key in uniqueness:
            comment = comment_tmpl.format(n=uniqueness.get(count_key, "")) if count_key else comment_tmpl
            results.append(EvaluationResult(key=key, score=uniqueness[key], comment=comment))
    
    return results


def _extract_embedding_metrics(state: EvalGraphState) -> List[EvaluationResult]:
    """Extract embedding-based metrics."""
    results = []
    embedding = _get_nested(state, "overall_result.embedding_metrics")
    if not embedding or "error" in embedding:
        return results
    
    # Title embedding uniqueness
    title_emb = embedding.get("title_embedding", {})
    for level in ["module", "submodule", "section"]:
        key = f"{level}_embedding_uniqueness"
        if key in title_emb:
            results.append(EvaluationResult(
                key=key, score=title_emb[key],
                comment=f"Embedding-based uniqueness for {level}s"
            ))
    
    # Content similarity
    content_sim = embedding.get("content_similarity", {})
    if content_sim and "error" not in content_sim:
        avg_sim = content_sim.get("avg_similarity", 0.5)
        results.append(EvaluationResult(
            key="content_diversity", score=round(1 - avg_sim, 4),
            comment=f"Avg similarity: {avg_sim:.3f}"
        ))
        if "max_similarity" in content_sim:
            results.append(EvaluationResult(
                key="content_max_diversity", score=round(1 - content_sim["max_similarity"], 4),
                comment=f"Max similarity: {content_sim['max_similarity']:.3f}"
            ))
        if "min_similarity" in content_sim:
            results.append(EvaluationResult(
                key="content_min_diversity", score=round(1 - content_sim["min_similarity"], 4),
                comment=f"Min similarity: {content_sim['min_similarity']:.3f}"
            ))
        if "std_similarity" in content_sim:
            results.append(EvaluationResult(
                key="content_consistency", score=round(1 - min(1, content_sim["std_similarity"] * 2), 4),
                comment=f"Std similarity: {content_sim['std_similarity']:.3f}"
            ))
    
    return results


def extract_metrics(state: EvalGraphState) -> List[EvaluationResult]:
    """Extract all metrics from evaluation graph state."""
    results = []
    
    # Extract declarative metrics
    for metric in METRIC_DEFS:
        value = _get_nested(state, metric.path)
        if value is not None:
            try:
                results.append(EvaluationResult(
                    key=metric.key,
                    score=metric.normalizer(value),
                    comment=metric.comment_fn(value)
                ))
            except (TypeError, KeyError):
                pass  # Skip metrics that can't be computed
    
    # Extract complex metrics via helper functions
    results.extend(_extract_completeness_metrics(state))
    results.extend(_extract_activities_completeness(state))
    results.extend(_extract_html_completeness(state))
    results.extend(_extract_structure_metrics(state))
    results.extend(_extract_embedding_metrics(state))
    
    return results


# ---- Combined Evaluator ----

def combined_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """Single evaluator that runs the LangGraph evaluation workflow."""
    client = Client()
    course_state = CourseState.model_validate(example.inputs["course_data"])
    course_title = example.inputs.get("course_title", "Unknown")
    
    print(f"\n📊 Evaluating: {course_title}")
    
    eval_state = EvalGraphState(
        course_state=course_state,
        provider=_eval_settings["provider"],
        max_retries=_eval_settings["max_retries"],
        steps=_eval_settings.get("steps"),
    )
    
    graph = get_evaluation_graph(steps=_eval_settings.get("steps"))
    final_state = graph.invoke(eval_state)
    
    if isinstance(final_state, dict):
        final_state = EvalGraphState(**final_state)
    
    metrics = extract_metrics(final_state)
    print(f"   📈 Total metrics extracted: {len(metrics)}")
    
    # Log metrics and compute average
    exclude_from_avg = {"word_count", "avg_sentence_length"}
    overall_score, count = 0.0, 0
    
    for metric in metrics:
        try:
            client.create_feedback(run_id=run.id, key=metric.key, score=metric.score, comment=metric.comment)
            if metric.score is not None and metric.key not in exclude_from_avg:
                overall_score += metric.score
                count += 1
            print(f"      ✓ {metric.key}: {metric.score:.3f}" if metric.score else f"      ✓ {metric.key}: N/A")
        except Exception as e:
            print(f"      ⚠ Failed to log {metric.key}: {e}")
    
    avg_score = overall_score / count if count > 0 else 0.0
    print(f"   📊 Average score: {avg_score:.3f}")
    
    return EvaluationResult(key="average_score", score=avg_score, comment=f"Average of {count} metrics")


# ---- Main Evaluation Functions ----

def run_evaluation(
    dataset_name: str,
    experiment_prefix: str = "course-eval",
    provider: str = "mistral",
    max_retries: int = 3,
    steps: Optional[List[str]] = None,
    example_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run evaluation experiment against a LangSmith dataset."""
    print(f"\n🔬 Running evaluation experiment")
    print(f"   Dataset: {dataset_name}")
    print(f"   Experiment prefix: {experiment_prefix}")
    print(f"   Provider: {provider}")
    print(f"   Evaluators: {', '.join(steps) if steps else 'all (full evaluation)'}")
    print(f"   Examples: {len(example_ids) if example_ids else 'all in dataset'}")
    print("-" * 50)
    
    set_evaluator_settings(provider=provider, max_retries=max_retries, steps=steps)
    
    def target(inputs: dict) -> dict:
        return {"course_title": inputs.get("course_title", "Unknown")}
    
    if example_ids:
        client = get_client()
        data_param = [client.read_example(eid) for eid in example_ids]
    else:
        data_param = dataset_name
    
    return evaluate(
        target,
        data=data_param,
        evaluators=[combined_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,
    )


def quick_evaluate(
    input_file: str,
    provider: str = "mistral",
    max_retries: int = 3,
    experiment_prefix: str = "quick-eval",
    steps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Quick evaluation of a single file (creates temporary dataset automatically)."""
    print(f"\n📦 Preparing dataset...")
    dataset_id, dataset_name, example_ids = create_dataset(
        [input_file], dataset_name=None, 
        description="Dataset for course evaluation", use_existing=True
    )
    
    return run_evaluation(
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
        provider=provider,
        max_retries=max_retries,
        steps=steps,
        example_ids=example_ids,
    )


# ---- CLI ----

def main():
    """Command line interface for evaluation workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Course Evaluation Workflow")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation on a dataset")
    eval_parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    eval_parser.add_argument("--experiment-prefix", type=str, default="course-eval")
    eval_parser.add_argument("--provider", type=str, default="mistral")
    eval_parser.add_argument("--max-retries", type=int, default=3)
    eval_parser.add_argument("--steps", nargs="*", type=str, default=None)
    
    quick_parser = subparsers.add_parser("quick", help="Quick evaluation of a single file")
    quick_parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    quick_parser.add_argument("--provider", type=str, default="mistral")
    quick_parser.add_argument("--max-retries", type=int, default=3)
    quick_parser.add_argument("--experiment-prefix", type=str, default="quick-eval")
    quick_parser.add_argument("--steps", nargs="*", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.command == "evaluate":
        run_evaluation(
            dataset_name=args.dataset,
            experiment_prefix=args.experiment_prefix,
            provider=args.provider,
            max_retries=args.max_retries,
            steps=args.steps,
        )
    elif args.command == "quick":
        quick_evaluate(
            input_file=args.input,
            provider=args.provider,
            max_retries=args.max_retries,
            experiment_prefix=args.experiment_prefix,
            steps=args.steps,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
