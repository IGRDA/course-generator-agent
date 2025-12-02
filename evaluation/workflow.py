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

from typing import Dict, Any, List, Optional

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


# ---- Helper Functions ----

def _get_course_state(inputs: dict) -> CourseState:
    """Extract CourseState from example inputs."""
    return CourseState.model_validate(inputs["course_data"])


# ---- Evaluation Graph State ----

class EvalGraphState(BaseModel):
    """State for the evaluation graph."""
    # Input
    course_state: CourseState
    provider: str = "mistral"
    max_retries: int = 3
    steps: Optional[List[str]] = None  # Optional list of evaluators to run
    
    # Results from each evaluator (populated by nodes)
    index_result: Optional[Dict[str, Any]] = None
    section_result: Optional[Dict[str, Any]] = None
    activities_result: Optional[Dict[str, Any]] = None
    html_result: Optional[Dict[str, Any]] = None
    overall_result: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True


# ---- Graph Nodes ----

def run_index_evaluation(state: EvalGraphState) -> Dict[str, Any]:
    """Run index evaluator and return result."""
    print("   Running index evaluation...")
    evaluator = IndexEvaluator(provider=state.provider, max_retries=state.max_retries)
    result = evaluator.evaluate(state.course_state)
    print("   ✓ Index evaluation complete")
    return {"index_result": result}


def run_section_evaluation(state: EvalGraphState) -> Dict[str, Any]:
    """Run section evaluator and return result."""
    print("   Running section evaluation...")
    evaluator = SectionEvaluator(provider=state.provider, max_retries=state.max_retries)
    result = evaluator.evaluate(state.course_state)
    print("   ✓ Section evaluation complete")
    return {"section_result": result}


def run_activities_evaluation(state: EvalGraphState) -> Dict[str, Any]:
    """Run activities evaluator and return result."""
    print("   Running activities evaluation...")
    evaluator = ActivitiesEvaluator(provider=state.provider, max_retries=state.max_retries)
    result = evaluator.evaluate(state.course_state)
    print("   ✓ Activities evaluation complete")
    return {"activities_result": result}


def run_html_evaluation(state: EvalGraphState) -> Dict[str, Any]:
    """Run HTML evaluator and return result."""
    print("   Running HTML evaluation...")
    evaluator = HtmlEvaluator(provider=state.provider, max_retries=state.max_retries)
    result = evaluator.evaluate(state.course_state)
    print("   ✓ HTML evaluation complete")
    return {"html_result": result}


def run_overall_evaluation(state: EvalGraphState) -> Dict[str, Any]:
    """Run overall evaluator and return result."""
    print("   Running overall evaluation...")
    evaluator = OverallEvaluator(provider=state.provider, max_retries=state.max_retries)
    result = evaluator.evaluate(state.course_state)
    print("   ✓ Overall evaluation complete")
    return {"overall_result": result}


# ---- Build Graph ----

def get_evaluation_graph(steps: Optional[List[str]] = None):
    """
    Build the evaluation StateGraph with optional step selection.
    
    Args:
        steps: Optional list of evaluator names to run. 
               Valid values: index_evaluator, section_evaluator, activities_evaluator, 
                            html_evaluator, overall_evaluator
               If None, runs all evaluators.
    
    Returns:
        Compiled StateGraph
    """
    # Define all available evaluators
    all_evaluators = [
        ("index_evaluator", "index_eval", run_index_evaluation),
        ("section_evaluator", "section_eval", run_section_evaluation),
        ("activities_evaluator", "activities_eval", run_activities_evaluation),
        ("html_evaluator", "html_eval", run_html_evaluation),
        ("overall_evaluator", "overall_eval", run_overall_evaluation),
    ]
    
    # Filter evaluators based on steps parameter
    if steps:
        evaluators_to_run = [
            (name, node_id, func) for name, node_id, func in all_evaluators
            if name in steps
        ]
    else:
        evaluators_to_run = all_evaluators
    
    # Build graph
    graph = StateGraph(EvalGraphState)
    
    # Add nodes for selected evaluators
    for _, node_id, func in evaluators_to_run:
        graph.add_node(node_id, func)
    
    # Define sequential flow
    if evaluators_to_run:
        # Connect START to first node
        graph.add_edge(START, evaluators_to_run[0][1])
        
        # Connect nodes sequentially
        for i in range(len(evaluators_to_run) - 1):
            graph.add_edge(evaluators_to_run[i][1], evaluators_to_run[i + 1][1])
        
        # Connect last node to END
        graph.add_edge(evaluators_to_run[-1][1], END)
    else:
        # No evaluators selected, direct path from START to END
        graph.add_edge(START, END)
    
    return graph.compile()


# ---- Global Evaluator Settings ----

_eval_settings = {"provider": "mistral", "max_retries": 3, "steps": None}


def set_evaluator_settings(provider: str = "mistral", max_retries: int = 3, steps: Optional[List[str]] = None):
    """Set global evaluator settings."""
    global _eval_settings
    _eval_settings = {"provider": provider, "max_retries": max_retries, "steps": steps}


# ---- Metrics Extraction ----

def _normalize_readability(score: float) -> float:
    """Normalize readability score (Coleman-Liau/ARI ~5-18) to 0-1 range."""
    # Grade level 5-18 maps to 1.0-0.0 (lower grade = more readable = higher score)
    clamped = max(5, min(18, score))
    return round(1 - (clamped - 5) / 13, 4)


def extract_metrics(state: EvalGraphState) -> List[EvaluationResult]:
    """
    Extract all metrics from evaluation graph state.
    Converts evaluator results to LangSmith EvaluationResult format.
    All scores normalized to 0-1 range.
    """
    results = []
    
    # --- Index Metrics ---
    if state.index_result:
        llm_scores = state.index_result.get("llm_scores", {})
        
        if "coverage" in llm_scores:
            results.append(EvaluationResult(
                key="index_coverage",
                score=llm_scores["coverage"]["score"] / 5.0,
                comment=llm_scores["coverage"]["reasoning"],
            ))
        
        if "structure" in llm_scores:
            results.append(EvaluationResult(
                key="index_structure",
                score=llm_scores["structure"]["score"] / 5.0,
                comment=llm_scores["structure"]["reasoning"],
            ))
        
        if "balance" in llm_scores:
            results.append(EvaluationResult(
                key="index_balance",
                score=llm_scores["balance"]["score"] / 5.0,
                comment=llm_scores["balance"]["reasoning"],
            ))
    
    # --- Section Metrics ---
    if state.section_result:
        summary = state.section_result.get("summary", {})
        nlp = state.section_result.get("nlp_metrics", {})
        
        if "average_accuracy_score" in summary:
            results.append(EvaluationResult(
                key="section_accuracy",
                score=summary["average_accuracy_score"] / 5.0,
                comment=f"Evaluated {summary.get('total_sections', 0)} sections",
            ))
        
        # Readability metrics
        readability = nlp.get("readability", {})
        if "readability_score" in readability:
            score = readability["readability_score"]
            results.append(EvaluationResult(
                key="readability_score",
                score=_normalize_readability(score),
                comment=f"Coleman-Liau+ARI mean: {score:.1f}",
            ))
        
        if "avg_sentence_length" in readability:
            length = readability["avg_sentence_length"]
            results.append(EvaluationResult(
                key="avg_sentence_length",
                score=length,  # Raw value, not normalized
                comment=f"Avg words/sentence: {length:.1f}",
            ))
        
        if "word_count" in readability:
            count = readability["word_count"]
            results.append(EvaluationResult(
                key="word_count",
                score=count,  # Raw word count, not normalized
                comment=f"Total words: {count}",
            ))
        
        # Repetition metrics
        repetition = nlp.get("repetition", {})
        if "type_token_ratio" in repetition:
            ttr = repetition["type_token_ratio"]
            results.append(EvaluationResult(
                key="vocabulary_diversity",
                score=ttr,
                comment=f"Type-token ratio: {ttr:.3f}",
            ))
        
        if "weighted_ngram_repetition" in repetition:
            rep = repetition["weighted_ngram_repetition"]
            results.append(EvaluationResult(
                key="ngram_originality",
                score=round(1 - rep, 4),  # Invert: low repetition = high originality
                comment=f"Weighted n-gram repetition: {rep:.3f}",
            ))
    
    # --- Activities Metrics ---
    if state.activities_result:
        checks = state.activities_result.get("schema_checks", {})
        summary = state.activities_result.get("summary", {})
        
        if "average_quality_score" in summary:
            results.append(EvaluationResult(
                key="activities_quality",
                score=summary["average_quality_score"] / 5.0,
                comment=f"Activity type coverage: {checks.get('activity_type_coverage', 0):.0%}",
            ))
        
        if "total_sections" in checks and "sections_with_activities" in checks:
            total = checks["total_sections"]
            with_activities = checks["sections_with_activities"]
            results.append(EvaluationResult(
                key="activities_completeness",
                score=with_activities / total if total > 0 else 0,
                comment=f"{with_activities}/{total} sections have activities",
            ))
    
    # --- HTML Metrics ---
    if state.html_result:
        checks = state.html_result.get("schema_checks", {})
        summary = state.html_result.get("summary", {})
        
        if "average_formatting_score" in summary:
            results.append(EvaluationResult(
                key="html_formatting",
                score=summary["average_formatting_score"] / 5.0,
                comment=f"Element types: {summary.get('element_type_usage', {})}",
            ))
        
        if "total_sections" in checks and "sections_with_html" in checks:
            total = checks["total_sections"]
            with_html = checks["sections_with_html"]
            results.append(EvaluationResult(
                key="html_completeness",
                score=with_html / total if total > 0 else 0,
                comment=f"{with_html}/{total} sections have HTML",
            ))
    
    # --- Overall Metrics ---
    if state.overall_result:
        # Coherence (LLM-based)
        coherence = state.overall_result.get("coherence", {})
        if "score" in coherence:
            results.append(EvaluationResult(
                key="overall_coherence",
                score=coherence["score"] / 5.0 if coherence["score"] else 0,
                comment=coherence.get("reasoning", "No reasoning"),
            ))
        
        # Completeness
        completeness = state.overall_result.get("completeness", {})
        if completeness:
            comp_score = (
                completeness.get("theory_completeness", 0) +
                completeness.get("activities_completeness", 0) +
                completeness.get("html_completeness", 0)
            ) / 3
            results.append(EvaluationResult(
                key="overall_completeness",
                score=comp_score,
                comment=f"Theory: {completeness.get('theory_completeness', 0):.0%}, "
                        f"Activities: {completeness.get('activities_completeness', 0):.0%}, "
                        f"HTML: {completeness.get('html_completeness', 0):.0%}",
            ))
        
        # Structure metrics - Title Uniqueness (exact match)
        structure = state.overall_result.get("structure_metrics", {})
        if structure:
            uniqueness = structure.get("title_uniqueness", {})
            
            if "module_uniqueness" in uniqueness:
                results.append(EvaluationResult(
                    key="module_uniqueness",
                    score=uniqueness["module_uniqueness"],
                    comment=f"Exact match uniqueness for {uniqueness.get('total_modules', 0)} modules",
                ))
            
            if "submodule_uniqueness" in uniqueness:
                results.append(EvaluationResult(
                    key="submodule_uniqueness",
                    score=uniqueness["submodule_uniqueness"],
                    comment=f"Exact match uniqueness for {uniqueness.get('total_submodules', 0)} submodules",
                ))
            
            if "section_uniqueness" in uniqueness:
                results.append(EvaluationResult(
                    key="section_uniqueness",
                    score=uniqueness["section_uniqueness"],
                    comment=f"Exact match uniqueness for {uniqueness.get('total_sections', 0)} sections",
                ))
            
            # N-gram based uniqueness
            if "module_ngram_uniqueness" in uniqueness:
                results.append(EvaluationResult(
                    key="module_ngram_uniqueness",
                    score=uniqueness["module_ngram_uniqueness"],
                    comment="Weighted n-gram uniqueness for modules",
                ))
            
            if "submodule_ngram_uniqueness" in uniqueness:
                results.append(EvaluationResult(
                    key="submodule_ngram_uniqueness",
                    score=uniqueness["submodule_ngram_uniqueness"],
                    comment="Weighted n-gram uniqueness for submodules",
                ))
            
            if "section_ngram_uniqueness" in uniqueness:
                results.append(EvaluationResult(
                    key="section_ngram_uniqueness",
                    score=uniqueness["section_ngram_uniqueness"],
                    comment="Weighted n-gram uniqueness for sections",
                ))
        
        # Embedding metrics
        embedding = state.overall_result.get("embedding_metrics", {})
        if embedding and "error" not in embedding:
            # Title embedding uniqueness
            title_emb = embedding.get("title_embedding", {})
            
            if "module_embedding_uniqueness" in title_emb:
                results.append(EvaluationResult(
                    key="module_embedding_uniqueness",
                    score=title_emb["module_embedding_uniqueness"],
                    comment="Embedding-based uniqueness for modules",
                ))
            
            if "submodule_embedding_uniqueness" in title_emb:
                results.append(EvaluationResult(
                    key="submodule_embedding_uniqueness",
                    score=title_emb["submodule_embedding_uniqueness"],
                    comment="Embedding-based uniqueness for submodules",
                ))
            
            if "section_embedding_uniqueness" in title_emb:
                results.append(EvaluationResult(
                    key="section_embedding_uniqueness",
                    score=title_emb["section_embedding_uniqueness"],
                    comment="Embedding-based uniqueness for sections",
                ))
            
            # Content similarity metrics
            content_sim = embedding.get("content_similarity", {})
            if content_sim and "error" not in content_sim:
                avg_sim = content_sim.get("avg_similarity", 0.5)
                results.append(EvaluationResult(
                    key="content_diversity",
                    score=round(1 - avg_sim, 4),
                    comment=f"Avg similarity: {avg_sim:.3f}",
                ))
                
                if "max_similarity" in content_sim:
                    results.append(EvaluationResult(
                        key="content_max_diversity",
                        score=round(1 - content_sim["max_similarity"], 4),
                        comment=f"Max similarity: {content_sim['max_similarity']:.3f}",
                    ))
                
                if "min_similarity" in content_sim:
                    results.append(EvaluationResult(
                        key="content_min_diversity",
                        score=round(1 - content_sim["min_similarity"], 4),
                        comment=f"Min similarity: {content_sim['min_similarity']:.3f}",
                    ))
                
                if "std_similarity" in content_sim:
                    # Lower std = more consistent diversity
                    results.append(EvaluationResult(
                        key="content_consistency",
                        score=round(1 - min(1, content_sim["std_similarity"] * 2), 4),
                        comment=f"Std similarity: {content_sim['std_similarity']:.3f}",
                    ))
    
    return results


# ---- Combined Evaluator (Single Trace via LangGraph) ----

def combined_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Single evaluator that runs the LangGraph evaluation workflow.
    Creates a single trace containing all evaluation steps.
    Manually logs all metrics as feedback to create separate columns.
    """
    client = Client()
    course_state = _get_course_state(example.inputs)
    course_title = example.inputs.get("course_title", "Unknown")
    
    print(f"\n📊 Evaluating: {course_title}")
    
    # Create evaluation graph state
    eval_state = EvalGraphState(
        course_state=course_state,
        provider=_eval_settings["provider"],
        max_retries=_eval_settings["max_retries"],
        steps=_eval_settings.get("steps"),
    )
    
    # Run the evaluation graph (creates single trace)
    graph = get_evaluation_graph(steps=_eval_settings.get("steps"))
    final_state = graph.invoke(eval_state)
    
    # Convert to EvalGraphState if needed (LangGraph returns dict)
    if isinstance(final_state, dict):
        final_state = EvalGraphState(**final_state)
    
    # Extract metrics from final state
    metrics = extract_metrics(final_state)
    print(f"   📈 Total metrics extracted: {len(metrics)}")
    
    # Log all metrics as feedback for UI columns
    overall_score = 0.0
    count = 0
    
    # Metrics to exclude from average (not normalized to 0-1)
    exclude_from_avg = {"word_count", "avg_sentence_length"}
    
    for metric in metrics:
        try:
            # Log feedback for UI columns
            client.create_feedback(
                run_id=run.id,
                key=metric.key,
                score=metric.score,
                comment=metric.comment,
            )
            
            # Track for average calculation (exclude non-normalized metrics)
            if metric.score is not None and metric.key not in exclude_from_avg:
                overall_score += metric.score
                count += 1
                
            # Print metric to console
            score_str = f"{metric.score:.3f}" if metric.score is not None else "N/A"
            print(f"      ✓ {metric.key}: {score_str}")
            
        except Exception as e:
            print(f"      ⚠ Failed to log feedback for {metric.key}: {e}")
    
    # Calculate and print average
    avg_score = overall_score / count if count > 0 else 0.0
    print(f"   📊 Average score: {avg_score:.3f}")
    
    # Return an EvaluationResult with average score
    # This satisfies LangSmith's requirement and creates a single summary column
    return EvaluationResult(
        key="average_score",
        score=avg_score,
        comment=f"Average of {count} metrics"
    )


# ---- Main Evaluation Functions ----

def run_evaluation(
    dataset_name: str,
    experiment_prefix: str = "course-eval",
    provider: str = "mistral",
    max_retries: int = 3,
    steps: Optional[List[str]] = None,
    example_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run evaluation experiment against a LangSmith dataset.
    
    Uses a single combined evaluator that runs the graph and logs metrics manually.
    
    Args:
        dataset_name: Name of the LangSmith dataset
        experiment_prefix: Prefix for experiment name (for comparison)
        provider: LLM provider for evaluation
        max_retries: Max retries for LLM calls
        steps: Optional list of specific evaluators to run
               (e.g., ['index_evaluator', 'overall_evaluator'])
               If None, runs all evaluators
        example_ids: Optional list of specific example IDs to evaluate
                    If None, evaluates all examples in the dataset
        
    Returns:
        Experiment results
    """
    print(f"\n🔬 Running evaluation experiment")
    print(f"   Dataset: {dataset_name}")
    print(f"   Experiment prefix: {experiment_prefix}")
    print(f"   Provider: {provider}")
    if steps:
        print(f"   Evaluators: {', '.join(steps)}")
    else:
        print(f"   Evaluators: all (full evaluation)")
    if example_ids:
        print(f"   Examples to evaluate: {len(example_ids)}")
    else:
        print(f"   Examples to evaluate: all in dataset")
    print("-" * 50)
    
    # Set global evaluator settings
    set_evaluator_settings(provider=provider, max_retries=max_retries, steps=steps)
    
    # Define a simple target function (identity - we're evaluating stored outputs)
    def target(inputs: dict) -> dict:
        """Target function - returns inputs as outputs for evaluation."""
        return {"course_title": inputs.get("course_title", "Unknown")}
    
    # Prepare data parameter for evaluate()
    # If specific examples are requested, filter by creating example list
    if example_ids:
        # Get client and fetch specific examples
        client = get_client()
        dataset = next((ds for ds in client.list_datasets() if ds.name == dataset_name), None)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        # Create a generator of specific examples
        def get_examples():
            for example_id in example_ids:
                try:
                    example = client.read_example(example_id)
                    yield example
                except Exception as e:
                    print(f"   ⚠ Warning: Could not read example {example_id}: {e}")
        
        data_param = list(get_examples())
    else:
        data_param = dataset_name
    
    # Run evaluation with single combined evaluator
    # This evaluator manually logs all metrics as feedback, creating separate columns
    results = evaluate(
        target,
        data=data_param,
        evaluators=[combined_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,  # Sequential to ensure proper feedback logging
    )
    
    return results


def quick_evaluate(
    input_file: str,
    provider: str = "mistral",
    max_retries: int = 3,
    experiment_prefix: str = "quick-eval",
    steps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Quick evaluation of a single file (creates temporary dataset automatically).
    
    This function:
    - Creates or reuses a dataset based on the course title
    - Adds the file to the dataset (or reuses existing if duplicate)
    - Evaluates ONLY the specified file (not all examples in the dataset)
    
    Args:
        input_file: Path to course JSON file
        provider: LLM provider for evaluation
        max_retries: Max retries for LLM calls
        experiment_prefix: Prefix for experiment name
        steps: Optional list of specific evaluators to run
               If None, runs all evaluators (full evaluation)
        
    Returns:
        Experiment results
    """
    # Create dataset with single file - name will be auto-generated from course title
    print(f"\n📦 Preparing dataset...")
    dataset_id, dataset_name, example_ids = create_dataset(
        [input_file], 
        dataset_name=None, 
        description="Dataset for course evaluation",
        use_existing=True  # Reuse existing dataset for quick evaluations
    )
    
    # Run evaluation on ONLY this specific example
    results = run_evaluation(
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
        provider=provider,
        max_retries=max_retries,
        steps=steps,
        example_ids=example_ids,  # Evaluate only the specific file
    )
    
    return results


# ---- CLI ----

def main():
    """Command line interface for evaluation workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Course Evaluation Workflow")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation on a dataset")
    eval_parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    eval_parser.add_argument("--experiment-prefix", type=str, default="course-eval", help="Experiment prefix")
    eval_parser.add_argument("--provider", type=str, default="mistral", help="LLM provider")
    eval_parser.add_argument("--max-retries", type=int, default=3, help="Max retries for LLM calls")
    eval_parser.add_argument("--steps", nargs="*", type=str, default=None, 
                           help="Specific evaluators to run (e.g., index_evaluator overall_evaluator). If not specified, runs all evaluators.")
    
    # Quick evaluate command
    quick_parser = subparsers.add_parser("quick", help="Quick evaluation of a single file (auto-creates dataset)")
    quick_parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    quick_parser.add_argument("--provider", type=str, default="mistral", help="LLM provider")
    quick_parser.add_argument("--max-retries", type=int, default=3, help="Max retries for LLM calls")
    quick_parser.add_argument("--experiment-prefix", type=str, default="quick-eval", help="Experiment prefix")
    quick_parser.add_argument("--steps", nargs="*", type=str, default=None,
                           help="Specific evaluators to run (e.g., index_evaluator overall_evaluator). If not specified, runs all evaluators.")
    
    args = parser.parse_args()
    
    if args.command == "evaluate":
        results = run_evaluation(
            dataset_name=args.dataset,
            experiment_prefix=args.experiment_prefix,
            provider=args.provider,
            max_retries=args.max_retries,
            steps=args.steps if args.steps else None,
        )
        
    elif args.command == "quick":
        results = quick_evaluate(
            input_file=args.input,
            provider=args.provider,
            max_retries=args.max_retries,
            experiment_prefix=args.experiment_prefix,
            steps=args.steps if args.steps else None,
        )
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
