"""Evaluation workflow using LangSmith Evaluations API.

Uses proper LangSmith experiments, datasets, and evaluators for:
- Experiment tracking and comparison in the LangSmith UI
- Custom LLM-as-judge evaluators with rubric scoring
- Single trace per complete evaluation (via LangGraph)

Usage:
    # Create a dataset from course outputs
    python -m evaluation.workflow create-dataset --inputs output/*.json --name my-courses
    
    # Run evaluation experiment
    python -m evaluation.workflow evaluate --dataset my-courses --experiment-prefix v1
    
    # Quick evaluation of a single file (creates temp dataset)
    python -m evaluation.workflow quick --input output/course.json
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult
from langsmith.schemas import Example, Run
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from main.state import CourseState
from evaluators.index_evaluator import IndexEvaluator
from evaluators.section_evaluator import SectionEvaluator
from evaluators.activities_evaluator import ActivitiesEvaluator
from evaluators.html_evaluator import HtmlEvaluator
from evaluators.overall_evaluator import OverallEvaluator


# ---- Utility Functions ----

def load_course_state(input_file: str) -> CourseState:
    """Load CourseState from JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return CourseState.model_validate(data)


def get_client() -> Client:
    """Get LangSmith client."""
    return Client()


# ---- Dataset Management ----

def create_dataset(
    input_files: List[str],
    dataset_name: str,
    description: str = "Course generation outputs for evaluation"
) -> str:
    """
    Create a LangSmith dataset from course JSON files.
    
    Args:
        input_files: List of paths to course JSON files
        dataset_name: Name for the dataset
        description: Dataset description
        
    Returns:
        Dataset ID
    """
    client = get_client()
    
    # Create dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )
    
    print(f"📦 Created dataset: {dataset_name} (ID: {dataset.id})")
    
    # Add examples
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"   ⚠ Skipping {file_path} - file not found")
            continue
            
        try:
            course_state = load_course_state(file_path)
            
            # Store course data as the example input
            client.create_example(
                inputs={
                    "file_path": file_path,
                    "course_title": course_state.title,
                    "course_data": course_state.model_dump(),
                },
                outputs={},  # No ground truth for generative tasks
                dataset_id=dataset.id,
                metadata={
                    "num_modules": len(course_state.modules),
                    "language": course_state.language,
                }
            )
            print(f"   ✓ Added: {course_state.title}")
        except Exception as e:
            print(f"   ✗ Failed to add {file_path}: {e}")
    
    print(f"\n✅ Dataset created with {len(input_files)} examples")
    return str(dataset.id)


def list_datasets():
    """List all available datasets."""
    client = get_client()
    datasets = list(client.list_datasets())
    
    print("\n📚 Available Datasets:")
    print("-" * 60)
    for ds in datasets:
        print(f"  {ds.name}")
        print(f"    ID: {ds.id}")
        print(f"    Examples: {ds.example_count}")
        print(f"    Created: {ds.created_at}")
        print()


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
    quick: bool = False
    
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
    try:
        evaluator = IndexEvaluator(provider=state.provider, max_retries=state.max_retries)
        result = evaluator.evaluate(state.course_state)
        print("   ✓ Index evaluation complete")
        return {"index_result": result}
    except Exception as e:
        print(f"   ✗ Index evaluation failed: {e}")
        return {"index_result": {"error": str(e)}}


def run_section_evaluation(state: EvalGraphState) -> Dict[str, Any]:
    """Run section evaluator and return result."""
    if state.quick:
        print("   Skipping section evaluation (quick mode)")
        return {"section_result": {"skipped": True}}
    
    print("   Running section evaluation...")
    try:
        evaluator = SectionEvaluator(provider=state.provider, max_retries=state.max_retries)
        result = evaluator.evaluate(state.course_state)
        print("   ✓ Section evaluation complete")
        return {"section_result": result}
    except Exception as e:
        print(f"   ✗ Section evaluation failed: {e}")
        return {"section_result": {"error": str(e)}}


def run_activities_evaluation(state: EvalGraphState) -> Dict[str, Any]:
    """Run activities evaluator and return result."""
    if state.quick:
        # Quick mode: just compute basic completeness
        print("   Running activities completeness check (quick mode)...")
        try:
            total_sections = sum(
                len(sm.sections)
                for m in state.course_state.modules
                for sm in m.submodules
            )
            with_activities = sum(
                1 for m in state.course_state.modules
                for sm in m.submodules
                for s in sm.sections
                if s.other_elements and s.other_elements.activities
            )
            return {"activities_result": {
                "quick_check": True,
                "schema_checks": {
                    "total_sections": total_sections,
                    "sections_with_activities": with_activities,
                }
            }}
        except Exception as e:
            return {"activities_result": {"error": str(e)}}
    
    print("   Running activities evaluation...")
    try:
        evaluator = ActivitiesEvaluator(provider=state.provider, max_retries=state.max_retries)
        result = evaluator.evaluate(state.course_state)
        print("   ✓ Activities evaluation complete")
        return {"activities_result": result}
    except Exception as e:
        print(f"   ✗ Activities evaluation failed: {e}")
        return {"activities_result": {"error": str(e)}}


def run_html_evaluation(state: EvalGraphState) -> Dict[str, Any]:
    """Run HTML evaluator and return result."""
    if state.quick:
        # Quick mode: just compute basic completeness
        print("   Running HTML completeness check (quick mode)...")
        try:
            total_sections = sum(
                len(sm.sections)
                for m in state.course_state.modules
                for sm in m.submodules
            )
            with_html = sum(
                1 for m in state.course_state.modules
                for sm in m.submodules
                for s in sm.sections
                if s.html and s.html.theory
            )
            return {"html_result": {
                "quick_check": True,
                "schema_checks": {
                    "total_sections": total_sections,
                    "sections_with_html": with_html,
                }
            }}
        except Exception as e:
            return {"html_result": {"error": str(e)}}
    
    print("   Running HTML evaluation...")
    try:
        evaluator = HtmlEvaluator(provider=state.provider, max_retries=state.max_retries)
        result = evaluator.evaluate(state.course_state)
        print("   ✓ HTML evaluation complete")
        return {"html_result": result}
    except Exception as e:
        print(f"   ✗ HTML evaluation failed: {e}")
        return {"html_result": {"error": str(e)}}


def run_overall_evaluation(state: EvalGraphState) -> Dict[str, Any]:
    """Run overall evaluator and return result."""
    print("   Running overall evaluation...")
    try:
        evaluator = OverallEvaluator(provider=state.provider, max_retries=state.max_retries)
        result = evaluator.evaluate(state.course_state)
        print("   ✓ Overall evaluation complete")
        return {"overall_result": result}
    except Exception as e:
        print(f"   ✗ Overall evaluation failed: {e}")
        return {"overall_result": {"error": str(e)}}


# ---- Build Graph ----

_evaluation_graph = None


def get_evaluation_graph():
    """Build and cache the evaluation StateGraph."""
    global _evaluation_graph
    if _evaluation_graph is None:
        graph = StateGraph(EvalGraphState)
        
        # Add nodes for each evaluator
        graph.add_node("index_eval", run_index_evaluation)
        graph.add_node("section_eval", run_section_evaluation)
        graph.add_node("activities_eval", run_activities_evaluation)
        graph.add_node("html_eval", run_html_evaluation)
        graph.add_node("overall_eval", run_overall_evaluation)
        
        # Define sequential flow
        graph.add_edge(START, "index_eval")
        graph.add_edge("index_eval", "section_eval")
        graph.add_edge("section_eval", "activities_eval")
        graph.add_edge("activities_eval", "html_eval")
        graph.add_edge("html_eval", "overall_eval")
        graph.add_edge("overall_eval", END)
        
        _evaluation_graph = graph.compile()
    
    return _evaluation_graph


# ---- Global Evaluator Settings ----

_eval_settings = {"provider": "mistral", "max_retries": 3, "quick": False}


def set_evaluator_settings(provider: str = "mistral", max_retries: int = 3, quick: bool = False):
    """Set global evaluator settings."""
    global _eval_settings
    _eval_settings = {"provider": provider, "max_retries": max_retries, "quick": quick}


# ---- Metrics Extraction ----

def extract_metrics(state: EvalGraphState) -> List[EvaluationResult]:
    """
    Extract all metrics from evaluation graph state.
    Converts evaluator results to LangSmith EvaluationResult format.
    """
    results = []
    
    # --- Index Metrics ---
    if state.index_result and "error" not in state.index_result:
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
    elif state.index_result and "error" in state.index_result:
        results.append(EvaluationResult(
            key="index_error",
            score=0,
            comment=f"Error: {state.index_result['error']}",
        ))
    
    # --- Section Metrics ---
    if state.section_result and "error" not in state.section_result and not state.section_result.get("skipped"):
        summary = state.section_result.get("summary", {})
        nlp = state.section_result.get("nlp_metrics", {})
        
        if "average_accuracy_score" in summary:
            results.append(EvaluationResult(
                key="section_accuracy",
                score=summary["average_accuracy_score"] / 5.0,
                comment=f"Evaluated {summary.get('total_sections', 0)} sections",
            ))
        
        readability = nlp.get("readability", {})
        if "flesch_reading_ease" in readability:
            flesch = readability["flesch_reading_ease"]
            results.append(EvaluationResult(
                key="section_readability",
                score=max(0, min(1, flesch / 100)),
                comment=f"Flesch: {flesch:.1f}, SMOG: {readability.get('smog_index', 'N/A')}",
            ))
        
        repetition = nlp.get("repetition", {})
        if "type_token_ratio" in repetition:
            ttr = repetition["type_token_ratio"]
            results.append(EvaluationResult(
                key="vocabulary_diversity",
                score=ttr,
                comment=f"TTR: {ttr:.3f}, Unique tokens: {repetition.get('unique_tokens', 'N/A')}",
            ))
    
    # --- Activities Metrics ---
    if state.activities_result and "error" not in state.activities_result:
        checks = state.activities_result.get("schema_checks", {})
        summary = state.activities_result.get("summary", {})
        
        # Quality (only in full mode)
        if "average_quality_score" in summary:
            results.append(EvaluationResult(
                key="activities_quality",
                score=summary["average_quality_score"] / 5.0,
                comment=f"Activity type coverage: {checks.get('activity_type_coverage', 0):.0%}",
            ))
        
        # Completeness (available in both modes)
        if "total_sections" in checks and "sections_with_activities" in checks:
            total = checks["total_sections"]
            with_activities = checks["sections_with_activities"]
            results.append(EvaluationResult(
                key="activities_completeness",
                score=with_activities / total if total > 0 else 0,
                comment=f"{with_activities}/{total} sections have activities",
            ))
    
    # --- HTML Metrics ---
    if state.html_result and "error" not in state.html_result:
        checks = state.html_result.get("schema_checks", {})
        summary = state.html_result.get("summary", {})
        
        # Formatting (only in full mode)
        if "average_formatting_score" in summary:
            results.append(EvaluationResult(
                key="html_formatting",
                score=summary["average_formatting_score"] / 5.0,
                comment=f"Element types: {summary.get('element_type_usage', {})}",
            ))
        
        # Completeness (available in both modes)
        if "total_sections" in checks and "sections_with_html" in checks:
            total = checks["total_sections"]
            with_html = checks["sections_with_html"]
            results.append(EvaluationResult(
                key="html_completeness",
                score=with_html / total if total > 0 else 0,
                comment=f"{with_html}/{total} sections have HTML",
            ))
    
    # --- Overall Metrics ---
    if state.overall_result and "error" not in state.overall_result:
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
        
        # Title Uniqueness (structure metrics)
        structure = state.overall_result.get("structure_metrics", {})
        if structure and "error" not in structure:
            uniqueness = structure.get("title_uniqueness", {})
            if "section_uniqueness" in uniqueness:
                results.append(EvaluationResult(
                    key="title_uniqueness",
                    score=uniqueness["section_uniqueness"],
                    comment=f"Unique sections: {uniqueness.get('unique_sections', 'N/A')}/{uniqueness.get('total_sections', 'N/A')}",
                ))
        
        # Content Diversity (embedding metrics)
        embedding = state.overall_result.get("embedding_metrics", {})
        if embedding and "error" not in embedding:
            avg_sim = embedding.get("avg_similarity", 0.5)
            results.append(EvaluationResult(
                key="content_diversity",
                score=1 - avg_sim,  # Lower similarity = higher diversity
                comment=f"Avg similarity: {avg_sim:.3f}, Flagged pairs: {embedding.get('num_flagged_pairs', 0)}",
            ))
    elif state.overall_result and "error" in state.overall_result:
        results.append(EvaluationResult(
            key="overall_error",
            score=0,
            comment=f"Error: {state.overall_result['error']}",
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
        quick=_eval_settings.get("quick", False),
    )
    
    # Run the evaluation graph (creates single trace)
    graph = get_evaluation_graph()
    final_state = graph.invoke(eval_state)
    
    # Convert to EvalGraphState if needed (LangGraph returns dict)
    if isinstance(final_state, dict):
        final_state = EvalGraphState(**final_state)
    
    # Extract metrics from final state
    metrics = extract_metrics(final_state)
    print(f"   📈 Total metrics: {len(metrics)}")
    
    # Return all metrics for local display
    final_results = {}
    overall_score = 0.0
    count = 0
    
    for metric in metrics:
        try:
            # Log feedback for UI columns
            client.create_feedback(
                run_id=run.id,
                key=metric.key,
                score=metric.score,
                comment=metric.comment,
            )
            
            # Add to return dict for local display
            if metric.score is not None:
                final_results[metric.key] = metric.score
                overall_score += metric.score
                count += 1
        except Exception as e:
            print(f"   ⚠ Failed to log feedback for {metric.key}: {e}")
    
    # Add average
    avg_score = overall_score / count if count > 0 else 0.0
    final_results["overall_avg_score"] = avg_score
    
    return final_results


# ---- Main Evaluation Functions ----

def run_evaluation(
    dataset_name: str,
    experiment_prefix: str = "course-eval",
    provider: str = "mistral",
    max_retries: int = 3,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation experiment against a LangSmith dataset.
    
    Uses a single combined evaluator that runs the graph and logs metrics manually.
    
    Args:
        dataset_name: Name of the LangSmith dataset
        experiment_prefix: Prefix for experiment name (for comparison)
        provider: LLM provider for evaluation
        max_retries: Max retries for LLM calls
        quick: Run only quick evaluators (skip expensive LLM calls)
        
    Returns:
        Experiment results
    """
    print(f"\n🔬 Running evaluation experiment")
    print(f"   Dataset: {dataset_name}")
    print(f"   Experiment prefix: {experiment_prefix}")
    print(f"   Provider: {provider}")
    print(f"   Mode: {'quick' if quick else 'full'}")
    print("-" * 50)
    
    # Set global evaluator settings
    set_evaluator_settings(provider=provider, max_retries=max_retries, quick=quick)
    
    # Define a simple target function (identity - we're evaluating stored outputs)
    def target(inputs: dict) -> dict:
        """Target function - returns inputs as outputs for evaluation."""
        return {"course_title": inputs.get("course_title", "Unknown")}
    
    # Run evaluation with single combined evaluator
    # This evaluator manually logs all metrics as feedback, creating separate columns
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[combined_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,  # Sequential to ensure proper feedback logging
    )
    
    print(f"\n✅ Experiment completed!")
    print(f"   View results at: https://smith.langchain.com")
    
    return results


def quick_evaluate(
    input_file: str,
    provider: str = "mistral",
    max_retries: int = 3,
    experiment_prefix: str = "quick-eval",
    full: bool = False,
) -> Dict[str, Any]:
    """
    Quick evaluation of a single file (creates temporary dataset).
    
    Args:
        input_file: Path to course JSON file
        provider: LLM provider for evaluation
        max_retries: Max retries for LLM calls
        experiment_prefix: Prefix for experiment name
        full: Run full evaluation (not just quick evaluators)
        
    Returns:
        Experiment results
    """
    # Create temporary dataset name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"temp-eval-{timestamp}"
    
    # Create dataset with single file
    print(f"\n📦 Creating temporary dataset...")
    create_dataset([input_file], dataset_name, "Temporary dataset for quick evaluation")
    
    # Run evaluation
    results = run_evaluation(
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
        provider=provider,
        max_retries=max_retries,
        quick=not full,
    )
    
    return results


def print_results_summary(results):
    """Print a summary of evaluation results."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    
    try:
        # Iterate through experiment results
        for result in results:
            # Access example inputs
            example = result.get("example") if isinstance(result, dict) else getattr(result, "example", None)
            if example:
                inputs = example.inputs if hasattr(example, "inputs") else example.get("inputs", {})
                course_title = inputs.get("course_title", "Unknown") if isinstance(inputs, dict) else "Unknown"
                print(f"\n  Example: {course_title}")
            
            # Access evaluation results
            eval_results = result.get("evaluation_results") if isinstance(result, dict) else getattr(result, "evaluation_results", None)
            
            if eval_results:
                # Check if results is a list (standard) or a dict (our custom return)
                results_data = eval_results.get("results") if isinstance(eval_results, dict) else getattr(eval_results, "results", None)
                
                # If it's a list of results (standard behavior)
                if isinstance(results_data, list):
                    for eval_result in results_data:
                        if isinstance(eval_result, dict):
                            key = eval_result.get("key", "unknown")
                            score = eval_result.get("score")
                        else:
                            key = getattr(eval_result, "key", "unknown")
                            score = getattr(eval_result, "score", None)
                        score_str = f"{score:.2f}" if score is not None else "N/A"
                        print(f"    {key}: {score_str}")
                
                # If it's a dictionary (our custom return from combined_evaluator)
                elif isinstance(results_data, dict):
                    for key, score in results_data.items():
                        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
                        print(f"    {key}: {score_str}")
                        
                # Fallback: try to print eval_results directly if it's the dict
                elif isinstance(eval_results, dict):
                     for key, val in eval_results.items():
                        if key != "results" and isinstance(val, (int, float)):
                             print(f"    {key}: {val:.2f}")
    except Exception as e:
        print(f"\n  Could not parse results: {e}")
        print("  Check LangSmith UI for detailed results.")
    
    print("\n" + "=" * 60)
    print("View detailed results and comparisons at: https://smith.langchain.com")


# ---- CLI ----

def main():
    """Command line interface for evaluation workflow."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Course Evaluation Workflow")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create dataset command
    create_parser = subparsers.add_parser("create-dataset", help="Create a LangSmith dataset")
    create_parser.add_argument("--inputs", type=str, required=True, help="Glob pattern for input JSON files")
    create_parser.add_argument("--name", type=str, required=True, help="Dataset name")
    create_parser.add_argument("--description", type=str, default="Course generation outputs", help="Dataset description")
    
    # List datasets command
    subparsers.add_parser("list-datasets", help="List available datasets")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation on a dataset")
    eval_parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    eval_parser.add_argument("--experiment-prefix", type=str, default="course-eval", help="Experiment prefix")
    eval_parser.add_argument("--provider", type=str, default="mistral", help="LLM provider")
    eval_parser.add_argument("--max-retries", type=int, default=3, help="Max retries for LLM calls")
    eval_parser.add_argument("--quick", action="store_true", help="Quick mode (skip expensive LLM calls)")
    
    # Quick evaluate command
    quick_parser = subparsers.add_parser("quick", help="Quick evaluation of a single file")
    quick_parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    quick_parser.add_argument("--provider", type=str, default="mistral", help="LLM provider")
    quick_parser.add_argument("--max-retries", type=int, default=3, help="Max retries for LLM calls")
    quick_parser.add_argument("--experiment-prefix", type=str, default="quick-eval", help="Experiment prefix")
    quick_parser.add_argument("--full", action="store_true", help="Run full evaluation (not quick)")
    
    args = parser.parse_args()
    
    if args.command == "create-dataset":
        input_files = glob.glob(args.inputs)
        if not input_files:
            print(f"No files found matching: {args.inputs}")
            return
        create_dataset(input_files, args.name, args.description)
        
    elif args.command == "list-datasets":
        list_datasets()
        
    elif args.command == "evaluate":
        results = run_evaluation(
            dataset_name=args.dataset,
            experiment_prefix=args.experiment_prefix,
            provider=args.provider,
            max_retries=args.max_retries,
            quick=args.quick,
        )
        print_results_summary(results)
        
    elif args.command == "quick":
        results = quick_evaluate(
            input_file=args.input,
            provider=args.provider,
            max_retries=args.max_retries,
            experiment_prefix=args.experiment_prefix,
            full=args.full,
        )
        print_results_summary(results)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
