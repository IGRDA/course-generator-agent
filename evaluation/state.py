"""Evaluation state models for storing evaluation results."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class EvaluatorResult(BaseModel):
    """Result from a single evaluator."""
    evaluator: str = Field(..., description="Name of the evaluator")
    status: str = Field(default="pending", description="Status: pending, completed, failed")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation results")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    duration_seconds: Optional[float] = Field(default=None, description="Time taken to evaluate")


class EvaluationSummary(BaseModel):
    """Summary of all evaluations."""
    total_evaluators: int = Field(default=0, description="Total number of evaluators run")
    completed: int = Field(default=0, description="Number of successful evaluations")
    failed: int = Field(default=0, description="Number of failed evaluations")
    average_llm_score: Optional[float] = Field(default=None, description="Average LLM-as-judge score")
    all_schema_checks_passed: bool = Field(default=False, description="Whether all schema checks passed")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation run."""
    input_file: str = Field(..., description="Path to input JSON file")
    provider: str = Field(default="mistral", description="LLM provider for evaluation")
    max_retries: int = Field(default=3, description="Max retries for LLM calls")
    run_index: bool = Field(default=True, description="Run index evaluator")
    run_sections: bool = Field(default=True, description="Run section evaluator")
    run_activities: bool = Field(default=True, description="Run activities evaluator")
    run_html: bool = Field(default=True, description="Run HTML evaluator")
    run_overall: bool = Field(default=True, description="Run overall evaluator")


class EvaluationState(BaseModel):
    """
    Main state for evaluation workflow.
    Stores configuration, results from each evaluator, and summary.
    """
    # Configuration
    config: EvaluationConfig = Field(..., description="Evaluation configuration")
    
    # Metadata
    course_title: str = Field(default="", description="Title of the course being evaluated")
    started_at: Optional[str] = Field(default=None, description="ISO timestamp when evaluation started")
    completed_at: Optional[str] = Field(default=None, description="ISO timestamp when evaluation completed")
    
    # Results from each evaluator
    index_result: Optional[EvaluatorResult] = Field(default=None, description="Index evaluator result")
    section_result: Optional[EvaluatorResult] = Field(default=None, description="Section evaluator result")
    activities_result: Optional[EvaluatorResult] = Field(default=None, description="Activities evaluator result")
    html_result: Optional[EvaluatorResult] = Field(default=None, description="HTML evaluator result")
    overall_result: Optional[EvaluatorResult] = Field(default=None, description="Overall evaluator result")
    
    # Summary
    summary: Optional[EvaluationSummary] = Field(default=None, description="Evaluation summary")
    
    def to_report(self) -> Dict[str, Any]:
        """Convert state to a report dictionary."""
        return {
            "course_title": self.course_title,
            "config": self.config.model_dump(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": {
                "index": self.index_result.model_dump() if self.index_result else None,
                "section": self.section_result.model_dump() if self.section_result else None,
                "activities": self.activities_result.model_dump() if self.activities_result else None,
                "html": self.html_result.model_dump() if self.html_result else None,
                "overall": self.overall_result.model_dump() if self.overall_result else None,
            },
            "summary": self.summary.model_dump() if self.summary else None,
        }

