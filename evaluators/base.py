"""Base evaluator with shared LLM-as-judge logic and retry pattern."""

from typing import Type, TypeVar
from pydantic import BaseModel, Field
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from LLMs.text2text import create_text_llm, resolve_text_model_name


# ---- Rubric Score Models ----
class RubricScore(BaseModel):
    """Standard rubric score with 1-5 scale."""
    score: int = Field(..., ge=1, le=5, description="Score from 1 (Poor) to 5 (Excellent)")
    reasoning: str = Field(..., description="Brief explanation for the score")


class MultiCriteriaScore(BaseModel):
    """Multiple rubric scores for different criteria."""
    coverage: RubricScore = Field(..., description="Topic coverage score")
    structure: RubricScore = Field(..., description="Logical structure score")
    balance: RubricScore = Field(..., description="Balance across sections score")


class SectionScore(BaseModel):
    """Score for section content evaluation."""
    accuracy: RubricScore = Field(..., description="Factual accuracy score")


class ActivityScore(BaseModel):
    """Score for activities evaluation."""
    quality: RubricScore = Field(..., description="Activity quality and relevance score")


class HtmlScore(BaseModel):
    """Score for HTML structure evaluation."""
    formatting: RubricScore = Field(..., description="HTML formatting quality score")


class OverallScore(BaseModel):
    """Score for overall course evaluation."""
    coherence: RubricScore = Field(..., description="Course-wide coherence score")


T = TypeVar('T', bound=BaseModel)


class BaseEvaluator:
    """
    Base class for all evaluators with shared LLM creation and retry logic.
    Uses the same patterns as the agents for consistency.
    """
    
    def __init__(self, provider: str = "mistral", max_retries: int = 3):
        """
        Initialize the evaluator.
        
        Args:
            provider: LLM provider name (mistral | gemini | groq | openai)
            max_retries: Maximum retry attempts for parsing
        """
        self.provider = provider
        self.max_retries = max_retries
        self._llm = None
    
    @property
    def llm(self):
        """Lazy-load the LLM."""
        if self._llm is None:
            model_name = resolve_text_model_name(self.provider)
            llm_kwargs = {"temperature": 0.0}  # Low temperature for consistent evaluation
            if model_name:
                llm_kwargs["model_name"] = model_name
            self._llm = create_text_llm(provider=self.provider, **llm_kwargs)
        return self._llm
    
    def evaluate_with_rubric(
        self,
        prompt: ChatPromptTemplate,
        output_model: Type[T],
        prompt_variables: dict,
        correction_prompt: ChatPromptTemplate | None = None
    ) -> T:
        """
        Run LLM-as-judge evaluation with structured output and retry logic.
        
        Args:
            prompt: The evaluation prompt template
            output_model: Pydantic model for structured output
            prompt_variables: Variables to fill the prompt
            correction_prompt: Optional correction prompt for retry
            
        Returns:
            Parsed evaluation result as Pydantic model
        """
        parser = PydanticOutputParser(pydantic_object=output_model)
        
        # Add format instructions to variables
        prompt_variables["format_instructions"] = parser.get_format_instructions()
        
        # Create chain
        chain = prompt | self.llm | StrOutputParser()
        
        # Create fix parser for retry
        fix_parser = RetryWithErrorOutputParser.from_llm(
            llm=self.llm,
            parser=parser,
            max_retries=self.max_retries,
        )
        
        # Invoke the chain
        raw = chain.invoke(prompt_variables)
        
        # Try to parse, with fallback to retry parser
        try:
            result = parser.parse(raw)
        except Exception as e:
            if correction_prompt is None:
                # Use default correction prompt
                correction_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Fix the JSON output to match the required schema."),
                    ("human", """The previous output had an error: {error}

Original output:
{completion}

Required format:
{format_instructions}

Please provide a corrected JSON output.""")
                ])
            
            result = fix_parser.parse_with_prompt(
                completion=raw,
                prompt_value=correction_prompt.format_prompt(
                    error=str(e),
                    completion=raw,
                    format_instructions=parser.get_format_instructions(),
                ),
            )
        
        return result
    
    def compute_average_score(self, scores: list[RubricScore]) -> float:
        """Compute average score from a list of RubricScores."""
        if not scores:
            return 0.0
        return sum(s.score for s in scores) / len(scores)

