"""Base evaluator with shared LLM-as-judge logic and retry pattern."""

from typing import Type, TypeVar, Iterator, Tuple
from pydantic import BaseModel, Field
from langchain.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from LLMs.text2text import create_text_llm, resolve_text_model_name
from main.state import CourseState


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


class SingleCriteriaScore(BaseModel):
    """Generic single-criteria score for evaluations with one metric."""
    score: RubricScore = Field(..., description="Evaluation score")


T = TypeVar('T', bound=BaseModel)


class BaseEvaluator:
    """
    Base class for all evaluators with shared LLM creation, retry logic,
    and course iteration helpers.
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
            llm_kwargs = {"temperature": 0}
            if model_name:
                llm_kwargs["model_name"] = model_name
            self._llm = create_text_llm(provider=self.provider, **llm_kwargs)
        return self._llm
    
    # ---- Course Iteration Helpers ----
    
    def iter_sections(self, course_state: CourseState) -> Iterator[Tuple[str, any, any, any]]:
        """
        Iterate over all sections in a course.
        
        Yields:
            Tuple of (section_id, module, submodule, section)
        """
        for m_idx, module in enumerate(course_state.modules):
            for sm_idx, submodule in enumerate(module.submodules):
                for s_idx, section in enumerate(submodule.sections):
                    yield f"{m_idx+1}.{sm_idx+1}.{s_idx+1}", module, submodule, section
    
    def count_sections(self, course_state: CourseState) -> int:
        """Count total sections in a course."""
        return sum(
            len(sm.sections)
            for m in course_state.modules
            for sm in m.submodules
        )
    
    # ---- LLM Evaluation ----
    
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
        prompt_variables["format_instructions"] = parser.get_format_instructions()
        
        chain = prompt | self.llm | StrOutputParser()
        fix_parser = RetryWithErrorOutputParser.from_llm(
            llm=self.llm,
            parser=parser,
            max_retries=self.max_retries,
        )
        
        raw = chain.invoke(prompt_variables)
        
        try:
            return parser.parse(raw)
        except Exception as e:
            if correction_prompt is None:
                correction_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Fix the JSON output to match the required schema."),
                    ("human", """The previous output had an error: {error}

Original output:
{completion}

Required format:
{format_instructions}

Please provide a corrected JSON output.""")
                ])
            
            return fix_parser.parse_with_prompt(
                completion=raw,
                prompt_value=correction_prompt.format_prompt(
                    error=str(e),
                    completion=raw,
                    format_instructions=parser.get_format_instructions(),
                ),
            )
    
    def compute_average_score(self, scores: list[RubricScore]) -> float:
        """Compute average score from a list of RubricScores."""
        if not scores:
            return 0.0
        return sum(s.score for s in scores) / len(scores)
