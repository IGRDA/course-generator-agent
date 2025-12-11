from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field, model_validator

# ---- Activity Models ----
class GlossaryTerm(BaseModel):
    term: str = Field(..., description="Glossary term")
    explanation: str = Field(..., description="Explanation of the term")

class OrderListContent(BaseModel):
    question: str = Field(..., description="Question for order list activity")
    solution: List[str] = Field(..., min_length=2, description="Ordered list of items")

class FillGapsContent(BaseModel):
    question: str = Field(..., description="Question with *blanquito* placeholders")
    solution: List[str] = Field(..., min_length=2, description="List of words to fill gaps")

class SwipperContent(BaseModel):
    question: str = Field(..., description="Question for swipper activity")
    solution: dict = Field(..., description="Dictionary with 'true' and 'false' arrays")

class LinkingTermsContent(BaseModel):
    question: str = Field(..., description="Question for linking terms")
    solution: List[dict] = Field(..., min_length=2, description="List of concept-related pairs")

class MultipleChoiceContent(BaseModel):
    question: str = Field(..., description="Multiple choice question")
    solution: str = Field(..., description="Correct answer")
    other_options: List[str] = Field(..., min_length=3, description="Incorrect options")

class MultiSelectionContent(BaseModel):
    question: str = Field(..., description="Multi-selection question")
    solution: List[str] = Field(..., min_length=1, description="Correct answers")
    other_options: List[str] = Field(..., min_length=1, description="Incorrect options")

class Activity(BaseModel):
    type: Literal["order_list", "fill_gaps", "swipper", "linking_terms", "multiple_choice", "multi_selection"] = Field(..., description="Type of activity")
    content: Union[OrderListContent, FillGapsContent, SwipperContent, LinkingTermsContent, MultipleChoiceContent, MultiSelectionContent] = Field(..., description="Activity content")

class FinalActivityContent(BaseModel):
    question: str = Field(..., description="Final activity question or task description")

class FinalActivity(BaseModel):
    type: Literal["group_activity", "discussion_forum", "individual_project", "open_ended_quiz"] = Field(..., description="Type of final activity")
    content: FinalActivityContent = Field(..., description="Final activity content")

# ---- Meta Elements ----
class MetaElements(BaseModel):
    glossary: List[GlossaryTerm] = Field(default_factory=list, description="Glossary terms for the section")
    key_concept: str = Field(default="", description="Key concept summary for the section")
    interesting_fact: str = Field(default="", description="Interesting fact related to the section")
    quote: Optional[dict] = Field(default=None, description="Quote with author and text fields")

# ---- Activities Section ----
class ActivitiesSection(BaseModel):
    quiz: List[Activity] = Field(default_factory=list, description="Quiz activities for the section")
    application: List[FinalActivity] = Field(default_factory=list, description="Application activities for the section")

# ---- HTML Models ----
class HtmlElement(BaseModel):
    """HTML element with type-specific content structure.
    
    Content types by element type:
    - 'p': str (paragraph text)
    - 'ul': List[str] (unordered list items)
    - 'quote', 'table': Dict[str, Any] (structured data)
    - 'paragraphs', 'accordion', 'tabs', 'carousel', 'flip', 'timeline', 'conversation': List[ParagraphBlock]
      (all interactive formats use the same block structure)
    """
    type: Literal["p", "ul", "quote", "table", "paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"] = Field(..., description="Type of HTML element")
    content: Union[str, List[str], Dict[str, Any], List['ParagraphBlock']] = Field(..., description="Content of the element")
    
    @model_validator(mode='after')
    def validate_content_type(self) -> 'HtmlElement':
        """Ensure content type matches element type."""
        element_type = self.type
        content = self.content
        
        # Simple text paragraph
        if element_type == "p":
            if not isinstance(content, str):
                raise ValueError(f"Element type 'p' requires content to be a string, got {type(content).__name__}")
        
        # Unordered list
        elif element_type == "ul":
            if not isinstance(content, list) or not all(isinstance(item, str) for item in content):
                raise ValueError(f"Element type 'ul' requires content to be a list of strings")
        
        # Structured elements (quote, table)
        elif element_type in ["quote", "table"]:
            if not isinstance(content, dict):
                raise ValueError(f"Element type '{element_type}' requires content to be a dictionary")
        
        # Interactive formats (ALL use the same ParagraphBlock structure)
        elif element_type in ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]:
            if not isinstance(content, list):
                raise ValueError(f"Interactive format '{element_type}' requires content to be a list of ParagraphBlock")
            # Check that items are ParagraphBlock (will be validated by Pydantic)
            for idx, block in enumerate(content):
                if not isinstance(block, (dict, ParagraphBlock)):
                    raise ValueError(f"Interactive format '{element_type}' block {idx} must be a ParagraphBlock")
        
        return self
class ParagraphBlock(BaseModel):
    title: str = Field(..., description="Title of the paragraph block")
    icon: str = Field(..., description="Material Design Icon class")
    elements: List[HtmlElement] = Field(..., description="List of HTML elements within this block")

# Update forward references for recursive definition
HtmlElement.model_rebuild()

# ---- Section level ----
class Section(BaseModel):
    title: str = Field(..., description="Title of the section")
    index: int = Field(default=0, description="Section index within submodule")
    description: str = Field(default="", description="Description of the section")
    
    theory: str = Field(
        default="", 
        description="Theory text content for the section"
    )
    
    html: Optional[List[HtmlElement]] = Field(
        default=None,
        description="Structured HTML format as direct array of elements"
    )
    
    meta_elements: Optional[MetaElements] = Field(
        default=None,
        description="Metadata elements: glossary, key concepts, facts, quotes"
    )
    
    activities: Optional[ActivitiesSection] = Field(
        default=None,
        description="Activities section with quiz and application activities"
    )

# ---- Submodule level ----
class Submodule(BaseModel):
    title: str = Field(..., description="Title of the submodule")
    index: int = Field(default=0, description="Submodule index within module")
    description: str = Field(default="", description="Description of the submodule")
    duration: float = Field(default=0.0, description="Duration in hours")
    
    sections: List[Section] = Field(
        ..., description="List of sections in this submodule"
    )

# ---- Module level ----
class Module(BaseModel):
    title: str = Field(..., description="Title of the module")
    id: str = Field(default="", description="Simple string identifier matching index")
    index: int = Field(default=0, description="Module index in course")
    description: str = Field(default="", description="Description of the module")
    duration: float = Field(default=0.0, description="Duration in hours")
    type: Literal["module"] = Field(default="module", description="Type identifier")
    
    submodules: List[Submodule] = Field(
        ..., description="List of submodules in this module"
    )

# ---- Course Configuration ----
class CourseConfig(BaseModel):
    """Configuration parameters for course generation - should not be modified by agents"""
    title: str = Field(default="", description="Initial title of the course")
    text_llm_provider: str = Field(default="mistral", description="LLM provider for text generation (mistral | gemini | groq | openai)")
    web_search_provider: str = Field(default="ddg", description="Web search provider (ddg | tavily)")
    total_pages: int = Field(default=50, description="Total number of pages for the course")
    words_per_page: int = Field(default=400, description="Target words per page for content estimation")
    description: str = Field(default="", description="Optional description or context for the course")
    language: str = Field(default="English", description="Language for the content generation")
    pdf_syllabus_path: str = Field(default="", description="Path to PDF syllabus file")
    max_retries: int = Field(default=3, description="Maximum number of retries for generation")
    concurrency: int = Field(default=8, description="Number of concurrent section theory generations")
    use_reflection: bool = Field(default=False, description="Whether to use reflection pattern for fact verification")
    num_reflection_queries: int = Field(default=5, description="Number of verification queries to generate during reflection")
    
    # Activities configuration
    activities_concurrency: int = Field(default=8, description="Number of concurrent section activity generations")
    activity_selection_mode: Literal["random", "deterministic"] = Field(default="deterministic", description="How to select activity types")
    num_activities_per_section: int = Field(default=2, description="Number of quiz activities per section (in addition to multiple_choice and multi_selection)")
    
    # HTML configuration
    html_concurrency: int = Field(default=8, description="Number of concurrent HTML structure generations")
    select_html: Literal["LLM", "random"] = Field(default="LLM", description="HTML format selection mode: LLM chooses or random selection")
    html_formats: str = Field(default="paragraphs|accordion|tabs|carousel|flip|timeline|conversation", description="Available HTML formats (pipe-separated)")
    html_random_seed: int = Field(default=42, description="Seed for deterministic random format selection")
    include_quotes_in_html: bool = Field(default=False, description="Whether to include quote elements in HTML structure")
    include_tables_in_html: bool = Field(default=False, description="Whether to include table elements in HTML structure")

# ---- Course State ----
class CourseState(BaseModel):
    """Main course state - agents should only modify content fields"""
    # Configuration (should not be modified by agents)
    config: CourseConfig = Field(..., description="Course generation configuration")
    
    # Content fields (can be modified by agents)
    title: str = Field(..., description="Title of the course (initialized from config, can be refined by agents)")
    modules: List[Module] = Field(
        default_factory=list, description="Full course structure with all modules"
    )
    
    @property
    def description(self) -> str:
        return self.config.description
    
    @property
    def language(self) -> str:
        return self.config.language
    
    @property
    def max_retries(self) -> int:
        return self.config.max_retries
    
    @property
    def concurrency(self) -> int:
        return self.config.concurrency
