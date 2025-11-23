from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field

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

# ---- HTML Models ----
class HtmlElement(BaseModel):
    type: Literal["p", "ul", "quote", "table"] = Field(..., description="Type of HTML element")
    content: Union[str, List[str], dict] = Field(..., description="Content of the element")

class HtmlItem(BaseModel):
    title: str = Field(..., description="Title of the content item")
    icon: str = Field(..., description="Material Design Icon class")
    elements: List[HtmlElement] = Field(..., min_length=2, description="List of HTML elements")

class HtmlContent(BaseModel):
    format: Literal["tabs", "accordion", "timeline", "cards"] = Field(..., description="Format type for content display")
    items: List[HtmlItem] = Field(..., min_length=3, description="List of content items")

class HtmlStructure(BaseModel):
    intro: HtmlElement = Field(..., description="Introduction paragraph (type must be 'p')")
    content: HtmlContent = Field(..., description="Main content structure")
    conclusion: HtmlElement = Field(..., description="Conclusion paragraph (type must be 'p')")

# ---- Section level ----
class Section(BaseModel):
    title: str = Field(..., description="Title of the section")
    theory: str = Field(
        default="", 
        description="Text of the section, expected to be ~n_words words. Can be empty initially for skeleton generation."
    )
    glossary: List[GlossaryTerm] = Field(
        default_factory=list,
        description="Glossary terms for the section"
    )
    key_concept: str = Field(
        default="",
        description="Key concept summary for the section"
    )
    activities: List[Activity] = Field(
        default_factory=list,
        description="Interactive activities for the section"
    )
    final_activities: List[FinalActivity] = Field(
        default_factory=list,
        description="Final assessment activities for the section"
    )
    html_structure: Optional[HtmlStructure] = Field(
        default=None,
        description="Structured HTML format for the section content"
    )

# ---- Submodule level ----
class Submodule(BaseModel):
    title: str = Field(..., description="Title of the submodule")
    sections: List[Section] = Field(
        ..., description="List of sections in this submodule"
    )

# ---- Module level ----
class Module(BaseModel):
    title: str = Field(..., description="Title of the module")
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
    html_format: Literal["tabs", "accordion", "timeline", "cards"] = Field(default="tabs", description="Format for HTML content display")
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