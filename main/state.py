from typing import List
from pydantic import BaseModel, Field

# ---- Section level ----
class Section(BaseModel):
    title: str = Field(..., description="Title of the section")
    theory: str = Field(
        default="", 
        description="Text of the section, expected to be ~n_words words. Can be empty initially for skeleton generation."
    )
    html: str = Field(
        default="",
        description="HTML formatted version of the theory content. Generated after theory content."
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
    total_pages: int = Field(default=50, description="Total number of pages for the course")
    words_per_page: int = Field(default=400, description="Target words per page for content estimation")
    description: str = Field(default="", description="Optional description or context for the course")
    language: str = Field(default="English", description="Language for the content generation")
    max_retries: int = Field(default=3, description="Maximum number of retries for generation")
    concurrency: int = Field(default=8, description="Number of concurrent section theory generations")

# ---- Course State ----
class CourseState(BaseModel):
    """Main course state - agents should only modify content fields"""
    # Configuration (should not be modified by agents)
    config: CourseConfig = Field(..., description="Course generation configuration")
    
    # Content fields (can be modified by agents)
    title: str = Field(..., description="Title of the course")
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