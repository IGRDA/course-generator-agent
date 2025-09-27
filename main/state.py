from typing import List
from pydantic import BaseModel, Field

# ---- Section level ----
class Section(BaseModel):
    title: str = Field(..., description="Title of the section")
    theory: str = Field(
        default="", 
        description="Text of the section, expected to be ~n_words words. Can be empty initially for skeleton generation."
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

# ---- Course State ----
class CourseState(BaseModel):
    title: str = Field(..., description="Title of the course")
    n_modules: int = Field(..., description="Number of modules in the course")
    n_submodules: int = Field(..., description="Number of submodules per module")
    n_sections: int = Field(..., description="Number of sections per submodule")
    n_words: int = Field(..., description="Target number of words per section")
    
    modules: List[Module] = Field(
        ..., description="Full course structure with all modules"
    )
    
    # Workflow configuration fields
    total_pages: int = Field(default=100, description="Total number of pages for the course")
    description: str = Field(default="", description="Optional description or context for the course")
    language: str = Field(default="English", description="Language for the course content generation")
    max_retries: int = Field(default=3, description="Maximum number of retries for generation")
    concurrency: int = Field(default=8, description="Number of concurrent section theory generations")

    model_config = {
        "extra": "forbid"  # prevent stray fields
    }