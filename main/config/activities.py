"""Activities generation configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class ActivitiesConfig(BaseModel):
    """Configuration for section activities generation."""
    
    concurrency: int = Field(
        default=8,
        description="Number of concurrent section activity generations"
    )
    selection_mode: Literal["random", "deterministic"] = Field(
        default="deterministic",
        description="How to select activity types"
    )
    num_per_section: int = Field(
        default=2,
        description="Number of quiz activities per section (in addition to multiple_choice and multi_selection)"
    )

