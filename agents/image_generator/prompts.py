from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from typing import List


# ---- Image Query Generation Prompt ----
image_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at creating concise image search queries.
Generate a 5-10 word description that captures the visual concept.
Focus on concrete, visual elements that would make a good educational illustration.
Always keep the query relevant to the course topic and the block title and content preview."""),
    
    ("human", """Generate an image search query (3-5 words) for this content block:

Course Topic: {course_title}
Block Title: {block_title}
Content Preview: {content_preview}

The image must visually relate to "{course_title}". Return ONLY the search query, nothing else.""")
])


# ---- Image Ranking Schema ----
class ImageRankingScore(BaseModel):
    """Rubric-based score for evaluating an image's suitability for educational content."""
    
    alignment: int = Field(
        ge=0, le=5,
        description="How well the image aligns with the content (0=no relation, 5=perfect match)"
    )
    no_watermark: int = Field(
        ge=0, le=2,
        description="2 if the image has NO watermarks/logos, 0 if it has watermarks"
    )
    has_text: int = Field(
        ge=0, le=1,
        description="1 if the image contains readable text/labels, 0 if no text"
    )
    style: int = Field(
        ge=0, le=2,
        description="Style appropriateness (0=inappropriate, 1=acceptable, 2=professional/educational)"
    )
    total: int = Field(
        ge=0, le=10,
        description="Sum of all scores (alignment + no_watermark + has_text + style)"
    )
    
    @field_validator('total', mode='before')
    @classmethod
    def compute_total(cls, v, info):
        """Auto-compute total if not provided or validate if provided."""
        data = info.data
        computed = data.get('alignment', 0) + data.get('no_watermark', 0) + data.get('has_text', 0) + data.get('style', 0)
        if v is None:
            return computed
        return v


class ImageRankingResult(BaseModel):
    """Result of ranking multiple images."""
    
    scores: List[ImageRankingScore] = Field(
        description="List of scores for each image, in the same order as input images"
    )


# ---- Image Ranking Prompt (for Vision LLM) ----
IMAGE_RANKING_SYSTEM_PROMPT = """You are an expert image evaluator for educational content.
You will be shown one or more images and must score each one using the following rubric:

SCORING RUBRIC (Max 10 points total):
1. ALIGNMENT (0-5 points): How well does the image match the course topic and block content?
   - 0: No relation to the topic
   - 1-2: Loosely related
   - 3-4: Good match
   - 5: Perfect visual representation

2. NO WATERMARKS (0 or 2 points): Does the image have watermarks, logos, or stock photo marks?
   - 2: Clean image with NO watermarks
   - 0: Has visible watermarks or stock photo marks

3. HAS TEXT (0 or 1 point): Does the image contain readable text, labels, or annotations?
   - 0: Contains helpful text/labels
   - 1: No text visible

4. STYLE (0-2 points): Is the image style appropriate for educational/course material?
   - 0: Inappropriate (low quality, offensive, or unprofessional)
   - 1: Acceptable quality and no text
   - 2: Professional, high-quality, suitable for textbook/course

Return your evaluation as a JSON object with the exact structure specified."""


def create_image_ranking_prompt(
    course_title: str,
    block_title: str,
    content_preview: str,
    num_images: int
) -> str:
    """
    Create the human message prompt for image ranking.
    
    Args:
        course_title: The course topic
        block_title: The content block title
        content_preview: Preview of the block content
        num_images: Number of images being evaluated
    
    Returns:
        Formatted prompt string
    """
    return f"""Evaluate the following {num_images} image(s) for use in educational content.

CONTEXT:
- Course Topic: {course_title}
- Block Title: {block_title}
- Content Preview: {content_preview}

Score each image using the rubric. Return a JSON object with this exact structure:
{{
    "scores": [
        {{
            "alignment": <0-5>,
            "no_watermark": <0 or 2>,
            "has_text": <0 or 1>,
            "style": <0-2>,
            "total": <sum of above>
        }}
    ]
}}

The "scores" array must have exactly {num_images} element(s), one for each image in order.
Return ONLY the JSON object, no other text."""


# ---- Fallback Parsing Prompt ----
FALLBACK_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a JSON parser assistant. Your task is to extract structured data from text and return valid JSON.
The output must match the exact schema provided. If the input contains scoring information, extract it.
If values are missing, use reasonable defaults: alignment=0, no_watermark=0, has_text=0, style=0, total=0."""),
    
    ("human", """Parse the following text and extract image ranking scores.

Expected JSON structure:
{{
    "scores": [
        {{
            "alignment": <0-5>,
            "no_watermark": <0 or 2>,
            "has_text": <0 or 1>,
            "style": <0-2>,
            "total": <sum of above>
        }}
    ]
}}

Number of images to extract scores for: {num_images}

Text to parse:
{raw_text}

Return ONLY valid JSON matching the structure above, nothing else.""")
])
