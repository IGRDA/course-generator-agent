"""Prompts for index evaluation with rubric scoring."""

from langchain_core.prompts import ChatPromptTemplate

INDEX_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert course evaluator. Evaluate the course index/structure based on the following criteria.

Use a 1-5 scale for each criterion:
1 = Poor: Major issues, fails to meet basic requirements
2 = Fair: Some issues, partially meets requirements
3 = Good: Adequate, meets basic requirements
4 = Very Good: Above average, exceeds basic requirements
5 = Excellent: Outstanding, exemplary quality

Evaluate objectively and provide brief reasoning for each score."""),
    ("human", """Evaluate this course index:

Course Title: {course_title}
Total Modules: {num_modules}
Total Submodules: {num_submodules}
Total Sections: {num_sections}

Course Structure:
{course_structure}

Evaluate on these criteria:

1. **Coverage**: Does the index comprehensively cover the subject matter? Are there gaps or missing essential topics?

2. **Structure**: Is the organization logical? Do modules flow from basic to advanced? Is the hierarchy clear?

3. **Balance**: Are sections roughly equal in scope? Are modules balanced in size?

{format_instructions}""")
])

CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Fix the JSON output to match the required schema for course index evaluation."),
    ("human", """The previous output had an error: {error}

Original output:
{completion}

Required format:
{format_instructions}

Provide corrected JSON with coverage, structure, and balance scores (each with score 1-5 and reasoning).""")
])

