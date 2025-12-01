"""Prompts for overall course evaluation with rubric scoring."""

from langchain_core.prompts import ChatPromptTemplate

OVERALL_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert course evaluator. Evaluate the overall course coherence.

Use a 1-5 scale:
1 = Poor: Course lacks coherence, sections feel disconnected
2 = Fair: Some coherence issues, flow is inconsistent
3 = Good: Adequate coherence, sections generally connect
4 = Very Good: Good coherence, logical progression throughout
5 = Excellent: Exceptional coherence, seamless flow and progression

Consider:
- Logical progression from module to module
- Connection between sections within submodules
- Consistent depth and tone throughout
- Clear learning path from beginning to end"""),
    ("human", """Evaluate this course for overall coherence:

Course Title: {course_title}
Language: {language}

Course Overview:
{course_overview}

Sample Section Titles (showing flow):
{section_titles_sample}

{format_instructions}""")
])

CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Fix the JSON output to match the required schema for overall course evaluation."),
    ("human", """The previous output had an error: {error}

Original output:
{completion}

Required format:
{format_instructions}

Provide corrected JSON with coherence score (score 1-5 and reasoning).""")
])

