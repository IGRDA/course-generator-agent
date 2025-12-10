"""Prompts for section theory evaluation with rubric scoring."""

from langchain_core.prompts import ChatPromptTemplate

SECTION_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert educational content evaluator. Evaluate the section content for factual accuracy.

Use a 1-5 scale:
1 = Poor: Contains significant factual errors or misleading information
2 = Fair: Some inaccuracies or questionable claims
3 = Good: Generally accurate with minor issues
4 = Very Good: Accurate and well-researched
5 = Excellent: Highly accurate, well-sourced, authoritative

Focus ONLY on factual accuracy - not style, length, or formatting."""),
    ("human", """Evaluate this section content for factual accuracy:

Course: {course_title}
Module: {module_title}
Submodule: {submodule_title}
Section: {section_title}

Content:
{theory}

{format_instructions}""")
])

CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Fix the JSON output to match the required schema for section evaluation."),
    ("human", """The previous output had an error: {error}

Original output:
{completion}

Required format:
{format_instructions}

Provide corrected JSON with accuracy score (score 1-5 and reasoning).""")
])

