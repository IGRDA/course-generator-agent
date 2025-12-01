"""Prompts for HTML structure evaluation with rubric scoring."""

from langchain_core.prompts import ChatPromptTemplate

HTML_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert educational content evaluator. Evaluate the HTML structure formatting quality.

Use a 1-5 scale:
1 = Poor: HTML structure is broken, missing elements, or unusable
2 = Fair: Basic structure present but with significant issues
3 = Good: Functional HTML structure with minor issues
4 = Very Good: Well-structured HTML with good variety of elements
5 = Excellent: Exemplary HTML structure, optimal use of elements for content

Consider:
- Appropriate use of element types (p, ul, quote, table, paragraphs)
- Content organization within elements
- Logical flow and structure
- Element variety and appropriateness"""),
    ("human", """Evaluate this HTML structure:

Section: {section_title}

HTML Structure:
{html_structure}

Element Summary:
- Total elements: {total_elements}
- Element types used: {element_types}

{format_instructions}""")
])

CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Fix the JSON output to match the required schema for HTML evaluation."),
    ("human", """The previous output had an error: {error}

Original output:
{completion}

Required format:
{format_instructions}

Provide corrected JSON with formatting score (score 1-5 and reasoning).""")
])

