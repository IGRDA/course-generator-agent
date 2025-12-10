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


INFO_PRESERVATION_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert quality auditor specializing in content transformation.
Your task is to evaluate information preservation when theory is converted to HTML.

Use a 1-5 scale:
1 = Poor: Major concepts lost, HTML misses critical information
2 = Fair: Several important concepts missing or significantly altered
3 = Good: Most content preserved, minor omissions
4 = Very Good: Nearly complete preservation with trivial differences
5 = Excellent: Perfect fidelity, all key concepts fully represented

Evaluation Process:
1. Identify 5-8 key concepts from the original theory
2. Check if each concept is visible in the HTML
3. Weight the importance of any missing concepts
4. Score based on weighted concept preservation"""),
    ("human", """Evaluate content fidelity for this transformation:

Section: {section_title}
Description: {section_description}

ORIGINAL CONTENT (Theory/Source):
---
{theory_content}
---

FINAL CONTENT (HTML/Target):
---
{html_content}
---

Score based on how well the key concepts are preserved. In your reasoning, list any important concepts that were lost or significantly altered.

{format_instructions}""")
])

