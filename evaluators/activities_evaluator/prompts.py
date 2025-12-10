"""Prompts for activities evaluation with rubric scoring."""

from langchain_core.prompts import ChatPromptTemplate

ACTIVITIES_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert educational content evaluator. Evaluate the section activities for quality and relevance.

Use a 1-5 scale:
1 = Poor: Activities are confusing, irrelevant, or poorly designed
2 = Fair: Activities have issues but are somewhat usable
3 = Good: Activities are functional and relevant
4 = Very Good: Activities are well-designed and engaging
5 = Excellent: Activities are exceptional, pedagogically sound, and highly engaging

Consider:
- Relevance to section content
- Clarity of questions/instructions
- Appropriateness of difficulty
- Variety of activity types"""),
    ("human", """Evaluate these section activities:

Section: {section_title}
Section Content Summary: {theory_summary}

Activities:
{activities_text}

Glossary Terms: {glossary_terms}
Key Concept: {key_concept}

{format_instructions}""")
])

CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Fix the JSON output to match the required schema for activities evaluation."),
    ("human", """The previous output had an error: {error}

Original output:
{completion}

Required format:
{format_instructions}

Provide corrected JSON with quality score (score 1-5 and reasoning).""")
])

