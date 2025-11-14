from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# Prompt template for section theory generation
SECTION_THEORY_TEMPLATE = """
You are an expert course content writer. Generate educational theory content for a specific section.

Course Context:
- Course Title: {course_title}
- Module: {module_title}  
- Submodule: {submodule_title}
- Section: {section_title}
- Target Language: {language}

Requirements:
- Write approximately {n_words} words of educational content
- Create engaging, informative theory that fits the section topic
- Use clear, educational language appropriate for learning
- Structure the content with proper paragraphs and flow
- Make it practical and actionable where appropriate

CRITICAL LANGUAGE INSTRUCTION:
Write the ENTIRE content exclusively in {language}. Every single word, phrase, sentence, and paragraph must be in {language}. Do not mix languages or include any English text unless specifically requested.

Generate comprehensive theory content for this section:
"""

section_theory_prompt = PromptTemplate.from_template(SECTION_THEORY_TEMPLATE)


# ============================================================================
# REFLECTION PATTERN PROMPTS
# ============================================================================

# Prompt for generating verification queries from section content
query_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert fact-checker for educational content. Your task is to generate exactly {k} search queries to verify the factual accuracy and currency of the course section content.

Your queries should investigate:
- Accuracy of formulas, equations, and mathematical expressions
- Validity of cited laws, regulations, standards, or policies
- Correctness of theorems, principles, and scientific concepts
- Accuracy of dates, historical facts, and events
- Current best practices and methodologies
- Statistical data, figures, and numerical information
- Technical specifications and definitions
- Whether information is up to date or has been superseded

Generate queries that will reveal:
✓ If formulas and equations are correctly stated
✓ If laws and regulations are current and accurately referenced
✓ If dates and historical facts are correct
✓ If technical concepts are properly explained
✓ If there are newer standards or updated information
✓ If numerical data and statistics are accurate

Focus on verifiable facts, not subjective opinions or writing style."""),
    ("user", """Course Section Context:
Title: {section_title}
Module: {module_title}
Submodule: {submodule_title}

Section Content:
{theory}

Generate exactly {k} targeted search queries to verify the factual accuracy of the content above.""")
])


# Prompt for reflecting on content quality based on search results
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert educational content reviewer. Analyze the course section content and identify factual errors, outdated information, or improvements needed.

Focus on:
- Factual accuracy (formulas, laws, dates, definitions)
- Currency of information (is it up to date?)
- Technical correctness
- Missing important context or caveats
- Misleading or imprecise statements

Be specific in your critique. Point out:
✓ Exact errors found (e.g., "Formula X should be Y")
✓ Outdated information (e.g., "Standard Z was updated in 2024")
✓ Missing important details
✓ Corrections needed based on search results

Be constructive but thorough. If the content is accurate, acknowledge it."""),
    ("user", """Course Section Context:
Title: {section_title}
Module: {module_title}
Submodule: {submodule_title}

Current Section Content:
{theory}

Verification Search Results:
{search_results}

Provide a detailed critique identifying any factual errors, outdated information, or needed improvements based on the search results.""")
])


# Prompt for regenerating improved content based on reflection
regeneration_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert course content writer. Your task is to improve the section content by incorporating the feedback and corrections identified in the reflection.

Requirements:
- Maintain the original structure and flow
- Keep approximately {n_words} words
- Fix all factual errors identified
- Update outdated information
- Add missing important context
- Ensure all formulas, laws, dates, and facts are accurate
- Keep the engaging, educational tone
- Preserve what was already good

CRITICAL LANGUAGE INSTRUCTION:
Write the ENTIRE improved content exclusively in {language}. Every single word must be in {language}."""),
    ("user", """Course Section Context:
Title: {section_title}
Module: {module_title}
Submodule: {submodule_title}

Original Content:
{theory}

Reflection and Corrections Needed:
{reflection}

Search Results for Reference:
{search_results}

Generate the improved section content incorporating all necessary corrections and improvements:""")
])