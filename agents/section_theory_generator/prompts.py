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

Style Guidelines:
{style_guidelines}

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
# STYLE STRATEGIES
# ============================================================================

STYLE_COURSE_INTRO = """
- This is the very first section of the entire course.
- Welcome the student warmly and establish the course's value.
- Set expectations and motivate the learning journey.
- Maintain an encouraging, professional, and inviting tone.
"""

STYLE_MODULE_START = """
- This is the first section of a new module.
- Briefly introduce the module's theme and how it builds on previous content.
- Establish the relevance of this module to the overall course.
- Use a transitional tone that connects to what came before.
- Keep it concise - no lengthy welcomes.
"""

STYLE_SUBMODULE_START = """
- This is the first section of a new submodule.
- Provide a brief contextual bridge from the previous submodule.
- Introduce the specific focus of this submodule.
- Use a focused, professional tone.
- Avoid generic phrases like "In this section..." or "We will now...".
"""

STYLE_CONTINUATION = """
- This is a continuation section within the same submodule.
- DO NOT write any introduction or transition.
- DIVE STRAIGHT into the content. Start immediately with the core concept, definition, or explanation.
- Maintain a direct, information-dense style.
- Assume the student is already engaged and in learning mode.
- Avoid all meta-commentary about structure or what will be covered.
"""

STYLE_DEEP_DIVE = """
- This section goes deeper into a topic already introduced.
- Start with the assumption that foundational concepts are understood.
- Use a more technical and detailed tone.
- Present advanced aspects, edge cases, or nuanced perspectives.
- Maintain rigor without repeating basics.
"""


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