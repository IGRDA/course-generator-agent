from langchain.prompts import PromptTemplate

# ============================================================================
# ROLE & EXPERTISE DEFINITION
# ============================================================================

CURRICULUM_ARCHITECT_ROLE = """You are an expert curriculum architect and instructional designer with deep expertise in:
- Learning progression design and scaffolded instruction
- Cognitive load management and optimal content chunking
- Backward design principles (starting from learning outcomes)

Your course structures should enable effective learning by ensuring:
- Concepts build logically upon each other
- Prerequisites are taught before dependent topics
- Complexity increases gradually and appropriately
- Each section has a clear, focused learning objective"""

# ============================================================================
# PEDAGOGICAL SCAFFOLDING PRINCIPLES
# ============================================================================

SCAFFOLDING_PRINCIPLES = """
=====================================
PEDAGOGICAL SCAFFOLDING REQUIREMENTS
=====================================
Your course structure MUST follow these learning design principles:

LOGICAL PROGRESSION:
- Earlier modules/sections should establish foundations for later content
- Complex topics should be broken into digestible components
- Each section should have ONE clear focus (not multiple competing ideas)
- Avoid circular dependencies (don't require knowledge not yet taught)

LEARNING OUTCOMES:
- Module titles should reflect what learners will ACHIEVE (not just topics)
- Section descriptions should hint at the specific skill or knowledge gained
- Structure should support measurable learning objectives

COGNITIVE LOAD MANAGEMENT:
- Balance theoretical and practical content across the course
- Avoid clustering all difficult concepts in one module
- Ensure variety in section types (concepts, examples, applications)

ENGAGEMENT ARCHITECTURE:
- Start with motivating, accessible content to build momentum
- Place challenging content after foundational concepts are secure
- End modules with synthesis or application to reinforce learning
"""

# ============================================================================
# VERIFICATION MANDATE
# ============================================================================

STRUCTURE_VERIFICATION = """
=====================================
FINAL VERIFICATION (MANDATORY)
=====================================
Before outputting, verify that your structure:
✓ Has exactly the required number of modules, submodules, and sections
✓ Follows logical learning progression (foundations before advanced topics)
✓ Uses clear, descriptive titles
✓ Has descriptions that clarify the learning objective of each element
✓ Maintains the EXACT course title as provided (no translation or modification)
✓ All titles and descriptions are in the target language
✓ Conforms exactly to the JSON schema with no extra fields
"""

# ============================================================================
# MAIN GENERATION TEMPLATE
# ============================================================================

GEN_TEMPLATE = """
{curriculum_role}

{scaffolding_principles}

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Optional description/context: "{course_description}"
- Target language: {language}

=====================================
STRUCTURE CONSTRAINTS (EXACT)
=====================================
- n_modules: {n_modules}
- n_submodules per module: {n_submodules}
- n_sections per submodule: {n_sections}
- Target words per section (for reference only): {n_words}

=====================================
CONTENT RULES
=====================================
1. The COURSE TITLE must remain EXACTLY as provided: "{course_title}" (do not translate or modify it)
2. Every module, submodule, and section must have:
   - A concise, descriptive title (clear learning focus)
   - A brief description (1-2 sentences explaining the learning objective)
3. All titles and descriptions must be written in {language}
4. Only include structural fields: title, index, description
5. Ensure all lists have the EXACT specified lengths
6. Use only the fields defined in the schema (extra fields are forbidden)

=====================================
QUALITY CHECKLIST FOR TITLES
=====================================
Good titles:
✓ Clearly indicate what will be learned
✓ Use accessible language
✓ Are specific and focused
✓ Build logically from previous sections

Avoid:
✗ Vague titles like "Introduction" or "Advanced Topics"
✗ Overly technical jargon without context
✗ Titles that don't indicate learning outcomes
✗ Repetitive or overlapping titles

=====================================
OUTPUT FORMAT
=====================================
Return ONLY valid JSON that conforms to the schema below.
- No code fences, no extra commentary
- The top-level object contains the course structure skeleton

IMPORTANT REMINDERS:
- Keep the course title EXACTLY as provided: "{course_title}"
- Write module titles, submodule titles, section titles, and descriptions in {language}
- Ensure logical learning progression throughout

{structure_verification}

{format_instructions}
"""

gen_prompt = PromptTemplate(
    template=GEN_TEMPLATE,
    input_variables=[
        "course_title", "course_description", "language",
        "n_modules", "n_submodules", "n_sections", "n_words",
        "format_instructions"
    ],
    partial_variables={
        "curriculum_role": CURRICULUM_ARCHITECT_ROLE,
        "scaffolding_principles": SCAFFOLDING_PRINCIPLES,
        "structure_verification": STRUCTURE_VERIFICATION
    }
)


# ============================================================================
# RETRY TEMPLATE
# ============================================================================

retry_template = """Your previous response doesn't follow the required schema and fails parsing.
Fix ONLY the formatting to comply with the schema; do not change the meaning or counts.

=====================================
EXPECTED SCHEMA
=====================================
{schema}

=====================================
CONSTRAINTS
=====================================
- Return pure JSON (no code fences, no extra text)
- Use exactly the specified counts and field names
- No extra fields are allowed
- Only include structural fields: title, index, description
- All titles and descriptions must be written in {language}
- Ensure logical learning progression is maintained
- Course title must remain exactly as originally provided
"""

retry_prompt = PromptTemplate.from_template(retry_template)
