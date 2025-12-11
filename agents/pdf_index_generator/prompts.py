from langchain.prompts import PromptTemplate

SYLLABUS_ANALYSIS_TEMPLATE = """
You are analyzing a course syllabus extracted from a PDF document to create a structured course outline.

**IMPORTANT**: Your primary goal is to follow the existing structure and content from the syllabus as closely as possible. Extract the course title (without translation), description, and hierarchical structure (modules, submodules, sections) directly from the table of contents, index, or curriculum outline in the syllabus.

Follow these steps carefully, and output ONLY valid JSON that satisfies the schema and counts:

1) Analyze the syllabus content below:

SYLLABUS CONTENT:
---
{syllabus_markdown}
---

2) Extract information:
   - Course title: Extract from the syllabus header or title section (keep it in its ORIGINAL language, do NOT translate)
   - Course description: Extract from course overview, description, or objectives section
   - Target language: {language}
   - Table of contents/Index: Identify the course structure (modules, topics, units, sections, etc.)

3) Structure constraints (you must adapt the syllabus structure to fit these exact counts):
   - n_modules: {n_modules}
   - n_submodules per module: {n_submodules}
   - n_sections per submodule: {n_sections}
   - n_words (target words per section, excluding 'theory' which must be empty): {n_words}

4) Mapping strategy:
   - Map the syllabus's main units/modules to the n_modules structure
   - Map sub-topics/chapters to submodules
   - Map individual lessons/topics/sections to sections
   - If the syllabus has more detail than the structure allows, group related topics together
   - If the syllabus has less detail than required, break down topics into logical subtopics
   - PRESERVE the original topic names and order from the syllabus as much as possible

5) Content rules:
   - Every module, submodule, and section must have a concise, descriptive title EXTRACTED OR DERIVED from the syllabus
   - ALL TITLES must be written in {language}
   - theory: DO NOT generate any content. Set every 'theory' field to the empty string "" exactly
   - Ensure all lists have the exact specified lengths
   - Use only the fields defined in the schema (extra fields are forbidden)
   - Follow the hierarchical structure and topic order from the syllabus

6) Output format:
   - Return ONLY valid JSON that conforms to the schema below (no code fences, no extra commentary)
   - The top-level object contains: "title" (string) and "modules" (array)

IMPORTANT: 
- Follow the syllabus structure, table of contents, and topic organization as closely as possible
- Preserve topic names, terminology, and sequence from the original document
- Extract the course title in its ORIGINAL language without translating it
- Module titles, submodule titles, and section titles must be in {language}

{format_instructions}
"""

syllabus_analysis_prompt = PromptTemplate.from_template(SYLLABUS_ANALYSIS_TEMPLATE)


RETRY_TEMPLATE = (
    "Your previous response doesn't follow the required schema and fails parsing. "
    "Fix ONLY the formatting to comply with the schema; do not change the meaning or counts.\n\n"
    "EXPECTED SCHEMA:\n{schema}\n"
    "Constraints:\n"
    "- Return pure JSON (no code fences, no extra text).\n"
    "- Use exactly the specified counts and field names.\n"
    "- No extra fields are allowed.\n"
    "- Every 'theory' field must be the empty string \"\" (no content, no whitespace).\n"
    "- All titles must be written in {language}.\n"
)
retry_prompt = PromptTemplate.from_template(RETRY_TEMPLATE)

