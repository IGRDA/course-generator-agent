from langchain.prompts import PromptTemplate

GEN_TEMPLATE = """
You are designing a complete course structure (skeleton only).

Follow these steps carefully, and output ONLY valid JSON that satisfies the schema and counts:

1) Understand the course:
   - Course title: "{course_title}"
   - Optional description/context: "{course_description}"
   - Target language: "{language}"

2) Structure constraints (must be exact):
   - n_modules: {n_modules}
   - n_submodules per module: {n_submodules}
   - n_sections per submodule: {n_sections}
   - Target words per section (for reference only): {n_words}

3) Content rules:
   - The COURSE TITLE must remain EXACTLY as provided: "{course_title}" (do not translate or modify it)
   - Every module, submodule, and section must have a concise, descriptive title.
   - Each module, submodule, and section should have a brief description (1-2 sentences).
   - Module titles, submodule titles, section titles, and descriptions must be written in {language}.
   - Only include structural fields: title, index, description.
   - Ensure all lists have the exact specified lengths.
   - Use only the fields defined in the schema (extra fields are forbidden).

4) Output format:
   - Return ONLY valid JSON that conforms to the schema below (no code fences, no extra commentary).
   - The top-level object contains the course structure skeleton.

IMPORTANT: 
- Keep the course title EXACTLY as provided: "{course_title}"
- Write module titles, submodule titles, section titles, and descriptions in {language}.

{format_instructions}
"""

gen_prompt = PromptTemplate.from_template(GEN_TEMPLATE)


retry_template = (
    "Your previous response doesn't follow the required schema and fails parsing. "
    "Fix ONLY the formatting to comply with the schema; do not change the meaning or counts.\n\n"
    "EXPECTED SCHEMA:\n{schema}\n"
    "Constraints:\n"
    "- Return pure JSON (no code fences, no extra text).\n"
    "- Use exactly the specified counts and field names.\n"
    "- No extra fields are allowed.\n"
    "- Only include structural fields: title, index, description.\n"
    "- All titles and descriptions must be written in {language}.\n"
)
retry_prompt = PromptTemplate.from_template(retry_template)