from langchain.prompts import PromptTemplate

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