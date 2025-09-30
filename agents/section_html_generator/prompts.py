from langchain.prompts import PromptTemplate

# Prompt template for converting theory to HTML
SECTION_HTML_TEMPLATE = """
You are an expert HTML developer. Convert the following educational theory content into well-structured, valid HTML.

Section Context:
- Course Title: {course_title}
- Module: {module_title}  
- Submodule: {submodule_title}
- Section: {section_title}

Theory Content:
{theory}

Requirements:
- Convert the theory into semantic HTML with proper structure
- Use appropriate HTML tags (h2, h3, p, ul, ol, li, strong, em, code, pre, blockquote, etc.)
- Ensure all tags are properly closed and nested
- Do NOT include <!DOCTYPE>, <html>, <head>, or <body> tags
- Only return the content HTML that goes inside a section element
- Preserve all the original text content and language exactly
- Use proper heading hierarchy starting with h2 for the main section title

Generate the HTML content:
"""

section_html_prompt = PromptTemplate.from_template(SECTION_HTML_TEMPLATE)
