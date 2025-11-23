from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ---- Main HTML Generation Prompt ----
html_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in web content structuring and educational design.
Your task is to convert plain theory text into a structured JSON format suitable for modern web display.

You must create content that:
- Has a clear introduction and conclusion
- Is organized into logical, themed sections
- Uses appropriate icons for visual representation
- Maintains educational value while being visually organized
- Follows the exact JSON schema provided

Language: {language}"""),
    
    ("human", """Convert this theory text into a structured HTML JSON format:

THEORY TEXT:
{theory}

SECTION TITLE: {section_title}
FORMAT TYPE: {html_format}

REQUIREMENTS:

1. INTRO (required):
   - Type: "p" (paragraph)
   - Content: A compelling introduction paragraph (2-3 sentences)

2. CONTENT (required):
   - Format: Must be "{html_format}" (tabs | accordion | timeline | cards)
   - Items: Minimum 3 items, each representing a key aspect of the theory
   
   For each item:
   - Title: Short, descriptive title (3-6 words)
   - Icon: Material Design Icon name (e.g., "mdi-book", "mdi-lightbulb", "mdi-chart-line")
     Suggested icon based on section: {suggested_icon}
   - Elements: Minimum 2 elements per item
   
   Element types:
   - "p": Paragraph text (string content)
   - "ul": Bullet list (array of strings)
   {quote_instruction}
   {table_instruction}

3. CONCLUSION (required):
   - Type: "p" (paragraph)
   - Content: A summary or closing thought (2-3 sentences)

EXAMPLE STRUCTURE:
{{
  "intro": {{
    "type": "p",
    "content": "Introduction paragraph explaining the main topic..."
  }},
  "content": {{
    "format": "{html_format}",
    "items": [
      {{
        "title": "First Key Concept",
        "icon": "mdi-lightbulb",
        "elements": [
          {{"type": "p", "content": "Explanation of the first concept..."}},
          {{"type": "ul", "content": ["Point 1", "Point 2", "Point 3"]}}
        ]
      }},
      {{
        "title": "Second Key Concept",
        "icon": "mdi-chart-line",
        "elements": [
          {{"type": "p", "content": "Explanation of the second concept..."}},
          {{"type": "p", "content": "Additional detail..."}}
        ]
      }},
      {{
        "title": "Third Key Concept",
        "icon": "mdi-cog",
        "elements": [
          {{"type": "p", "content": "Explanation of the third concept..."}},
          {{"type": "ul", "content": ["Detail 1", "Detail 2"]}}
        ]
      }}
    ]
  }},
  "conclusion": {{
    "type": "p",
    "content": "Summary paragraph tying together the key concepts..."
  }}
}}

{format_instructions}

IMPORTANT:
- All formats (tabs, accordion, timeline, cards) use the SAME structure
- Only the "format" field value changes
- Ensure minimum 3 items in content
- Each item must have at least 2 elements
- Intro and conclusion must be type "p"
- Choose icons that match the content theme
""")
])


# ---- Correction Prompt ----
correction_prompt = PromptTemplate.from_template("""The previous output had validation errors.

ERROR:
{error}

ORIGINAL OUTPUT:
{completion}

Please fix the errors and return valid JSON following these requirements:
{format_instructions}

Common issues to check:
1. Intro and conclusion must have type "p" (not "ul", "quote", or "table")
2. Content format must match the specified format exactly
3. Must have at least 3 items in content
4. Each item must have at least 2 elements
5. Element types must be "p", "ul", "quote", or "table"
6. Quotes need both "author" and "quote" fields
7. Tables need "title", "headers" (array), and "rows" (array of arrays)
""")

