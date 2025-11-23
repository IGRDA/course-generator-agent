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

REQUIREMENTS:

1. INTRO (required):
   - First element in the theory list
   - Type: "p" (paragraph)
   - Content: A compelling introduction paragraph (2-3 sentences)

2. MAIN CONTENT (required):
   - Use "paragraphs" type for the main body content
   - Content is a list of blocks, each representing a key aspect
   - Minimum 3 blocks
   
   For each block in "paragraphs":
   - Title: Short, descriptive title (3-6 words)
   - Icon: Material Design Icon name (e.g., "mdi-book", "mdi-lightbulb", "mdi-chart-line")
     Suggested icon based on section: {suggested_icon}
   - Elements: Minimum 2 elements per block
   
   Element types inside blocks:
   - "p": Paragraph text (string content)
   - "ul": Bullet list (array of strings)
   {quote_instruction}
   {table_instruction}

3. CONCLUSION (required):
   - Last element in the theory list
   - Type: "p" (paragraph)
   - Content: A summary or closing thought (2-3 sentences)

EXAMPLE STRUCTURE:
{{
  "theory": [
    {{
      "type": "p",
      "content": "Introduction paragraph explaining the main topic..."
    }},
    {{
      "type": "paragraphs",
      "content": [
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
        }}
      ]
    }},
    {{
      "type": "p",
      "content": "Summary paragraph tying together the key concepts..."
    }}
  ]
}}

{format_instructions}

IMPORTANT:
- The root object must contain a "theory" list
- Intro and conclusion MUST be simple "p" elements at the start and end of the list
- The main content MUST be wrapped in a "paragraphs" element
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
1. Root object must have "theory" list
2. "paragraphs" content must be a list of blocks with title, icon, and elements
3. Element types must be "p", "ul", "quote", "table", or "paragraphs"
4. Quotes need both "author" and "quote" fields
5. Tables need "title", "headers" (array), and "rows" (array of arrays)
""")
