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
    
    ("human", """Convert this theory text into a structured HTML JSON format as a direct array:

THEORY TEXT:
{theory}

SECTION TITLE: {section_title}

REQUIREMENTS:

1. INTRO (required):
   - First element in the array
   - Type: "p" (paragraph)
   - Content: A compelling introduction paragraph (2-3 sentences)

2. MAIN CONTENT (required):
   - Use one of the interactive format types for the main body content
   - Available formats: "paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"
   - Content is a list of blocks, each representing a key aspect
   - Minimum 3 blocks
   
   For each block in the interactive format:
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
   - Last element in the array
   - Type: "p" (paragraph)
   - Content: A summary or closing thought (2-3 sentences)

INTERACTIVE FORMAT TYPES:
- "paragraphs": Standard sections with title, icon, and elements
- "accordion": Collapsible expandable sections
- "tabs": Tab-based navigation between sections
- "carousel": Slideshow/carousel format
- "flip": Flip card animation effects
- "timeline": Chronological timeline display
- "conversation": Dialog/chat-style presentation

EXAMPLE STRUCTURE:
{{
  "elements": [
    {{
      "type": "p",
      "content": "Introduction paragraph explaining the main topic..."
    }},
    {{
      "type": "accordion",
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
- Return a direct array wrapped in an "elements" field
- Intro and conclusion MUST be simple "p" elements at the start and end
- The main content MUST use one of the interactive format types
- Choose the format type that best fits the content (accordion for expandable, tabs for categorized, timeline for chronological, etc.)
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
1. Root object must have "elements" array (not "theory")
2. Interactive formats (paragraphs, accordion, tabs, carousel, flip, timeline, conversation) content must be a list of blocks with title, icon, and elements
3. Element types must be "p", "ul", "quote", "table", "paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", or "conversation"
4. Quotes need both "author" and "quote" fields
5. Tables need "title", "headers" (array), and "rows" (array of arrays)
""")
