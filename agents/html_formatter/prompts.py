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

INTERACTIVE FORMAT TYPES (Choose the most appropriate based on content):
- "paragraphs": General educational content, standard explanations, default choice for most topics
- "accordion": Large amounts of information that benefit from collapsible sections, FAQs, detailed breakdowns
- "tabs": Content with distinct categories or topics that can be clearly separated (e.g., theory vs practice, different methods)
- "carousel": Sequential information, step-by-step processes, progressive learning concepts
- "flip": Key concepts with front/back information (terms and definitions, questions and answers, concept pairs)
- "timeline": Chronological events, historical progressions, evolution of ideas or technologies
- "conversation": Dialogues, Q&A format, contrasting viewpoints, debates between perspectives

EXAMPLE STRUCTURE (using accordion format):
{{
  "elements": [
    {{
      "type": "p",
      "content": "Introduction paragraph explaining the main topic..."
    }},
    {{
      "type": "INTERACTIVE FORMAT",
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
            {{"type": "ul", "content": ["Detail A", "Detail B"]}}
          ]
        }},
        {{
          "title": "Third Key Concept",
          "icon": "mdi-brain",
          "elements": [
            {{"type": "p", "content": "Explanation of the third concept..."}}
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

Note: Replace "INTERACTIVE FORMAT" with the format type you determine is most appropriate for the content (paragraphs, accordion, tabs, carousel, flip, timeline, conversation).
Choose the best mdi icon for the content.

{format_instructions}

IMPORTANT:
- Return a direct array wrapped in an "elements" field
- Intro and conclusion MUST be simple "p" elements at the start and end
- The main content MUST use one of the interactive format types
- YOU MUST CHOOSE the format type that best fits the content characteristics (analyze the theory text and select the most appropriate format)
- Consider: Is the content chronological? Use timeline. Is it Q&A? Use conversation. Are there distinct categories? Use tabs.
- Choose icons that match the content theme
- Each format should have minimum 4 blocks to provide substantial content
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
