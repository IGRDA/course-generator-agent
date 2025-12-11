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
   
   UNIVERSAL BLOCK STRUCTURE (ALL interactive formats use this EXACT structure):
   - Content: List of blocks (minimum 3 blocks)
   - Each block has THREE required fields:
     * title: Short, descriptive title (3-6 words)
     * icon: Material Design Icon name (e.g., "mdi-book", "mdi-lightbulb", "mdi-chart-line")
       Suggested icon based on section: {suggested_icon}
     * elements: List of HTML elements (minimum 2 elements per block)
   
   Element types allowed inside blocks:
   - "p": Paragraph text (string content)
   - "ul": Bullet list (array of strings)
   {quote_instruction}
   {table_instruction}
   
   NOTE: There is NO difference in structure between formats - only in how they're displayed in the UI.
   Do NOT create special structures like "front"/"back" for flip cards - use the universal block structure.

3. CONCLUSION (required):
   - Last element in the array
   - Type: "p" (paragraph)
   - Content: A summary or closing thought (2-3 sentences)

INTERACTIVE FORMAT TYPES (Choose the most appropriate based on content):

IMPORTANT: ALL interactive formats use the SAME structure - a list of blocks where each block has:
- title: Short descriptive title (3-6 words)
- icon: Material Design Icon name (mdi-*)
- elements: List of content elements (p, ul, quote, table)

Choose the format that best fits how users will interact with the content:
- "paragraphs": General educational content, standard explanations, default choice for most topics
- "accordion": Large amounts of information that benefit from collapsible sections, FAQs, detailed breakdowns
- "tabs": Content with distinct categories or topics that can be clearly separated (e.g., theory vs practice, different methods)
- "carousel": Sequential information, step-by-step processes, progressive learning concepts
- "flip": Interactive flip cards with blocks that reveal content on interaction (terms/definitions, questions/answers). Same block structure as other formats
- "timeline": Chronological events, historical progressions, evolution of ideas or technologies
- "conversation": Dialogues, Q&A format, contrasting viewpoints, debates between perspectives

EXAMPLE STRUCTURE (works for ANY interactive format - accordion, tabs, flip, carousel, etc.):
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

CRITICAL NOTES:
- Replace "INTERACTIVE FORMAT" with one of: paragraphs, accordion, tabs, carousel, flip, timeline, conversation
- This EXACT structure applies to ALL formats (including "flip" - no special "front"/"back" fields!)
- The only difference between formats is the UI presentation, NOT the data structure
- Choose the best mdi icon for each block's content

{format_instructions}

IMPORTANT:
- Return a direct array wrapped in an "elements" field
- Intro and conclusion MUST be simple "p" elements at the start and end
- The main content MUST use one of the interactive format types
- YOU MUST CHOOSE the format type that best fits the content characteristics (analyze the theory text and select the most appropriate format)
- Consider: Is the content chronological? Use timeline. Is it Q&A? Use conversation. Are there distinct categories? Use tabs.
- Choose icons that match the content theme
- Each format should have minimum 4 blocks to provide substantial content
- ALL INTERACTIVE FORMATS USE THE SAME STRUCTURE: {{title, icon, elements}} blocks - no exceptions!
- For "flip" cards: Use the SAME block structure, NOT "front"/"back" fields
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
2. ALL interactive formats (paragraphs, accordion, tabs, carousel, flip, timeline, conversation) MUST use the SAME structure:
   - content: List of blocks
   - Each block: {{"title": "...", "icon": "mdi-...", "elements": [...]}}
   - NO special fields like "front"/"back" for flip cards
3. Element types must be "p", "ul", "quote", "table", "paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", or "conversation"
4. Quotes need both "author" and "text" fields
5. Tables need "title", "headers" (array), and "rows" (array of arrays)
6. The "flip" format uses blocks with title/icon/elements, NOT front/back structure
""")
