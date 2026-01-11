"""Prompt templates for bibliography generation."""

from langchain_core.prompts import ChatPromptTemplate

# Prompt for LLM to suggest relevant books for a module
book_suggestion_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a librarian and academic advisor specializing in creating reading lists for educational courses.

Your task is to suggest relevant, real books for a course module. The books should:
1. Be real, published books (not fictional or made-up titles)
2. Be directly relevant to the module's topics
3. Include a mix of foundational texts and modern references
4. Be appropriate for the course language and audience level

IMPORTANT: Only suggest books you are confident actually exist. Include as much detail as you know:
- Full book title (including subtitle if applicable)
- Author name(s) - use full names when known
- Approximate publication year if known
- Publisher if known

Do NOT invent books or authors. If unsure about a book's existence, do not include it."""),
    
    ("human", """Generate a list of {num_books} recommended books for the following course module:

**Course Title:** {course_title}
**Module Title:** {module_title}
**Module Description:** {module_description}

**Topics covered in this module:**
{module_topics}

**Language:** {language}

{exclusion_note}

Return your response as a JSON array of book objects. Each book should have:
- "title": Full book title
- "authors": Array of author names (full names preferred)
- "year": Publication year (number or null if unknown)
- "publisher": Publisher name (or null if unknown)
- "relevance": Brief explanation of why this book is relevant (1 sentence)

Example format:
```json
[
  {{
    "title": "Introduction to Algorithms",
    "authors": ["Thomas H. Cormen", "Charles E. Leiserson", "Ronald L. Rivest", "Clifford Stein"],
    "year": 2009,
    "publisher": "MIT Press",
    "relevance": "Comprehensive algorithms textbook covering fundamental data structures and algorithmic paradigms."
  }}
]
```

Return ONLY the JSON array, no additional text.""")
])


# Prompt for formatting a single book citation in APA 7
apa_formatting_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in APA 7th edition citation formatting.

Format book citations according to APA 7 guidelines:

**Basic format:**
Author, A. A., & Author, B. B. (Year). Title of work: Capital letter also for subtitle (Edition ed.). Publisher. DOI or URL

**Rules:**
1. Authors: Last name, First initial. Middle initial. Use & before last author.
2. Year: In parentheses. Use (n.d.) if no date.
3. Title: Italicized. Only first word and proper nouns capitalized.
4. Edition: Include only if not first edition, e.g., (2nd ed.)
5. Publisher: Do not include location.
6. DOI/URL: Include if available, no period after URL.

**Examples:**
- Single author: Jackson, L. M. (2019). The psychology of prejudice: From attitudes to social action (2nd ed.). American Psychological Association.
- Multiple authors: Sapolsky, R. M., & Balt, S. (2020). Behave: The biology of humans at our best and worst. Penguin Books.
- With DOI: Brown, L. S. (2018). Feminist therapy (2nd ed.). American Psychological Association. https://doi.org/10.1037/0000092-000"""),
    
    ("human", """Format this book in APA 7 style:

Title: {title}
Authors: {authors}
Year: {year}
Publisher: {publisher}
Edition: {edition}
DOI: {doi}
URL: {url}

Return ONLY the formatted citation, nothing else.""")
])
