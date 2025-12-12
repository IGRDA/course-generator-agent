from langchain_core.prompts import ChatPromptTemplate

# ---- Image Query Generation Prompt ----
image_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at creating concise image search queries.
Generate a 5-10 word description that captures the visual concept.
Focus on concrete, visual elements that would make a good educational illustration.
Always keep the query relevant to the course topic and the block title and content preview."""),
    
    ("human", """Generate an image search query (3-5 words) for this content block:

Course Topic: {course_title}
Block Title: {block_title}
Content Preview: {content_preview}

The image must visually relate to "{course_title}". Return ONLY the search query, nothing else.""")
])

