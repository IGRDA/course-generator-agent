"""Prompts for video generator agent."""

from langchain_core.prompts import ChatPromptTemplate


# ---- Video Query Generation Prompt ----
video_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at creating YouTube search queries for educational content.
Generate a concise search query (4-8 words) that will find relevant educational videos.
Focus on finding tutorials, explanations, or educational content that matches the module topic.
Consider the target language when crafting the query - use the appropriate language for better results."""),
    
    ("human", """Generate a YouTube search query for educational videos about this module:

Course Title: {course_title}
Module Title: {module_title}
Module Description: {module_description}
Key Topics: {key_topics}
Target Language: {language}

Generate a search query in {language} that will find educational videos explaining this module's content.
Return ONLY the search query, nothing else.""")
])

