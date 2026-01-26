"""Prompts for video generator agent."""

from langchain_core.prompts import ChatPromptTemplate


# ---- Video Query Generation Prompt (Legacy - single query) ----
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


# ---- Multi-Query Video Generation Prompt ----
video_multi_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at creating YouTube search queries for educational content.
Your task is to generate multiple search queries that will find highly relevant educational videos.

You must generate:
1. ONE general query about the overall module theme
2. Multiple specific queries about concrete concepts, terms, or techniques mentioned in the module

IMPORTANT GUIDELINES:
- Each query should be 4-8 words, optimized for YouTube search
- Specific queries must target CONCRETE concepts from the key topics (named techniques, specific terms, particular methods)
- Specific queries should NOT be variations of the general theme - they must focus on distinct, concrete sub-topics
- Use the target language for all queries
- Include educational keywords like "tutorial", "explained", "introduction", or equivalents in the target language

Output your response as valid JSON with this exact structure:
{{
  "general_query": "module theme educational video",
  "specific_queries": ["concrete concept 1 tutorial", "specific term 2 explained"]
}}"""),
    
    ("human", """Generate YouTube search queries for educational videos about this module:

Course Title: {course_title}
Module Title: {module_title}
Module Description: {module_description}
Key Topics:
{key_topics}

Target Language: {language}
Number of specific queries needed: {num_specific_queries}

Generate 1 general query for the module theme AND {num_specific_queries} specific queries for concrete concepts from the key topics.
Each specific query should target a DISTINCT concrete concept, technique, or term - not just rephrase the general theme.

Return ONLY the JSON object, no additional text.""")
])

