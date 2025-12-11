from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ---- Main Activities Generation Prompt ----
activities_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert pedagogy specialist and educational content designer.
Your task is to extract key educational elements from theory text and create engaging interactive activities.

You must generate activities that:
- Are aligned with the theory content
- Test understanding at different levels (recall, comprehension, application)
- Are appropriate for the language: {language}
- Follow exact format specifications for each activity type

Be creative and educational, but follow the required formats precisely."""),
    
    ("human", """Analyze this theory text and generate educational activities:

THEORY TEXT:
{theory}

SECTION TITLE: {section_title}

Generate the following:

1. GLOSSARY (1-4 terms):
   - Extract 1 to 4 key terms that are essential for understanding this content
   - Each term must have a clear, concise explanation

2. KEY CONCEPT (one sentence):
   - Provide a single sentence that captures the main idea of this section

3. INTERESTING FACT:
   - Provide an interesting, relevant fact related to the section content
   - Should be engaging and enhance understanding

4. QUOTE:
   - Provide a relevant quote from an expert or notable figure
   - Format: {{"author": "Name", "text": "Quote text"}}

5. ACTIVITIES:
   You must generate activities of these types: {activity_types}
   
   For each activity type:
   
   - **order_list**: Create a question where students must order items in correct sequence
     Format: {{"question": "...", "solution": ["item1", "item2", "item3", ...]}}
     Requirements: At least 2 items, items should be distinct
   
   - **fill_gaps**: Create a text with 2-4 blanks marked with *blanquito*
     Format: {{"question": "Text with *blanquito* placeholders", "solution": ["word1", "word2", ...]}}
     Requirements: Number of *blanquito* must equal number of solutions, at least 2 blanks
   
   - **swipper**: Create statements students categorize as true or false
     Format: {{"question": "Context or instruction", "solution": {{"true": ["stmt1", ...], "false": ["stmt2", ...]}}}}
     Requirements: At least one statement in true or false
   
   - **linking_terms**: Create concept-definition pairs to match
     Format: {{"question": "Match the concepts", "solution": [{{"concept": "...", "related": "..."}}, ...]}}
     Requirements: At least 2 pairs
   
   - **multiple_choice**: Create a question with one correct answer and 3+ incorrect options
     Format: {{"question": "...", "solution": "correct answer", "other_options": ["wrong1", "wrong2", "wrong3"]}}
     Requirements: Solution must NOT appear in other_options, at least 3 other options, no duplicates
   
   - **multi_selection**: Create a question with multiple correct answers
     Format: {{"question": "...", "solution": ["correct1", "correct2"], "other_options": ["wrong1", "wrong2"]}}
     Requirements: No overlap between solution and other_options, at least 4 total options

6. FINAL ACTIVITIES:
   Generate activities of these types: {final_activity_types}
   
   - **group_activity**: Collaborative task for teams
   - **discussion_forum**: Thought-provoking discussion question
   - **individual_project**: Individual project applying knowledge
   - **open_ended_quiz**: Open-ended question requiring detailed response
   
   Format: {{"question": "Task or question description"}}

{format_instructions}

IMPORTANT:
- Ensure all activities are based on the theory content
- Follow the exact formats specified
- Make sure all required fields are present
- Activities should be engaging and educational""")
])


# ---- Correction Prompt ----
correction_prompt = PromptTemplate.from_template("""The previous output had validation errors.

ERROR:
{error}

ORIGINAL OUTPUT:
{completion}

Please fix the errors and return valid JSON following these requirements:
{format_instructions}

Make sure to:
1. Include all required fields
2. Follow the exact format specifications
3. Ensure data types are correct
4. Fix any validation issues mentioned in the error
""")

