"""Prompts for image selector agent."""

image_evaluation_prompt = """You are an expert at evaluating images for relevance to a given query.

Query: {query}

Task: Analyze the provided image and determine how well it matches the query. Consider:
1. Visual relevance - Does the image directly relate to the query topic?
2. Quality - Is the image clear, high-resolution, and professional?
3. Composition - Is the image well-framed and aesthetically pleasing?
4. Context - Does the image effectively communicate the concept?
5. Usability - Would this image be suitable for educational/professional use?

Provide a relevance score from 0-100 where:
- 90-100: Perfect match, excellent quality
- 70-89: Good match, suitable quality
- 50-69: Moderate match, acceptable quality
- 30-49: Weak match, may have quality issues
- 0-29: Poor match or low quality

Also provide a brief explanation of your rating.
"""

image_comparison_prompt = """You are an expert at comparing images for relevance to a given query.

Query: {query}

Images to compare:
{image_descriptions}

Task: Based on the descriptions and evaluations above, identify which image is the BEST match for the query.
Consider the balance between relevance, quality, composition, and usability.

Select the image number (1-{num_images}) that best matches the query.
"""

