# Image Selector Agent

An intelligent agentic system that searches for images and selects the best match using vision AI.

## Overview

The Image Selector Agent combines web scraping with vision model evaluation to find and select the most relevant image for a given query. It follows modern agentic design patterns with async operations, retry logic, and structured outputs.

## Features

### 🎯 Core Capabilities
- **Bing Image Search**: Scrapes Bing for images without requiring API keys
- **Vision AI Evaluation**: Uses Mistral's Pixtral-12B to intelligently evaluate images
- **Intelligent Selection**: Considers relevance, quality, composition, and context
- **Parallel Processing**: Evaluates multiple images concurrently for speed

### 🏗️ Agentic Design Patterns
- **Async/Await**: All operations are async for optimal performance
- **Retry Logic**: Exponential backoff with configurable retries
- **Structured Outputs**: Type-safe Pydantic models throughout
- **Concurrency Control**: Semaphore-based parallel execution
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Observability**: Clear progress indicators and logging

## Installation

Ensure you have the required dependencies:

```bash
# Required packages
pip install langchain-mistralai pydantic requests beautifulsoup4
```

Set your Mistral API key:

```bash
export MISTRAL_API_KEY=your_api_key_here
```

## Usage

### Basic Usage (Async)

```python
from agents.image_selector import select_best_image

# Search and select best image
result = await select_best_image(
    query="machine learning pipeline",
    max_results=5,
    concurrency=3
)

if result.selected_image:
    print(f"Best image URL: {result.selected_image.url}")
    print(f"Score: {result.selected_image.evaluation.relevance_score}/100")
    print(f"Reason: {result.selected_image.evaluation.explanation}")
else:
    print(f"Error: {result.error}")
```

### Synchronous Usage

```python
from agents.image_selector.agent import select_best_image_sync

# Runs in non-async context
result = select_best_image_sync(
    query="data engineering architecture",
    max_results=5
)
```

### Command Line Usage

```bash
# Run from agent directory
cd agents/image_selector
python agent.py "kubernetes architecture"
```

### Advanced Usage

```python
# Custom configuration
result = await select_best_image(
    query="neural network visualization",
    max_results=10,        # More candidates
    concurrency=5,         # Higher parallelism
    max_retries=5          # More retries for reliability
)

# Access all evaluated candidates
for candidate in result.all_candidates:
    print(f"URL: {candidate.url}")
    print(f"Score: {candidate.evaluation.relevance_score}/100")
    print(f"Reason: {candidate.evaluation.explanation}")
    print("-" * 80)
```

## API Reference

### Main Functions

#### `select_best_image()`

The primary async function for image selection.

**Parameters:**
- `query` (str): Search query for images
- `max_results` (int, default=5): Maximum number of images to retrieve
- `concurrency` (int, default=3): Number of concurrent evaluations
- `max_retries` (int, default=3): Maximum retry attempts per operation

**Returns:**
- `ImageSelectionResult`: Result object with selected image and metadata

#### `select_best_image_sync()`

Synchronous wrapper for non-async contexts.

**Parameters:** Same as `select_best_image()`

**Returns:** Same as `select_best_image()`

### Data Models

#### `ImageSelectionResult`

Result of the image selection process.

**Fields:**
- `query` (str): Original search query
- `selected_image` (ImageCandidate | None): Best selected image
- `all_candidates` (List[ImageCandidate]): All evaluated images
- `error` (str | None): Error message if selection failed

#### `ImageCandidate`

Represents a candidate image with evaluation.

**Fields:**
- `url` (str): Image URL
- `thumbnail_url` (str): Thumbnail URL
- `description` (str): Description from search
- `author` (str): Image author/source
- `evaluation` (ImageEvaluation | None): Vision model evaluation

#### `ImageEvaluation`

Structured evaluation from vision model.

**Fields:**
- `relevance_score` (int): Score from 0-100
- `explanation` (str): Reasoning for the score

## How It Works

### Agent Workflow

```
1. Query Input
   ↓
2. Bing Image Search
   ↓
3. Parallel Image Evaluation (Vision AI)
   │
   ├─→ Image 1 → Evaluate → Score + Explanation
   ├─→ Image 2 → Evaluate → Score + Explanation
   ├─→ Image 3 → Evaluate → Score + Explanation
   └─→ ...
   ↓
4. Select Best Match (Highest Score)
   ↓
5. Return Result
```

### Evaluation Criteria

The vision model evaluates images based on:

1. **Visual Relevance** - Direct relation to query topic
2. **Quality** - Clarity, resolution, professionalism
3. **Composition** - Framing and aesthetics
4. **Context** - Effective concept communication
5. **Usability** - Suitability for educational/professional use

### Score Ranges

- **90-100**: Perfect match, excellent quality
- **70-89**: Good match, suitable quality
- **50-69**: Moderate match, acceptable quality
- **30-49**: Weak match, may have issues
- **0-29**: Poor match or low quality

## Architecture

### Design Patterns

```
agents/image_selector/
├── __init__.py          # Public API exports
├── agent.py             # Core agent logic
├── prompts.py           # LLM prompts
└── README.md            # Documentation
```

**Key Components:**

1. **Agent Logic** (`agent.py`)
   - Async orchestration
   - Retry mechanisms
   - Parallel evaluation
   - Result aggregation

2. **Prompts** (`prompts.py`)
   - Vision model instructions
   - Evaluation criteria
   - Structured output templates

3. **Type System**
   - Pydantic models for data validation
   - Type hints throughout
   - Structured LLM outputs

### Integration Points

- **Bing Scraper**: `tools.imagesearch.bing_scraper`
- **Vision Model**: `langchain_mistralai.ChatMistralAI` (Pixtral-12B)
- **Async Runtime**: Python asyncio with semaphore control

## Examples

See `notebooks/image_selector_demo.ipynb` for interactive examples.

## Error Handling

The agent handles various failure scenarios:

- **Search Failures**: Returns error in result object
- **Evaluation Failures**: Assigns score of 0 to failed images
- **Network Issues**: Retry with exponential backoff
- **Invalid Images**: Filters out non-image URLs

All errors are captured and returned in the `ImageSelectionResult.error` field.

## Performance

- **Parallel Evaluation**: Multiple images evaluated simultaneously
- **Concurrency Control**: Configurable via `concurrency` parameter
- **Async Operations**: Non-blocking I/O throughout
- **Retry Logic**: Automatic recovery from transient failures

**Typical Performance:**
- 5 images: ~10-15 seconds
- 10 images: ~15-25 seconds
- (varies based on network and API latency)

## Limitations

- Requires Mistral API key (for Pixtral vision model)
- Bing scraping may be affected by site changes
- Rate limiting may apply (handled with retries)
- Image availability depends on source servers

## Future Enhancements

Potential improvements:

- [ ] Support for multiple image search sources
- [ ] Caching of evaluations
- [ ] Batch processing for multiple queries
- [ ] Custom evaluation criteria
- [ ] Image preprocessing/validation
- [ ] More detailed failure analytics

## Contributing

When extending this agent, follow these principles:

1. **Maintain async operations** throughout
2. **Add retry logic** for any external calls
3. **Use structured outputs** with Pydantic
4. **Include comprehensive error handling**
5. **Add observability** (logging, progress indicators)
6. **Write tests** for new functionality

## License

See project LICENSE file.

## Related

- `tools.imagesearch.bing_scraper`: Image search tool
- `agents.section_html_generator`: Example of async agent pattern
- `agents.index_generator`: Example of structured outputs

