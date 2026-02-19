# Course Generator Agent - Architecture & Agent Skills

This document provides comprehensive documentation for the AI-powered course generation system, designed to help Cursor AI understand the codebase architecture, agent capabilities, and common development tasks.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Core Data Models](#core-data-models)
4. [Workflow Pipelines](#workflow-pipelines)
5. [Agent Capabilities](#agent-capabilities)
6. [Tool Integrations](#tool-integrations)
7. [LLM Providers](#llm-providers)
8. [Evaluation System](#evaluation-system)
9. [Common Development Tasks](#common-development-tasks)

---

## Architecture Overview

The system uses **LangGraph** for workflow orchestration, treating course generation as a state machine where each node represents an agent performing a specific task.

### Dependency Groups

Default install (`pip install -e .`) is lightweight for text-based course generation. Heavy packages are in optional groups:

```bash
pip install -e "."                  # Default: LangChain, LLMs, web search, TTS, image search
pip install -e ".[pdf-extraction]"  # + docling, easyocr (PDF → Markdown)
pip install -e ".[ml]"             # + torch, torchaudio, transformers
pip install -e ".[all]"            # All heavy optional deps
pip install -e ".[evaluation]"     # Quality evaluation framework
pip install -e ".[macos-ocr]"      # macOS-only native OCR
```

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT SOURCES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Topic + Description          │  PDF Syllabus                               │
│  (workflow.py)                │  (workflow_pdf.py, workflow_pdf2podcast.py) │
└───────────────┬───────────────┴──────────────────┬──────────────────────────┘
                │                                  │
                ▼                                  ▼
┌───────────────────────────────┐  ┌──────────────────────────────────────────┐
│   Research Phase (optional)   │  │   PDF Extraction (Docling + OCR)         │
│   - Generate search queries   │  │   - Convert PDF to Markdown              │
│   - Execute web searches      │  │   - Extract structure from content       │
│   - Synthesize findings       │  │                                          │
└───────────────┬───────────────┘  └──────────────────┬───────────────────────┘
                │                                      │
                └──────────────┬───────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INDEX GENERATION                                     │
│  Generate course skeleton: Modules → Submodules → Sections                  │
│  (agents/index_generator or agents/pdf_index_generator)                     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONTENT GENERATION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Theory Generation     → Populate section.theory content                 │
│  2. Activities Generation → Generate quizzes and application activities     │
│  3. HTML Formatting       → Convert theory to structured HTML elements      │
│  4. Image Generation      → Find/rank images for HTML blocks                │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       OPTIONAL ENRICHMENT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  5. Bibliography Gen  → Generate book references per module (Open Library)  │
│  6. Podcast Gen       → Create host/guest dialogue + TTS audio              │
└───────────────────────────────────────────────────────────────────────────┬─┘
                                                                            │
                                    ┌───────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT GENERATION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  • course.json       - Full course data structure                           │
│  • course.html       - Rendered HTML version                                │
│  • module_N.json     - Individual module files                              │
│  • bibliography.json - Book references (APA 7 format, if enabled)           │
│  • podcast/          - Audio files (PDF2Podcast workflow)                   │
│  • book/             - PDF book via LaTeX (post-processing)                 │
│  • video_html_generator/ - Simplified JSON for video generation             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Workflow State Machine

```
START
  │
  ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ generate_index  │────▶│generate_theories│────▶│generate_activities│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌───────────────────────────────┘
                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│calculate_metadata│───▶│  generate_html  │────▶│ generate_images │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌───────────────────────────────┘
                        ▼
               ┌─────────────────────┐
               │generate_bibliography│  (optional, if enabled)
               └─────────────────────┘
                        │
                        ▼
                       END
```

---

## Directory Structure

```
course-generator-agent/
├── agents/                          # LangGraph agents (one per task)
│   ├── index_generator/             # Course skeleton generation
│   │   ├── agent.py                 # Main agent with LangGraph nodes
│   │   ├── prompts.py               # ChatPromptTemplate definitions
│   │   └── utils.py                 # Layout computation helpers
│   ├── pdf_index_generator/         # PDF-based index extraction
│   ├── section_theory_generator/    # Section content generation
│   ├── activities_generator/        # Quiz & activity generation
│   ├── html_formatter/              # Theory → HTML conversion
│   ├── image_search/                # Image search & ranking
│   ├── bibliography_generator/      # Book bibliography generation (APA 7)
│   ├── podcast_generator/           # Conversation script generation + TTS
│   └── video_html_generator/        # Deterministic JSON simplification for video
│
├── main/                            # Workflow orchestration
│   ├── state.py                     # Pydantic models (CourseState, CourseConfig)
│   ├── workflow.py                  # Topic-based workflow
│   ├── workflow_pdf.py              # PDF syllabus workflow
│   ├── workflow_pdf2podcast.py      # PDF → Podcast workflow
│   ├── output_manager.py            # Output file handling
│   └── audience_profiles.py         # Target audience configurations
│
├── tools/                           # External integrations
│   ├── websearch/                   # Web search providers
│   │   ├── factory.py               # create_web_search()
│   │   ├── ddg/                     # DuckDuckGo client
│   │   ├── tavily/                  # Tavily API client
│   │   └── wikipedia/               # Wikipedia client
│   ├── imagesearch/                 # Image search providers
│   │   ├── factory.py               # create_image_search()
│   │   ├── bing/                    # Bing Image Search
│   │   ├── ddg/                     # DuckDuckGo Images
│   │   ├── freepik/                 # Freepik API
│   │   └── google/                  # Google Images
│   ├── booksearch/                  # Book metadata search
│   │   ├── factory.py               # create_book_search()
│   │   └── openlibrary/             # Open Library API client
│   ├── pdf2md/                      # PDF conversion
│   │   └── docling/                 # Docling + OCR converter
│   ├── pdf_book/                    # PDF book generation via LaTeX
│   │   ├── factory.py               # Template selection
│   │   ├── generator.py             # LaTeX document generation
│   │   ├── utils.py                 # BibTeX, escaping helpers
│   │   └── templates/               # LaTeX templates (academic.tex)
│   ├── podcast/                     # TTS engines
│   │   ├── tts_engine.py            # Coqui & Edge TTS implementations
│   │   ├── edge_engine.py           # Edge TTS implementation
│   │   ├── audio_utils.py           # MP3 metadata, background music
│   │   └── models.py                # Language configs, conversation models
│   └── videosearch/                 # Video search providers
│
├── LLMs/                            # LLM provider abstractions
│   ├── text2text/                   # Text generation models
│   │   ├── factory.py               # create_text_llm()
│   │   ├── mistral/                 # Mistral client
│   │   ├── gemini/                  # Google Gemini client
│   │   ├── openai/                  # OpenAI client
│   │   ├── groq/                    # Groq client
│   │   └── deepseek/                # DeepSeek client
│   └── imagetext2text/              # Vision models
│       ├── factory.py               # create_vision_llm()
│       └── pixtral/                 # Pixtral vision client
│
├── evaluators/                      # Quality evaluation
│   ├── base.py                      # BaseEvaluator with rubric scoring
│   ├── index_evaluator/             # Course structure evaluation
│   ├── section_evaluator/           # Section content evaluation
│   ├── activities_evaluator/        # Quiz quality evaluation
│   ├── html_evaluator/              # HTML structure evaluation
│   └── overall_evaluator/           # End-to-end course evaluation
│
├── evaluation/                      # Evaluation workflows
│   ├── dataset.py                   # Dataset creation from outputs
│   ├── workflow.py                  # LangSmith evaluation runner
│   └── metrics/                     # Automated metrics
│       ├── embedding_metrics.py     # Semantic similarity
│       ├── nlp_metrics.py           # Readability, complexity
│       └── structure_metrics.py     # Structure validation
│
├── output/                          # Generated course outputs
├── env.example                      # Environment variable template
├── env.secrets                      # API keys (gitignored)
└── pyproject.toml                   # Dependencies & package config
                                     #   Default: lightweight text generation
                                     #   Optional: [pdf-extraction], [ml], [all]
```

---

## Core Data Models

All models defined in `main/state.py` using Pydantic v2.

### CourseConfig

Immutable configuration for course generation. Agents should READ but never MODIFY.

```python
class CourseConfig(BaseModel):
    # Content settings
    title: str
    total_pages: int = 50
    words_per_page: int = 400
    language: str = "English"
    target_audience: Literal["kids", "general", "advanced"] | None = None
    
    # LLM providers
    text_llm_provider: str = "mistral"          # mistral | gemini | groq | openai | deepseek
    vision_llm_provider: str = "pixtral"        # pixtral
    
    # Search providers
    web_search_provider: str = "ddg"            # ddg | tavily | wikipedia
    image_search_provider: str = "bing"         # bing | ddg | freepik | google
    
    # Processing settings
    concurrency: int = 8                        # Parallel section processing
    max_retries: int = 3                        # LLM retry attempts
    use_reflection: bool = False                # Fact verification
    
    # Research settings
    enable_research: bool = True
    research_max_queries: int = 5
    research_max_results_per_query: int = 3
    
    # Podcast settings
    podcast_target_words: int = 600
    podcast_tts_engine: Literal["edge", "coqui", "elevenlabs", "chatterbox"] = "edge"
    podcast_speaker_map: dict[str, str] | None = None
    
    # Bibliography settings
    generate_bibliography: bool = False              # Enable book bibliography generation
    bibliography_books_per_module: int = 5           # Books to recommend per module
```

### CourseState

Main state object passed through workflow. Agents modify content fields.

```python
class CourseState(BaseModel):
    config: CourseConfig                        # Immutable configuration
    research: CourseResearch | None = None      # Research phase output
    title: str                                  # Course title
    modules: list[Module] = []                  # Course content hierarchy
    bibliography: CourseBibliography | None     # Book bibliography (if enabled)
```

### Bibliography Models

```python
class BookReference(BaseModel):
    title: str                                  # Book title
    authors: list[str]                          # APA format: ['Last, F. M.']
    year: int | str | None                      # Publication year
    publisher: str | None                       # Publisher name
    isbn: str | None                            # ISBN-10
    isbn_13: str | None                         # ISBN-13
    url: str | None                             # Open Library URL
    apa_citation: str                           # Pre-formatted APA 7 citation

class ModuleBibliography(BaseModel):
    module_index: int                           # Module index (1-based)
    module_title: str                           # Module title
    books: list[BookReference]                  # Books for this module

class CourseBibliography(BaseModel):
    modules: list[ModuleBibliography]           # Per-module bibliographies
    all_books: list[BookReference]              # Deduplicated master list
```

### Content Hierarchy

```python
class Module(BaseModel):
    title: str
    id: str                                     # Set by calculate_metadata
    index: int                                  # Set by calculate_metadata
    description: str
    duration: float                             # Hours, calculated
    submodules: list[Submodule]

class Submodule(BaseModel):
    title: str
    index: int
    description: str
    duration: float
    sections: list[Section]

class Section(BaseModel):
    title: str
    index: int
    description: str
    summary: str                                # 3-line preview
    theory: str                                 # Main content text
    html: list[HtmlElement] | None              # Structured HTML
    activities: ActivitiesSection | None        # Quiz + application
    meta_elements: MetaElements | None          # Glossary, facts, quotes
```

### HTML Elements

```python
class HtmlElement(BaseModel):
    type: Literal["p", "ul", "quote", "table", 
                  "paragraphs", "accordion", "tabs", 
                  "carousel", "flip", "timeline", "conversation"]
    content: str | list[str] | dict | list[ParagraphBlock]

class ParagraphBlock(BaseModel):
    title: str
    icon: str                                   # Material Design Icon class
    image: dict | None                          # {type, query, content URL}
    elements: list[HtmlElement]
```

---

## Workflow Pipelines

### 1. Topic-Based Workflow (`main/workflow.py`)

Generates course from a topic title and description.

```bash
python3 -m main.workflow --total-pages 50

# With bibliography generation enabled
python3 -m main.workflow --total-pages 50 --generate-bibliography

# Custom topic (e.g. Excel 10-page course)
python3 -m main.workflow --total-pages 10 --title "Data Transformation and Modeling with Power Query and Power Pivot"
```

Structure is generated by the index agent from the title (topic-based); it is not parsed from a markdown index file. Output goes to a timestamped folder under `output/`.

**Pipeline stages:**
1. `generate_index` - Research topic (optional) + create skeleton
2. `generate_theories` - Populate section content (parallel)
3. `generate_activities` - Create quizzes (parallel)
4. `calculate_metadata` - Set IDs, indexes, durations
5. `generate_html` - Convert to HTML structure (parallel)
6. `generate_images` - Find images for blocks (parallel)
7. `generate_bibliography` - Generate book references (optional, if enabled)

### 2. PDF-Based Workflow (`main/workflow_pdf.py`)

Extracts course structure from PDF syllabus. Requires `pdf-extraction` optional deps: `pip install -e ".[pdf-extraction]"`

```bash
python3 -m main.workflow_pdf path/to/syllabus.pdf --total-pages 100

# With bibliography generation
python3 -m main.workflow_pdf syllabus.pdf --generate-bibliography
```

**Pipeline stages:**
1. `generate_index_from_pdf` - Convert PDF → extract structure
2. `generate_theories` - Same as topic workflow
3. `generate_activities` - Same as topic workflow
4. `calculate_metadata` - Same as topic workflow
5. `generate_html` - Same as topic workflow
6. `generate_images` - Same as topic workflow
7. `generate_bibliography` - Generate book references (optional, if enabled)

### 3. PDF-to-Podcast Workflow (`main/workflow_pdf2podcast.py`)

Generates audio podcasts from PDF syllabus. Skips activities/HTML/images. Requires `pdf-extraction` optional deps: `pip install -e ".[pdf-extraction]"`

```bash
python3 -m main.workflow_pdf2podcast syllabus.pdf --target-words 600 --tts-engine edge
```

**Pipeline stages:**
1. `generate_index_from_pdf` - Extract structure
2. `generate_theories` - Generate content
3. `calculate_metadata` - Set IDs
4. `generate_podcasts` - Generate conversation + synthesize audio

### 4. Post-Processing Tools

These tools run after main course generation to produce additional outputs.

**PDF Book Generation:**
```bash
# Generate PDF book from completed course
python3 -m tools.pdf_book output/course_name/course.json

# With custom output directory
python3 -m tools.pdf_book output/course_name/course.json --output-dir ./book

# Skip image downloads (faster, no images in PDF)
python3 -m tools.pdf_book output/course_name/course.json --no-images
```

**Video HTML Simplification:**
```bash
# Simplify a module for video generation
python3 -m agents.video_html_generator output/course_name/module_0.json

# Custom output directory
python3 -m agents.video_html_generator output/course_name/module_0.json --output-dir ./video

# Preview output without saving
python3 -m agents.video_html_generator output/course_name/module_0.json --preview
```

---

## Agent Capabilities

### Index Generator (`agents/index_generator/`)

**Purpose:** Creates course skeleton with modules, submodules, and sections.

**Key functions:**
- `generate_course_state()` - Main entry point
- `research_topic()` - Web search + synthesis (English)
- `generate_titles_phase()` - Hierarchical title generation
- `expand_descriptions_phase()` - Add descriptions to titles
- `generate_all_summaries()` - 3-line section previews

**LangGraph patterns:**
- Uses `build_skeleton_graph()` subgraph with 4 sequential nodes
- Uses `build_summary_graph()` with `Send` pattern for parallel module processing
- `RetryPolicy` for automatic retry on parse failures

**Output:** `CourseState` with empty `theory` fields, populated structure.

---

### PDF Index Generator (`agents/pdf_index_generator/`)

**Purpose:** Extracts course structure from PDF document.

**Key functions:**
- `generate_pdf_course_state()` - Main entry point
- Uses Docling for PDF → Markdown conversion
- Parses markdown structure into course hierarchy

**Output:** Same as Index Generator but derived from PDF content.

---

### Section Theory Generator (`agents/section_theory_generator/`)

**Purpose:** Generates educational content for each section.

**Key functions:**
- `generate_all_section_theories()` - Parallel section processing
- `generate_section_theory()` - Single section generation
- Optional reflection pattern for fact verification

**Features:**
- Uses research context if available
- Supports `use_reflection=True` for web search verification
- Configurable `words_per_page` for length control

**Output:** Populates `section.theory` text content.

---

### Activities Generator (`agents/activities_generator/`)

**Purpose:** Creates interactive quiz activities for each section.

**Activity types:**
| Type | Description |
|------|-------------|
| `order_list` | Arrange items in correct sequence |
| `fill_gaps` | Complete sentences with missing words |
| `swipper` | Swipe true/false classification |
| `linking_terms` | Match related concepts |
| `multiple_choice` | Single correct answer quiz |
| `multi_selection` | Multiple correct answers quiz |

**Application activities:**
- `group_activity` - Collaborative exercise
- `discussion_forum` - Open discussion prompt
- `individual_project` - Personal assignment
- `open_ended_quiz` - Free-form response

**Output:** Populates `section.activities` with quiz + application activities.

---

### HTML Formatter (`agents/html_formatter/`)

**Purpose:** Converts theory text to structured HTML elements.

**Key functions:**
- `generate_all_section_html()` - Parallel HTML generation
- `format_section_html()` - Single section formatting

**Format selection modes:**
- `select_html="LLM"` - LLM chooses best format per section
- `select_html="random"` - Random format from available options

**Available formats:** `paragraphs`, `accordion`, `tabs`, `carousel`, `flip`, `timeline`, `conversation`

**Output:** Populates `section.html` with `HtmlElement` list.

---

### Image Search (`agents/image_search/`)

**Purpose:** Finds and optionally ranks images for HTML blocks.

**Key functions:**
- `generate_all_section_images()` - Process all sections
- Uses image search providers (Bing, DDG, Freepik, Google)

**Ranking modes:**
- `use_vision_ranking=False` - Pick first search result (fast)
- `use_vision_ranking=True` - Use Pixtral to score/rank images (quality)

**Output:** Populates `image.content` URLs in `ParagraphBlock` elements.

---

### Bibliography Generator (`agents/bibliography_generator/`)

**Purpose:** Generates book bibliographies for course modules using a hybrid LLM + API approach.

**Workflow:**
1. LLM suggests relevant books based on module content and topics
2. Open Library API validates book existence and enriches metadata
3. Citations are formatted in APA 7 style
4. Deduplication prevents repeated books across modules

**Key functions:**
- `generate_course_bibliography()` - Main entry point, processes all modules
- `generate_module_bibliography()` - Single module bibliography
- `_format_apa_citation()` - APA 7 citation formatting

**Configuration:**
- `generate_bibliography: bool = False` - Enable/disable in CourseConfig
- `bibliography_books_per_module: int = 5` - Target books per module

**Output:** Populates `state.bibliography` with `CourseBibliography` containing:
- Per-module book lists
- Deduplicated master list sorted by author
- APA 7 formatted citations

---

### Video HTML Generator (`agents/video_html_generator/`)

**Purpose:** Deterministic (no LLM) simplification of module JSON for video generation.

**What it does:**
- Extracts only essential fields for video rendering
- Truncates HTML content arrays to first item (reduces payload size)
- Preserves structure: Module → Submodule → Section hierarchy

**Extracted fields:**
| Level | Fields |
|-------|--------|
| Module | `title`, `id`, `index`, `description`, `duration`, `type` |
| Submodule | `title`, `index`, `description`, `duration` |
| Section | `title`, `index`, `description`, `html` (truncated) |

**Key functions:**
- `simplify_module()` - Transform module dict in memory
- `simplify_module_from_path()` - Load, simplify, save to `video_html_generator/` subfolder

**CLI usage:**
```bash
python3 -m agents.video_html_generator output/course/module_0.json
python3 -m agents.video_html_generator module.json --output-dir ./custom_output
python3 -m agents.video_html_generator module.json --preview  # Print to stdout
```

**Output:** Simplified JSON files in `video_html_generator/` directory.

---

### Podcast Generator (`agents/podcast_generator/`)

**Purpose:** Creates conversational podcast scripts from course content with TTS audio synthesis.

**Key functions:**
- `generate_conversation()` - Create host/guest dialogue from module content
- `generate_module_podcast()` - Full pipeline: conversation + TTS synthesis
- `extract_module_context()` - Extract sections and summaries for prompt

**TTS Engines:**
| Engine | Features |
|--------|----------|
| Edge TTS | Cloud-based, many voices, SSML support, default |
| Coqui TTS | Local, offline, multi-speaker models |

**Audio features:**
- Background music support (intro/outro with fade)
- MP3 metadata: title, artist, album, track number
- Language-aware voice selection
- Custom speaker mapping for voices

**Language mapping:**
- `español`/`spanish` → `es`
- `english`/`inglés` → `en`
- `french`/`français` → `fr`
- `german`/`deutsch` → `de`

**Configuration:**
- `podcast_target_words: int = 600` - Target word count per episode
- `podcast_tts_engine: Literal["edge", "coqui", "elevenlabs", "chatterbox"] = "edge"` - TTS engine
- `podcast_speaker_map: dict[str, str] | None` - Voice mapping

**Output:** 
- `module_N_conversation.json` - Dialogue script
- `module_N.mp3` - Synthesized audio with background music

---

## Tool Integrations

### Web Search (`tools/websearch/`)

```python
from tools.websearch import create_web_search

search = create_web_search("ddg")  # ddg | tavily | wikipedia
results = search("machine learning basics", max_results=5)
```

**Providers:**
| Provider | API Key Required | Notes |
|----------|------------------|-------|
| `ddg` | No | DuckDuckGo, free, no rate limits |
| `tavily` | Yes | Research-focused, high quality |
| `wikipedia` | No | Encyclopedia articles only |

---

### Image Search (`tools/imagesearch/`)

```python
from tools.imagesearch import create_image_search

search = create_image_search("bing")  # bing | ddg | freepik | google
images = search("neural network diagram", max_results=10)
# Returns: [{"url": "...", "title": "...", "thumbnail": "..."}, ...]
```

**Providers:**
| Provider | Method | Notes |
|----------|--------|-------|
| `bing` | Playwright scraping | Good quality, reliable |
| `ddg` | API | Free, may have limits |
| `freepik` | API | Stock images, API key needed |
| `google` | Playwright scraping | Rate limited |

---

### PDF Conversion (`tools/pdf2md/`)

> **Requires optional dependency group:** `pip install -e ".[pdf-extraction]"`

```python
from tools.pdf2md.docling import convert_pdf_to_markdown

markdown_content = convert_pdf_to_markdown("syllabus.pdf")
```

Uses Docling library with EasyOCR (cross-platform) or ocrmac (macOS only).

---

### TTS/Podcast (`tools/podcast/`)

```python
from tools.podcast.tts_engine import generate_podcast_edge

generate_podcast_edge(
    conversation=[{"role": "host", "content": "Welcome!"}],
    output_path="episode.mp3",
    language="es",  # Spanish
    speaker_map={"host": "es-ES-AlvaroNeural", "guest": "es-ES-XimenaNeural"},
    # Background music options
    music_path="background_music.mp3",
    intro_duration_ms=10000,
    outro_duration_ms=10000,
    intro_fade_ms=5000,
    outro_fade_ms=5000,
    # MP3 metadata
    title="Module 1: Introduction",
    artist="Adinhub",
    album="Course Name",
    track_number=1,
)
```

---

### Book Search (`tools/booksearch/`)

```python
from tools.booksearch.openlibrary.client import search_books, validate_book

# Search for books by query
results = search_books("machine learning python", max_results=5)
# Returns: [{"title": "...", "authors": [...], "year": 2020, "isbn": "...", ...}]

# Validate a specific book exists
book = validate_book("Introduction to Algorithms", author="Cormen")
# Returns BookResult or None if not found

# Search by title and author
from tools.booksearch.openlibrary.client import search_books_by_title_author
results = search_books_by_title_author("Clean Code", "Robert Martin")
```

**Features:**
- No API key required (Open Library is free)
- Returns ISBN-10, ISBN-13, cover URLs
- Authors formatted for APA citations
- Deduplication by ISBN or title+author

---

### PDF Book Generator (`tools/pdf_book/`)

Converts `course.json` to professional academic PDF books via LaTeX.

```python
from tools.pdf_book.generator import generate_pdf_book

pdf_path = generate_pdf_book(
    course_json_path="output/course/course.json",
    output_dir="output/course/book",  # Optional, defaults to course_dir/book/
    template="academic",               # LaTeX template
    download_images=True,              # Download images from URLs
    cleanup=True,                      # Remove intermediate .aux, .log files
)
```

**CLI usage:**
```bash
# Generate PDF in default location
python3 -m tools.pdf_book output/course/course.json

# Custom output directory
python3 -m tools.pdf_book output/course/course.json --output-dir ./my_book

# Skip image downloads
python3 -m tools.pdf_book output/course/course.json --no-images

# Keep LaTeX files for debugging
python3 -m tools.pdf_book output/course/course.json --no-cleanup
```

**Features:**
- Professional academic layout with cover page
- Automatic bibliography from `bibliography.json` (BibTeX + Biber)
- Image downloading and validation (PNG, JPEG, PDF only)
- Module objectives boxes
- XeLaTeX for Unicode support (Greek letters, math symbols)
- Table of contents with hyperlinks

**Requirements:**
- TeX Live or BasicTeX installed (`xelatex` or `pdflatex` in PATH)
- Biber for bibliography processing

**Output structure:**
```
book/
├── book.tex          # Generated LaTeX document
├── book.pdf          # Final PDF output
├── references.bib    # BibTeX bibliography (if available)
└── images/           # Downloaded images
```

---

## LLM Providers

### Text Generation (`LLMs/text2text/`)

```python
from LLMs.text2text import create_text_llm, resolve_text_model_name

# Create LLM instance
model_name = resolve_text_model_name("mistral")  # From env var
llm = create_text_llm(provider="mistral", model_name=model_name, temperature=0)

# Use with LangChain
from langchain_core.prompts import ChatPromptTemplate
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "..."})
```

**Available providers:**

| Provider | Env Var | Default Model |
|----------|---------|---------------|
| `mistral` | `MISTRAL_API_KEY` | `mistral-small-latest` |
| `gemini` | `GEMINI_API_KEY` | `gemini-flash-latest` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `groq` | `GROQ_API_KEY` | `moonshotai/kimi-k2-instruct` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |

### Vision Models (`LLMs/imagetext2text/`)

```python
from LLMs.imagetext2text import create_vision_llm

vision_llm = create_vision_llm(provider="pixtral")
# Used for image ranking in image_search
```

---

## Evaluation System

### Base Evaluator Pattern

All evaluators inherit from `evaluators/base.py`:

```python
from evaluators.base import BaseEvaluator, RubricScore

class MyEvaluator(BaseEvaluator):
    def evaluate(self, course_state: CourseState) -> dict:
        # Use self.evaluate_with_rubric() for LLM scoring
        score = self.evaluate_with_rubric(
            prompt=MY_PROMPT,
            output_model=RubricScore,
            prompt_variables={"content": "..."}
        )
        return {"score": score.score, "reasoning": score.reasoning}
```

### Available Evaluators

| Evaluator | Criteria |
|-----------|----------|
| `index_evaluator` | Coverage, structure, balance |
| `section_evaluator` | Accuracy, clarity, engagement |
| `activities_evaluator` | Difficulty, variety, alignment |
| `html_evaluator` | Formatting, accessibility |
| `overall_evaluator` | End-to-end quality |

### Running Evaluations

```bash
# Create dataset from generated courses
python3 -m evaluation.dataset create-dataset --inputs output/*.json

# Run evaluation workflow
python3 -m evaluation.workflow evaluate --dataset my-courses

# Quick single-file evaluation
python3 -m evaluation.workflow quick --input output/course.json
```

---

## Common Development Tasks

### Adding a New LLM Provider

1. Create client in `LLMs/text2text/{provider}/client.py`:

```python
from langchain_core.language_models.chat_models import BaseChatModel

def build_my_provider_chat_model(**kwargs) -> BaseChatModel:
    # Return LangChain-compatible chat model
    ...
```

2. Register in `LLMs/text2text/factory.py`:

```python
BUILDERS["my_provider"] = build_my_provider_chat_model
MODEL_ENV_VARS["my_provider"] = "MY_PROVIDER_MODEL_NAME"
```

3. Add env vars to `env.example`.

---

### Adding a New Web Search Provider

1. Create client in `tools/websearch/{provider}/client.py`:

```python
def web_search(query: str, max_results: int) -> str:
    # Return formatted search results as string
    ...
```

2. Register in `tools/websearch/factory.py`:

```python
SEARCH_PROVIDERS["my_provider"] = my_web_search
```

---

### Adding a New Agent

1. Create directory `agents/{agent_name}/` with:
   - `__init__.py` - Export main functions
   - `agent.py` - Agent logic
   - `prompts.py` - Prompt templates

2. Agent function signature:

```python
def my_agent_node(state: CourseState, config: RunnableConfig = None) -> CourseState:
    # Read from state.config for settings
    # Modify state content fields
    # Return updated state
    ...
```

3. Add node to workflow in `main/workflow.py`:

```python
graph.add_node("my_agent", my_agent_node)
graph.add_edge("previous_node", "my_agent")
graph.add_edge("my_agent", "next_node")
```

---

### Adding a New Activity Type

1. Add content model in `main/state.py`:

```python
class MyActivityContent(BaseModel):
    question: str
    solution: ...
```

2. Add to `Activity.type` literal and union.

3. Update `agents/activities_generator/prompts.py` with generation prompt.

---

### Modifying Course Structure

The course hierarchy is fixed: `Module → Submodule → Section`

To add a new field to sections:

1. Add field to `Section` model in `main/state.py`
2. Create/modify agent to populate the field
3. Update HTML exporter if needed for rendering

---

## Debugging Tips

1. **Enable LangSmith tracing:**
   ```bash
   export LANGCHAIN_TRACING_V2=true
   export LANGSMITH_API_KEY=your_key
   ```

2. **Quick test runs:**
   ```bash
   python3 -m main.workflow --total-pages 2
   ```

3. **Check intermediate outputs:**
   - Look in `output/{course}_{timestamp}/` for step snapshots
   - Files: `step_index.json`, `step_theories.json`, etc.

4. **Parse failures:**
   - Check if LLM wrapped JSON in markdown fences
   - Use `strip_markdown_fences()` helper
   - Enable `RetryWithErrorOutputParser` for auto-correction

5. **Rate limits:**
   - Reduce `concurrency` in CourseConfig
   - Switch to different provider
   - Add delays between API calls








