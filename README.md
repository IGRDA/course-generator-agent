# Course Generator Agent 🎓

An AI-powered course content generator using a **multi-agent workflow** with **shareable tools** connected through a unified orchestration system. Specialized agents collaborate to create comprehensive educational content from structural planning to detailed content generation.

## 🌟 Overview

The system implements **collaborative AI agents** working together through a **shared workflow orchestration**:

- **🔄 Multi-Agent Architecture**: Specialized agents for structure and content generation
- **⚡ Parallel Processing**: Concurrent section generation with configurable concurrency  
- **🌍 Multi-Language Support**: Generate courses in any language
- **📊 Structured Data Flow**: Type-safe data models using Pydantic
- **🔄 Retry Logic**: Robust error handling and automatic retries

## 🏗️ Architecture

### Generative Workflow
```
📥 Input → 🏗️ Skeleton Generation → 📝 Parallel Content Generation → 📚 Course Output
```

### Agent System

**1. Index Generator Agent** (`agents/index_generator/`)
- Creates course structural skeleton
- Optimizes module/submodule/section layout
- Calculates word distributions

**2. Section Theory Generator Agent** (`agents/section_theory_generator/`)  
- Generates detailed educational content
- Maintains contextual awareness across course hierarchy
- Supports parallel generation with semaphore control

### Shared Workflow Orchestration
The `main/workflow.py` uses **LangGraph StateGraph** to connect agents through shared `CourseState`, ensuring consistency, error recovery, and progress tracking.

## 🚀 Quick Start

### Installation
```bash
git clone <your-repo-url>
cd course-generator-agent
pip install -e .
export MISTRAL_API_KEY="your_mistral_api_key_here"
```

### Basic Usage
```python
from main.workflow import build_course_generation_graph
from main.state import CourseState

# Build workflow
app = build_course_generation_graph()

# Create course parameters
initial_state = CourseState(
    title="Introduction to Machine Learning",
    n_modules=1, n_submodules=1, n_sections=1, n_words=400,
    modules=[],
    total_pages=50,
    description="A comprehensive introduction to ML concepts",
    language="English",  # Any language supported
    concurrency=4
)

# Generate complete course
result = app.invoke(initial_state)
print(result.model_dump_json(indent=2))
```

## 📁 Project Structure

```
course-generator-agent/
├── agents/                    # Specialized AI agents
│   ├── index_generator/      # Course structure planning
│   └── section_theory_generator/  # Content generation
├── main/                     # Workflow orchestration  
│   ├── workflow.py          # LangGraph workflow definition
│   └── state.py            # Shared state models
├── tools/                    # Shared utilities (extensible)
└── notebooks/               # Development and testing
```

## 🔧 Configuration

- **`total_pages`**: Target course length
- **`language`**: Content generation language
- **`concurrency`**: Parallel section generation limit
- **`max_retries`**: Error recovery attempts

## 🛠️ Technical Stack

- **LLM**: Mistral AI with configurable models
- **Orchestration**: LangGraph for workflow management
- **Async**: Python asyncio for parallel processing
- **Validation**: Pydantic for type-safe data models
- **Framework**: LangChain for LLM integration

## 🔮 Extensibility

### Adding Agents
1. Create agent directory under `agents/`
2. Implement shared state compatibility
3. Add to workflow graph
4. Define agent-specific tools

### Shared Tools
The `tools/` directory enables:
- Cross-agent utility sharing
- External API integrations
- Custom validation logic
- Output formatting

---

**Built with LangGraph, Mistral AI, and Python** • *Modern AI agent architecture for scalable content generation*