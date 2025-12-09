# Course Generator Agent

An AI-powered course generation system that creates comprehensive educational content from topics or PDF syllabi.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd course-generator-agent
```

2. Install the package in editable mode:
```bash
pip3 install -e .
```

## Environment Setup

1. Create your own copy of the environment template:
```bash
cp env.example env.secrets
```

2. Edit `env.secrets` and add your API keys for the services you plan to use:
   - Gemini API key
   - Mistral API key
   - Pixtral API key
   - OpenAI API key
   - Groq API key
   - Langsmith API key (for tracing)
   - Tavily API key (for search)

3. Load the environment variables:
```bash
source env.secrets
```


## Usage

### Workflow 1: Generate Course from Topic

Generate a course from scratch by specifying a topic:

```bash
python3 -m main.workflow
python3 -m main.workflow_pdf
```

### Evaluate Course Output

Install eval dependencies

```bash
pip3 install -e ".[evaluation]"
```

Evaluate generated courses using LangSmith:

```bash

# Dataset mode
# Create evaluation dataset from course outputs
python3 -m evaluation.dataset create-dataset --inputs output/*.json

# Run full evaluation on a dataset
python3 -m evaluation.workflow evaluate --dataset my-courses

# Single file mode
# Quick evaluation of a single file
python3 -m evaluation.workflow quick --input output/course.json
```
