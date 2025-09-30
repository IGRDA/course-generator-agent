import os
import asyncio
from html.parser import HTMLParser
from pydantic import BaseModel, Field
from main.state import CourseState
from langchain_mistralai import ChatMistralAI
from .prompts import section_html_prompt


MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")


class SectionHTML(BaseModel):
    """Structured output for section HTML content."""
    html_content: str = Field(
        ...,
        description="Valid HTML content for the section. Must use semantic HTML tags (h2, h3, p, ul, ol, li, strong, em). No DOCTYPE, html, head, or body tags."
    )


def validate_html(html_content: str) -> tuple[bool, str]:
    """Validate HTML content using built-in parser.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Basic checks
        if not html_content.strip():
            return False, "HTML content is empty"
        
        # Check for forbidden tags
        forbidden = ['<!DOCTYPE', '<html', '<head', '<body']
        for tag in forbidden:
            if tag.lower() in html_content.lower():
                return False, f"Contains forbidden tag: {tag}"
        
        # Validate using built-in HTMLParser
        parser = HTMLParser()
        parser.feed(html_content)
        parser.close()
        
        return True, ""
    except Exception as e:
        return False, f"HTML validation error: {str(e)}"


# Initialize the LLM for HTML generation
llm = ChatMistralAI(
    model=MODEL_NAME,
    temperature=0.3,  # Lower temperature for more consistent HTML structure
)

# Create structured output LLM
structured_llm = llm.with_structured_output(SectionHTML)


async def retry_async_llm_call(llm_call_func, max_retries: int = 3, delay: float = 1.0):
    """
    Retry async LLM function call with exponential backoff.
    
    Args:
        llm_call_func: Async function that makes the LLM call
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        
    Returns:
        Result of the LLM call
    """
    for attempt in range(max_retries):
        try:
            return await llm_call_func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"✗ LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(delay * (2 ** attempt))
    
    raise Exception("Max retries exceeded")


async def generate_section_html(
    course_state: CourseState,
    module_idx: int,
    submodule_idx: int, 
    section_title: str,
    theory: str,
    max_retries: int = 3
) -> str:
    """
    Convert theory content to HTML for a specific section using Mistral AI with retry logic.
    
    Args:
        course_state: The full course state for context
        module_idx: Index of the module (0-based)
        submodule_idx: Index of the submodule (0-based) 
        section_title: Title of the section
        theory: Theory content to convert to HTML
        max_retries: Maximum number of retry attempts for LLM calls
        
    Returns:
        Generated HTML content as string
    """
    # Extract context from course state
    module = course_state.modules[module_idx]
    submodule = module.submodules[submodule_idx]
    
    # Format the prompt with context
    prompt = section_html_prompt.format(
        course_title=course_state.title,
        module_title=module.title,
        submodule_title=submodule.title,
        section_title=section_title,
        theory=theory
    )
    
    # Generate and validate HTML with retry logic
    for attempt in range(max_retries):
        try:
            # Generate HTML using structured LLM
            response = await structured_llm.ainvoke(prompt)
            html_content = response.html_content.strip()
            
            # Validate the generated HTML
            is_valid, error_msg = validate_html(html_content)
            
            if is_valid:
                return html_content
            else:
                print(f"✗ HTML validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid HTML after {max_retries} attempts: {error_msg}")
                await asyncio.sleep(1.0 * (2 ** attempt))
        
        except ValueError:
            raise
        except Exception as e:
            print(f"✗ HTML generation failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(1.0 * (2 ** attempt))
    
    raise Exception("Max retries exceeded for HTML generation")


class ParallelHtmlUpdater:
    """
    Executor that generates HTML content for all sections in parallel.
    Uses the existing CourseState and integrates with LangGraph workflow.
    """
    
    def __init__(
        self,
        course_state: CourseState,
        concurrency: int = 8,
        max_retries: int = 5
    ):
        self.course_state = course_state
        self.concurrency = concurrency
        self.max_retries = max_retries
        self._tasks = []  # list of (m_idx, sm_idx, s_idx, title, theory)
    
    def plan_all_sections(self) -> None:
        """Collect every section as a 'node' to run."""
        self._tasks.clear()
        for m_idx, module in enumerate(self.course_state.modules):
            for sm_idx, submodule in enumerate(module.submodules):
                for s_idx, section in enumerate(submodule.sections):
                    self._tasks.append((m_idx, sm_idx, s_idx, section.title, section.theory))
    
    async def _run_one(self, sem: asyncio.Semaphore, m_idx, sm_idx, s_idx, title, theory):
        """Generate HTML for one section with semaphore for concurrency control."""
        async with sem:
            try:
                html = await generate_section_html(
                    self.course_state, m_idx, sm_idx, title, theory, self.max_retries
                )
                # Update the course state in-place
                self.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx].html = html
                print(f"✓ Generated HTML for Module {m_idx+1}, Submodule {sm_idx+1}, Section {s_idx+1}")
            except Exception as e:
                print(f"✗ Error generating HTML for Module {m_idx+1}, Submodule {sm_idx+1}, Section {s_idx+1}: {e}")
                # Set a fallback HTML content
                fallback_html = f"<div class='error'><p>HTML generation failed for {title}</p></div>"
                self.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx].html = fallback_html
    
    async def run(self) -> CourseState:
        """Execute all section HTML generation in parallel."""
        if not self._tasks:
            self.plan_all_sections()
        
        print(f"Starting parallel HTML generation of {len(self._tasks)} sections with concurrency {self.concurrency}")
        
        sem = asyncio.Semaphore(self.concurrency)
        await asyncio.gather(
            *(self._run_one(sem, m_idx, sm_idx, s_idx, title, theory) 
              for (m_idx, sm_idx, s_idx, title, theory) in self._tasks)
        )
        
        print("All section HTML generated!")
        return self.course_state


async def generate_all_section_html(course_state: CourseState, concurrency: int = 8, max_retries: int = 3) -> CourseState:
    """
    Main function to generate HTML for all sections in parallel.
    
    Args:
        course_state: CourseState with theories populated
        concurrency: Number of concurrent requests to make
        max_retries: Maximum number of retry attempts for each LLM call
        
    Returns:
        Updated CourseState with all HTML filled
    """
    updater = ParallelHtmlUpdater(course_state, concurrency, max_retries)
    return await updater.run()
