import os
import asyncio
from main.state import CourseState
from langchain_mistralai import ChatMistralAI
from .prompts import section_theory_prompt


MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")

# Initialize the LLM for section theory generation
llm = ChatMistralAI(
    model=MODEL_NAME,
    temperature=0.7,  # Slightly higher temperature for more creative content
)


async def generate_section_theory(
    course_state: CourseState,
    module_idx: int,
    submodule_idx: int, 
    section_title: str
) -> str:
    """
    Generate theory content for a specific section using Mistral AI.
    
    Args:
        course_state: The full course state for context
        module_idx: Index of the module (0-based)
        submodule_idx: Index of the submodule (0-based) 
        section_idx: Index of the section (0-based)
        section_title: Title of the section
        
    Returns:
        Generated theory content as string
    """
    # Extract context from course state
    module = course_state.modules[module_idx]
    submodule = module.submodules[submodule_idx]

    n_words = course_state.config.total_pages * course_state.config.words_per_page // sum(len(submodule.sections) for module in course_state.modules for submodule in module.submodules)

    
    # Format the prompt with context
    prompt = section_theory_prompt.format(
        course_title=course_state.title,
        module_title=module.title,
        submodule_title=submodule.title,
        section_title=section_title,
        language=course_state.language,
        n_words=n_words
    )
    
    # Generate content using the LLM
    response = await llm.ainvoke(prompt)
    
    return response.content.strip()


class ParallelSectionUpdater:
    """
    Executor that generates theory content for all sections in parallel.
    Uses the existing CourseState and integrates with LangGraph workflow.
    """
    
    def __init__(
        self,
        course_state: CourseState,
        concurrency: int = 8
    ):
        self.course_state = course_state
        self.concurrency = concurrency
        self._tasks = []  # list of (m_idx, sm_idx, s_idx, title)
    
    def plan_all_sections(self) -> None:
        """Collect every section as a 'node' to run."""
        self._tasks.clear()
        for m_idx, module in enumerate(self.course_state.modules):
            for sm_idx, submodule in enumerate(module.submodules):
                for s_idx, section in enumerate(submodule.sections):
                    self._tasks.append((m_idx, sm_idx, s_idx, section.title))
    
    async def _run_one(self, sem: asyncio.Semaphore, m_idx, sm_idx, s_idx, title):
        """Generate theory for one section with semaphore for concurrency control."""
        async with sem:
            try:
                theory = await generate_section_theory(
                    self.course_state, m_idx, sm_idx, title
                )
                # Update the course state in-place
                self.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx].theory = theory
                print(f"✓ Generated theory for Module {m_idx+1}, Submodule {sm_idx+1}, Section {s_idx+1}")
            except Exception as e:
                print(f"✗ Error generating theory for Module {m_idx+1}, Submodule {sm_idx+1}, Section {s_idx+1}: {e}")
                # Set a fallback theory content
                fallback_theory = f"Theory content for {title} (generation failed, please regenerate this section)"
                self.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx].theory = fallback_theory
    
    async def run(self) -> CourseState:
        """Execute all section theory generation in parallel."""
        if not self._tasks:
            self.plan_all_sections()
        
        print(f"Starting parallel generation of {len(self._tasks)} sections with concurrency {self.concurrency}")
        
        sem = asyncio.Semaphore(self.concurrency)
        await asyncio.gather(
            *(self._run_one(sem, m_idx, sm_idx, s_idx, title) 
              for (m_idx, sm_idx, s_idx, title) in self._tasks)
        )
        
        print("All section theories generated!")
        return self.course_state


async def generate_all_section_theories(course_state: CourseState, concurrency: int = 8) -> CourseState:
    """
    Main function to generate all section theories in parallel.
    
    Args:
        course_state: CourseState with skeleton structure (empty theories)
        concurrency: Number of concurrent requests to make
        
    Returns:
        Updated CourseState with all theories filled
    """
    updater = ParallelSectionUpdater(course_state, concurrency)
    return await updater.run()

