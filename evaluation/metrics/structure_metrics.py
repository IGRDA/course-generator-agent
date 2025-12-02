"""Structure-based metrics for course analysis.

Provides:
- Title uniqueness (exact match and n-gram based)
"""

from typing import Dict, Any, List
from langsmith import traceable
from main.state import CourseState


@traceable(name="compute_title_uniqueness")
def compute_title_uniqueness(course_state: CourseState) -> Dict[str, Any]:
    """
    Compute title uniqueness at each level using exact match and n-gram analysis.
    
    Args:
        course_state: The CourseState to analyze
        
    Returns:
        Dictionary with exact and n-gram based uniqueness scores
    """
    module_titles = [m.title for m in course_state.modules]
    submodule_titles = [
        sm.title 
        for m in course_state.modules 
        for sm in m.submodules
    ]
    section_titles = [
        s.title 
        for m in course_state.modules 
        for sm in m.submodules 
        for s in sm.sections
    ]
    
    def exact_uniqueness(titles: List[str]) -> float:
        if not titles:
            return 1.0
        return len(set(titles)) / len(titles)
    
    def ngram_uniqueness(titles: List[str]) -> float:
        """Compute uniqueness based on weighted n-gram overlap."""
        if len(titles) < 2:
            return 1.0
        
        # Compute pairwise n-gram similarities
        similarities = []
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                sim = _weighted_ngram_similarity(titles[i], titles[j])
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Uniqueness = 1 - average similarity
        avg_sim = sum(similarities) / len(similarities)
        return round(1 - avg_sim, 4)
    
    return {
        "module_uniqueness": round(exact_uniqueness(module_titles), 4),
        "submodule_uniqueness": round(exact_uniqueness(submodule_titles), 4),
        "section_uniqueness": round(exact_uniqueness(section_titles), 4),
        "module_ngram_uniqueness": ngram_uniqueness(module_titles),
        "submodule_ngram_uniqueness": ngram_uniqueness(submodule_titles),
        "section_ngram_uniqueness": ngram_uniqueness(section_titles),
        "total_modules": len(module_titles),
        "total_submodules": len(submodule_titles),
        "total_sections": len(section_titles),
    }


def _weighted_ngram_similarity(text1: str, text2: str) -> float:
    """
    Compute weighted n-gram similarity between two texts.
    Uses linear weights: 2gram=1, 3gram=2, 4gram=3 (normalized by 6).
    """
    t1 = text1.lower().split()
    t2 = text2.lower().split()
    
    if len(t1) < 2 or len(t2) < 2:
        return 1.0 if t1 == t2 else 0.0
    
    def get_ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    def jaccard(set1, set2):
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    sim_2 = jaccard(get_ngrams(t1, 2), get_ngrams(t2, 2)) if len(t1) >= 2 and len(t2) >= 2 else 0
    sim_3 = jaccard(get_ngrams(t1, 3), get_ngrams(t2, 3)) if len(t1) >= 3 and len(t2) >= 3 else 0
    sim_4 = jaccard(get_ngrams(t1, 4), get_ngrams(t2, 4)) if len(t1) >= 4 and len(t2) >= 4 else 0
    
    # Weighted sum: penalize higher n-gram matches more
    return (1 * sim_2 + 2 * sim_3 + 3 * sim_4) / 6

