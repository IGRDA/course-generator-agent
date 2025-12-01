"""Structure-based metrics for course analysis.

Leverages the hierarchical CourseState structure to compute:
- Title similarity and uniqueness
- Duplicate title detection
- Hierarchy balance analysis
"""

from typing import Dict, Any, List, Tuple
from langsmith import traceable
from main.state import CourseState


@traceable(name="compute_title_uniqueness")
def compute_title_uniqueness(course_state: CourseState) -> Dict[str, Any]:
    """
    Compute title uniqueness at each level of the hierarchy.
    
    Args:
        course_state: The CourseState to analyze
        
    Returns:
        Dictionary with uniqueness scores
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
    
    def uniqueness_score(titles: List[str]) -> float:
        if not titles:
            return 1.0
        return len(set(titles)) / len(titles)
    
    return {
        "module_uniqueness": round(uniqueness_score(module_titles), 4),
        "submodule_uniqueness": round(uniqueness_score(submodule_titles), 4),
        "section_uniqueness": round(uniqueness_score(section_titles), 4),
        "total_modules": len(module_titles),
        "unique_modules": len(set(module_titles)),
        "total_submodules": len(submodule_titles),
        "unique_submodules": len(set(submodule_titles)),
        "total_sections": len(section_titles),
        "unique_sections": len(set(section_titles)),
    }


@traceable(name="compute_hierarchy_balance")
def compute_hierarchy_balance(course_state: CourseState) -> Dict[str, Any]:
    """
    Analyze the balance of the course hierarchy.
    
    Args:
        course_state: The CourseState to analyze
        
    Returns:
        Dictionary with balance metrics
    """
    import statistics
    
    # Submodules per module
    submodules_per_module = [len(m.submodules) for m in course_state.modules]
    
    # Sections per submodule
    sections_per_submodule = [
        len(sm.sections) 
        for m in course_state.modules 
        for sm in m.submodules
    ]
    
    def compute_stats(values: List[int]) -> Dict[str, float]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}
        if len(values) == 1:
            return {"min": values[0], "max": values[0], "mean": values[0], "std": 0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 2),
            "std": round(statistics.stdev(values), 2)
        }
    
    submodule_stats = compute_stats(submodules_per_module)
    section_stats = compute_stats(sections_per_submodule)
    
    # Balance score: 1.0 is perfectly balanced, lower is less balanced
    # Based on coefficient of variation (lower CV = more balanced)
    def balance_score(stats: Dict[str, float]) -> float:
        if stats["mean"] == 0:
            return 1.0
        cv = stats["std"] / stats["mean"]
        return round(max(0, 1 - cv), 4)
    
    return {
        "num_modules": len(course_state.modules),
        "submodules_per_module": submodule_stats,
        "submodule_balance_score": balance_score(submodule_stats),
        "sections_per_submodule": section_stats,
        "section_balance_score": balance_score(section_stats),
    }


@traceable(name="find_duplicate_titles")
def find_duplicate_titles(
    course_state: CourseState,
    similarity_threshold: float = 0.9
) -> Dict[str, Any]:
    """
    Find duplicate or near-duplicate titles using fuzzy matching.
    
    Args:
        course_state: The CourseState to analyze
        similarity_threshold: Threshold for considering titles as duplicates (0-1)
        
    Returns:
        Dictionary with duplicate title information
    """
    from rapidfuzz import fuzz
    
    results = {
        "module_duplicates": [],
        "submodule_duplicates": [],
        "section_duplicates": [],
    }
    
    # Check module titles
    module_titles = [(i, m.title) for i, m in enumerate(course_state.modules)]
    results["module_duplicates"] = _find_fuzzy_duplicates(
        module_titles, similarity_threshold, fuzz
    )
    
    # Check submodule titles
    submodule_titles = []
    for m_idx, module in enumerate(course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            submodule_titles.append((f"{m_idx+1}.{sm_idx+1}", submodule.title))
    results["submodule_duplicates"] = _find_fuzzy_duplicates(
        submodule_titles, similarity_threshold, fuzz
    )
    
    # Check section titles
    section_titles = []
    for m_idx, module in enumerate(course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                section_titles.append((f"{m_idx+1}.{sm_idx+1}.{s_idx+1}", section.title))
    results["section_duplicates"] = _find_fuzzy_duplicates(
        section_titles, similarity_threshold, fuzz
    )
    
    results["total_duplicates"] = (
        len(results["module_duplicates"]) +
        len(results["submodule_duplicates"]) +
        len(results["section_duplicates"])
    )
    
    return results


def _find_fuzzy_duplicates(
    titles: List[Tuple[str, str]],
    threshold: float,
    fuzz
) -> List[Dict[str, Any]]:
    """Find fuzzy duplicate titles."""
    duplicates = []
    n = len(titles)
    
    for i in range(n):
        for j in range(i + 1, n):
            id_1, title_1 = titles[i]
            id_2, title_2 = titles[j]
            
            # Compute similarity ratio (0-100)
            similarity = fuzz.ratio(title_1.lower(), title_2.lower()) / 100.0
            
            if similarity >= threshold:
                duplicates.append({
                    "id_1": id_1,
                    "id_2": id_2,
                    "title_1": title_1,
                    "title_2": title_2,
                    "similarity": round(similarity, 4)
                })
    
    # Sort by similarity (highest first)
    duplicates.sort(key=lambda x: x["similarity"], reverse=True)
    return duplicates[:10]  # Return top 10


def _find_exact_duplicates(course_state: CourseState) -> Dict[str, Any]:
    """Fallback: find only exact duplicate titles."""
    from collections import Counter
    
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
    
    def find_duplicates(titles: List[str]) -> List[str]:
        counts = Counter(titles)
        return [title for title, count in counts.items() if count > 1]
    
    return {
        "module_duplicates": find_duplicates(module_titles),
        "submodule_duplicates": find_duplicates(submodule_titles),
        "section_duplicates": find_duplicates(section_titles),
        "note": "Using exact matching only (rapidfuzz not installed)"
    }


@traceable(name="compute_word_overlap")
def compute_word_overlap(course_state: CourseState) -> Dict[str, Any]:
    """
    Compute word overlap (Jaccard similarity) between titles at each level.
    
    Args:
        course_state: The CourseState to analyze
        
    Returns:
        Dictionary with word overlap metrics
    """
    def jaccard_similarity(set1: set, set2: set) -> float:
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def get_word_set(text: str) -> set:
        # Simple word extraction (lowercase, alphanumeric only)
        words = set(w.lower() for w in text.split() if w.isalnum())
        return words
    
    # Compute average Jaccard similarity for section titles within same submodule
    within_submodule_similarities = []
    
    for module in course_state.modules:
        for submodule in module.submodules:
            titles = [s.title for s in submodule.sections]
            word_sets = [get_word_set(t) for t in titles]
            
            for i in range(len(word_sets)):
                for j in range(i + 1, len(word_sets)):
                    sim = jaccard_similarity(word_sets[i], word_sets[j])
                    within_submodule_similarities.append(sim)
    
    avg_within_similarity = (
        sum(within_submodule_similarities) / len(within_submodule_similarities)
        if within_submodule_similarities else 0.0
    )
    
    return {
        "avg_word_overlap_within_submodules": round(avg_within_similarity, 4),
        "total_comparisons": len(within_submodule_similarities)
    }

