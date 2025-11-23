import math


def get_module_count(total_pages: int) -> int:
    """Determine number of modules based on course size"""
    if total_pages <= 50:  # small courses
        return min(8, max(5, total_pages // 10))
    elif total_pages <= 500:  # medium courses
        return min(12, max(8, 8 + (total_pages - 50) // 50))
    else:  # large courses (>500)
        return min(20, max(12, 12 + (total_pages - 500) // 100))


def compute_layout(total_pages: int, target_pages_per_section: int = 4) -> tuple[int, int, int]:
    """
    Compute course layout: (n_modules, n_submodules, n_sections)
    Target: ~4 pages per section with balanced distribution
    """
    if total_pages <= 0:
        raise ValueError("total_pages must be a positive integer")
    
    n_modules = get_module_count(total_pages)
    target_sections = max(1, total_pages // target_pages_per_section)
    sections_per_module = max(1, target_sections // n_modules)
    
    # Distribute into submodules (2-6 submodules per module)
    if sections_per_module <= 3:
        n_submodules = 1
        n_sections = sections_per_module
    elif sections_per_module <= 8:
        n_submodules = 2
        n_sections = max(1, sections_per_module // 2)
    elif sections_per_module <= 18:
        n_submodules = 3
        n_sections = max(1, sections_per_module // 3)
    else:
        n_submodules = min(6, max(3, int(math.sqrt(sections_per_module))))
        n_sections = max(1, sections_per_module // n_submodules)
    
    return n_modules, n_submodules, n_sections