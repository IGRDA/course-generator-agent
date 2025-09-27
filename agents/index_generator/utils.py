import math


def compute_layout(total_pages: int) -> tuple[int, int, int]:
    if total_pages <= 0:
        raise ValueError("total_pages must be a positive integer")

    target_pairs = max(1, math.ceil(total_pages / 3))  # modules * submodules

    best = None
    a = int(math.sqrt(target_pairs))
    for m in range(max(1, a - 2), a + 3):
        s = math.ceil(target_pairs / m)
        prod = m * s
        if best is None or prod < best[2]:
            best = (m, s, prod)

    n_modules, n_submodules, _ = best
    n_sections = math.ceil(total_pages / (n_modules * n_submodules))
    return n_modules, n_submodules, n_sections