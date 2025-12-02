"""Embedding-based metrics for content similarity detection.

Uses sentence-transformers to compute semantic similarity between sections
and detect potentially repetitive content.
"""

from typing import Dict, Any, List
import numpy as np
from langsmith import traceable


@traceable(name="compute_section_similarity")
def compute_section_similarity(
    sections: List[Dict[str, str]],
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Compute pairwise semantic similarity between sections.
    
    Args:
        sections: List of dicts with 'id', 'title', and 'text' keys
        model_name: Sentence transformer model to use
        
    Returns:
        Dictionary with similarity metrics
    """
    from sentence_transformers import SentenceTransformer
    
    if len(sections) < 2:
        return {"error": "Need at least 2 sections for similarity analysis"}
    
    model = SentenceTransformer(model_name)
    texts = [f"{s['title']}: {s['text'][:1000]}" for s in sections]
    embeddings = model.encode(texts, show_progress_bar=False)
    similarities = _compute_cosine_similarity_matrix(embeddings)
    
    # Collect pairwise similarities
    all_similarities = []
    n = len(sections)
    
    for i in range(n):
        for j in range(i + 1, n):
            all_similarities.append(similarities[i][j])
    
    return {
        "max_similarity": round(float(max(all_similarities)), 4) if all_similarities else 0,
        "avg_similarity": round(float(np.mean(all_similarities)), 4) if all_similarities else 0,
        "min_similarity": round(float(min(all_similarities)), 4) if all_similarities else 0,
        "std_similarity": round(float(np.std(all_similarities)), 4) if all_similarities else 0,
    }


@traceable(name="compute_title_embedding_similarity")
def compute_title_embedding_similarity(
    module_titles: List[str],
    submodule_titles: List[str],
    section_titles: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Compute embedding-based uniqueness for titles at each hierarchy level.
    
    Args:
        module_titles: List of module titles
        submodule_titles: List of submodule titles
        section_titles: List of section titles
        model_name: Sentence transformer model to use
        
    Returns:
        Dictionary with embedding uniqueness scores (1 - avg_similarity)
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    
    def compute_uniqueness(titles: List[str]) -> float:
        if len(titles) < 2:
            return 1.0
        
        embeddings = model.encode(titles, show_progress_bar=False)
        similarities = _compute_cosine_similarity_matrix(embeddings)
        
        # Get upper triangle (excluding diagonal)
        n = len(titles)
        pair_sims = [similarities[i][j] for i in range(n) for j in range(i + 1, n)]
        
        if not pair_sims:
            return 1.0
        
        avg_sim = float(np.mean(pair_sims))
        return round(1 - avg_sim, 4)
    
    return {
        "module_embedding_uniqueness": compute_uniqueness(module_titles),
        "submodule_embedding_uniqueness": compute_uniqueness(submodule_titles),
        "section_embedding_uniqueness": compute_uniqueness(section_titles),
    }


def _compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix for embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)

