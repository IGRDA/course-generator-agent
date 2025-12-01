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
    similarity_threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Compute pairwise semantic similarity between sections.
    
    Args:
        sections: List of dicts with 'id', 'title', and 'text' keys
        similarity_threshold: Threshold above which sections are flagged as similar
        model_name: Sentence transformer model to use
        
    Returns:
        Dictionary with similarity metrics and flagged pairs
    """
    from sentence_transformers import SentenceTransformer
    
    if len(sections) < 2:
        return {"error": "Need at least 2 sections for similarity analysis"}
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Prepare texts (combine title and content for better semantic representation)
    texts = [f"{s['title']}: {s['text'][:1000]}" for s in sections]
    
    # Compute embeddings
    embeddings = model.encode(texts, show_progress_bar=False)
    
    # Compute pairwise cosine similarity
    similarities = _compute_cosine_similarity_matrix(embeddings)
    
    # Find high similarity pairs (excluding self-similarity)
    high_similarity_pairs = []
    n = len(sections)
    all_similarities = []
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarities[i][j]
            all_similarities.append(sim)
            
            if sim >= similarity_threshold:
                high_similarity_pairs.append({
                    "section_1": sections[i]["id"],
                    "section_2": sections[j]["id"],
                    "title_1": sections[i]["title"],
                    "title_2": sections[j]["title"],
                    "similarity": round(float(sim), 4)
                })
    
    # Sort by similarity (highest first)
    high_similarity_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "total_sections": len(sections),
        "pairs_analyzed": len(all_similarities),
        "max_similarity": round(float(max(all_similarities)), 4) if all_similarities else 0,
        "avg_similarity": round(float(np.mean(all_similarities)), 4) if all_similarities else 0,
        "min_similarity": round(float(min(all_similarities)), 4) if all_similarities else 0,
        "std_similarity": round(float(np.std(all_similarities)), 4) if all_similarities else 0,
        "high_similarity_pairs": high_similarity_pairs[:10],  # Top 10 most similar
        "num_flagged_pairs": len(high_similarity_pairs),
        "similarity_threshold": similarity_threshold
    }


def _compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix for embeddings.
    
    Args:
        embeddings: 2D numpy array of embeddings (n_samples, embedding_dim)
        
    Returns:
        2D numpy array of pairwise cosine similarities
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix


@traceable(name="compute_title_embeddings_similarity")
def compute_title_embeddings_similarity(
    titles: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Compute semantic similarity specifically for titles.
    
    Args:
        titles: List of title strings
        model_name: Sentence transformer model to use
        
    Returns:
        Dictionary with title similarity metrics
    """
    from sentence_transformers import SentenceTransformer
    
    if len(titles) < 2:
        return {"error": "Need at least 2 titles for similarity analysis"}
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(titles, show_progress_bar=False)
    similarities = _compute_cosine_similarity_matrix(embeddings)
    
    # Find similar title pairs
    similar_pairs = []
    n = len(titles)
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarities[i][j]
            if sim >= 0.8:  # High threshold for titles
                similar_pairs.append({
                    "title_1": titles[i],
                    "title_2": titles[j],
                    "similarity": round(float(sim), 4)
                })
    
    similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "total_titles": len(titles),
        "similar_title_pairs": similar_pairs[:10],
        "num_similar_pairs": len(similar_pairs)
    }

