"""Traditional NLP metrics for text quality analysis.

Based on the evaluation notebooks:
- textstat.ipynb: Readability metrics (Flesch, SMOG, etc.)
- traditional_nlp.ipynb: N-gram repetition, entropy, TTR
"""

from typing import Dict, Any, List
from collections import Counter
import math
import re
from langsmith import traceable


@traceable(name="compute_readability")
def compute_readability(text: str) -> Dict[str, Any]:
    """
    Compute readability metrics using textstat library.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with readability scores
    """
    import textstat
    from langdetect import detect
    
    # Detect language and set for textstat
    lang = detect(text)
    textstat.set_lang(lang)
    
    return {
        "language_detected": lang,
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "avg_sentence_length": textstat.words_per_sentence(text),
        "lexicon_count": textstat.lexicon_count(text),
        "syllable_count": textstat.syllable_count(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
    }


@traceable(name="compute_repetition_metrics")
def compute_repetition_metrics(text: str, preprocess: bool = True) -> Dict[str, Any]:
    """
    Compute repetition and diversity metrics.
    
    Args:
        text: The text to analyze
        preprocess: Whether to lowercase and clean text
        
    Returns:
        Dictionary with repetition metrics
    """
    # Preprocess
    if preprocess:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        # Keep letters and basic punctuation
        text = re.sub(r'[^a-zA-Záéíóúñüäößàèùçâêîôûœæ\s]', '', text)
    
    # Tokenize (simple split, or use nltk if available)
    try:
        import nltk
        tokens = nltk.word_tokenize(text)
    except:
        tokens = text.split()
    
    # Filter to alphabetic tokens only
    tokens = [t for t in tokens if t.isalpha()]
    
    if len(tokens) < 10:
        return {"error": "Not enough tokens for analysis"}
    
    # Type-Token Ratio (TTR)
    ttr = len(set(tokens)) / len(tokens)
    
    # N-gram metrics
    ngram_results = {}
    for n in [2, 3, 4]:
        rep_rate, entropy = _compute_ngram_metrics(tokens, n)
        ngram_results[f"{n}gram_repetition_rate"] = rep_rate
        ngram_results[f"{n}gram_entropy"] = entropy
    
    return {
        "total_tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
        "type_token_ratio": round(ttr, 4),
        **ngram_results
    }


def _compute_ngram_metrics(tokens: List[str], n: int) -> tuple:
    """
    Compute repetition rate and entropy for n-grams.
    
    Args:
        tokens: List of tokens
        n: N-gram size
        
    Returns:
        Tuple of (repetition_rate, entropy)
    """
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    if not ngrams:
        return 0.0, 0.0
    
    counts = Counter(ngrams)
    total = sum(counts.values())
    
    # Repetition rate: proportion of repeated n-grams
    repeated = sum(c for c in counts.values() if c > 1)
    repetition_rate = repeated / total if total > 0 else 0
    
    # Shannon entropy
    entropy = -sum(
        (c / total) * math.log2(c / total) 
        for c in counts.values()
    ) if total > 0 else 0
    
    return round(repetition_rate, 4), round(entropy, 4)


@traceable(name="compute_all_nlp_metrics")
def compute_all_nlp_metrics(text: str) -> Dict[str, Any]:
    """
    Compute all NLP metrics for a text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with all metrics
    """
    return {
        "readability": compute_readability(text),
        "repetition": compute_repetition_metrics(text)
    }

