"""Traditional NLP metrics for text quality analysis.

Provides:
- Readability metrics (Coleman-Liau, ARI)
- Repetition metrics (TTR, n-gram repetition)
"""

from typing import Dict, Any, List
from collections import Counter
import re
from langsmith import traceable


@traceable(name="compute_readability")
def compute_readability(text: str) -> Dict[str, Any]:
    """
    Compute readability metrics using textstat library.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with:
        - avg_sentence_length: Average words per sentence
        - word_count: Total word count
        - readability_score: Mean of Coleman-Liau and ARI indices
    """
    import textstat
    from langdetect import detect
    
    # Detect language and set for textstat
    lang = detect(text)
    textstat.set_lang(lang)
    
    coleman_liau = textstat.coleman_liau_index(text)
    ari = textstat.automated_readability_index(text)
    readability_score = (coleman_liau + ari) / 2
    
    return {
        "language_detected": lang,
        "avg_sentence_length": round(textstat.words_per_sentence(text), 2),
        "word_count": textstat.lexicon_count(text),
        "readability_score": round(readability_score, 2),
    }


@traceable(name="compute_repetition_metrics")
def compute_repetition_metrics(text: str, preprocess: bool = True) -> Dict[str, Any]:
    """
    Compute repetition and diversity metrics.
    
    Args:
        text: The text to analyze
        preprocess: Whether to lowercase and clean text
        
    Returns:
        Dictionary with:
        - type_token_ratio: Unique tokens / total tokens
        - 2gram/3gram/4gram_repetition_rate: Proportion of repeated n-grams
        - weighted_ngram_repetition: Weighted sum (1×2gram + 2×3gram + 3×4gram) / 6
    """
    if preprocess:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Záéíóúñüäößàèùçâêîôûœæ\s]', '', text)
    
    # Tokenize
    try:
        import nltk
        tokens = nltk.word_tokenize(text)
    except:
        tokens = text.split()
    
    tokens = [t for t in tokens if t.isalpha()]
    
    if len(tokens) < 10:
        return {"error": "Not enough tokens for analysis"}
    
    # Type-Token Ratio
    ttr = len(set(tokens)) / len(tokens)
    
    # N-gram repetition rates
    rep_2gram = _compute_ngram_repetition(tokens, 2)
    rep_3gram = _compute_ngram_repetition(tokens, 3)
    rep_4gram = _compute_ngram_repetition(tokens, 4)
    
    # Weighted repetition: penalize higher n-grams more
    weighted = (1 * rep_2gram + 2 * rep_3gram + 3 * rep_4gram) / 6
    
    return {
        "type_token_ratio": round(ttr, 4),
        "2gram_repetition_rate": rep_2gram,
        "3gram_repetition_rate": rep_3gram,
        "4gram_repetition_rate": rep_4gram,
        "weighted_ngram_repetition": round(weighted, 4),
    }


def _compute_ngram_repetition(tokens: List[str], n: int) -> float:
    """
    Compute repetition rate for n-grams.
    
    Args:
        tokens: List of tokens
        n: N-gram size
        
    Returns:
        Repetition rate (proportion of tokens in repeated n-grams)
    """
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    if not ngrams:
        return 0.0
    
    counts = Counter(ngrams)
    total = sum(counts.values())
    repeated = sum(c for c in counts.values() if c > 1)
    
    return round(repeated / total, 4) if total > 0 else 0.0
