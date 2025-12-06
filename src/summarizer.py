"""
Text Summarization Module for Job Description Analysis
=======================================================
This module implements two summarization approaches:
1. TextRank: Extractive summarization using graph-based ranking
2. Transformer-based: Abstractive summarization using pre-trained models
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextRankSummarizer:
    """
    TextRank-based extractive summarization.
    Uses a graph-based ranking algorithm to identify important sentences.
    """
    
    def __init__(self, damping: float = 0.85, min_diff: float = 1e-5, 
                 max_iterations: int = 100):
        """
        Initialize TextRank summarizer.
        
        Args:
            damping: Damping factor for PageRank algorithm
            min_diff: Minimum difference for convergence
            max_iterations: Maximum iterations for convergence
        """
        self.damping = damping
        self.min_diff = min_diff
        self.max_iterations = max_iterations
        self.stop_words = set(stopwords.words('english'))
    
    def _sentence_similarity(self, sent1: List[str], sent2: List[str]) -> float:
        """
        Calculate similarity between two sentences using word overlap.
        
        Args:
            sent1: First sentence as list of words
            sent2: Second sentence as list of words
            
        Returns:
            Similarity score
        """
        # Remove stopwords
        sent1 = [w.lower() for w in sent1 if w.lower() not in self.stop_words]
        sent2 = [w.lower() for w in sent2 if w.lower() not in self.stop_words]
        
        all_words = list(set(sent1 + sent2))
        
        if len(all_words) == 0:
            return 0
        
        vector1 = [1 if w in sent1 else 0 for w in all_words]
        vector2 = [1 if w in sent2 else 0 for w in all_words]
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = sum(a * a for a in vector1) ** 0.5
        magnitude2 = sum(b * b for b in vector2) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _build_similarity_matrix(self, sentences: List[List[str]]) -> np.ndarray:
        """
        Build similarity matrix for sentences.
        
        Args:
            sentences: List of tokenized sentences
            
        Returns:
            Similarity matrix
        """
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(
                        sentences[i], sentences[j]
                    )
        
        # Normalize
        for i in range(n):
            row_sum = similarity_matrix[i].sum()
            if row_sum > 0:
                similarity_matrix[i] = similarity_matrix[i] / row_sum
        
        return similarity_matrix
    
    def _textrank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Apply TextRank algorithm.
        
        Args:
            similarity_matrix: Sentence similarity matrix
            
        Returns:
            Array of sentence scores
        """
        n = len(similarity_matrix)
        scores = np.ones(n) / n
        
        for _ in range(self.max_iterations):
            prev_scores = scores.copy()
            
            for i in range(n):
                scores[i] = (1 - self.damping) + self.damping * sum(
                    similarity_matrix[j][i] * prev_scores[j]
                    for j in range(n)
                )
            
            if np.abs(scores - prev_scores).sum() < self.min_diff:
                break
        
        return scores
    
    def summarize(self, text: str, num_sentences: int = 3, 
                  min_length: int = 30) -> str:
        """
        Generate summary using TextRank.
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
            min_length: Minimum text length to summarize
            
        Returns:
            Summary text
        """
        if not text or len(text) < min_length:
            return text
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Tokenize sentences into words
        tokenized_sentences = [word_tokenize(sent) for sent in sentences]
        
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(tokenized_sentences)
        
        # Apply TextRank
        scores = self._textrank(similarity_matrix)
        
        # Get top sentences
        ranked_indices = scores.argsort()[-num_sentences:][::-1]
        ranked_indices = sorted(ranked_indices)  # Keep original order
        
        summary = ' '.join([sentences[i] for i in ranked_indices])
        
        return summary


class TransformerSummarizer:
    """
    Transformer-based abstractive summarization using HuggingFace models.
    Supports: t5-small, bart-large-cnn, distilbart-cnn
    """
    
    def __init__(self, model_name: str = 't5-small'):
        """
        Initialize transformer summarizer.
        
        Args:
            model_name: Name of the pre-trained model
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        try:
            from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
            
            print(f"Loading {self.model_name} model...")
            
            if 't5' in self.model_name.lower():
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    tokenizer=self.model_name
                )
            elif 'bart' in self.model_name.lower():
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name
                )
            else:
                # Default to t5-small
                self.summarizer = pipeline(
                    "summarization",
                    model="t5-small"
                )
            
            print(f"Model {self.model_name} loaded successfully!")
            
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            print("Falling back to TextRank summarization")
            self.summarizer = None
    
    def summarize(self, text: str, max_length: int = 150, 
                  min_length: int = 30) -> str:
        """
        Generate summary using transformer model.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Summary text
        """
        if self.summarizer is None:
            # Fallback to TextRank
            fallback = TextRankSummarizer()
            return fallback.summarize(text)
        
        if not text or len(text.split()) < 10:
            return text
        
        try:
            # Truncate input if too long (transformer limitation)
            words = text.split()
            if len(words) > 500:
                text = ' '.join(words[:500])
            
            # Add prefix for T5
            if 't5' in self.model_name.lower():
                text = "summarize: " + text
            
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            print(f"Error in summarization: {e}")
            # Fallback to TextRank
            fallback = TextRankSummarizer()
            return fallback.summarize(text)


class HybridSummarizer:
    """
    Hybrid summarizer combining TextRank and Transformer approaches.
    """
    
    def __init__(self, transformer_model: str = 't5-small'):
        """
        Initialize hybrid summarizer.
        
        Args:
            transformer_model: Name of transformer model to use
        """
        self.textrank = TextRankSummarizer()
        self.transformer = None
        self.transformer_model = transformer_model
    
    def _init_transformer(self):
        """Lazy initialization of transformer model."""
        if self.transformer is None:
            self.transformer = TransformerSummarizer(self.transformer_model)
    
    def summarize(self, text: str, method: str = 'textrank', 
                  num_sentences: int = 3, max_length: int = 150) -> str:
        """
        Generate summary using specified method.
        
        Args:
            text: Input text
            method: 'textrank', 'transformer', or 'both'
            num_sentences: Number of sentences for TextRank
            max_length: Max length for transformer
            
        Returns:
            Summary text
        """
        if method == 'textrank':
            return self.textrank.summarize(text, num_sentences)
        
        elif method == 'transformer':
            self._init_transformer()
            return self.transformer.summarize(text, max_length)
        
        else:  # both - return TextRank by default, use transformer if needed
            textrank_summary = self.textrank.summarize(text, num_sentences)
            return textrank_summary


def summarize_dataframe(df: pd.DataFrame,
                        text_column: str = 'Job Description',
                        method: str = 'textrank',
                        num_sentences: int = 3) -> pd.DataFrame:
    """
    Summarize job descriptions in a DataFrame.
    
    Args:
        df: Input DataFrame
        text_column: Column containing text to summarize
        method: Summarization method
        num_sentences: Number of sentences for extractive summary
        
    Returns:
        DataFrame with summaries
    """
    summarizer = HybridSummarizer()
    
    df = df.copy()
    
    print(f"Summarizing {len(df)} job descriptions using {method}...")
    
    summaries = []
    for idx, row in df.iterrows():
        text = str(row[text_column])
        summary = summarizer.summarize(text, method, num_sentences)
        summaries.append(summary)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} descriptions")
    
    df['Summary'] = summaries
    
    return df


def compare_summaries(text: str) -> dict:
    """
    Compare summaries from different methods.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with summaries from each method
    """
    summarizer = HybridSummarizer()
    
    return {
        'original': text,
        'textrank': summarizer.summarize(text, method='textrank'),
        'original_length': len(text.split()),
        'textrank_length': len(summarizer.summarize(text, method='textrank').split())
    }


# Main execution
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('../data/jobs.csv')
    
    # Summarize using TextRank (fast)
    print("Generating summaries using TextRank...")
    result_df = summarize_dataframe(df, method='textrank', num_sentences=3)
    
    # Save results
    output_df = result_df[['id', 'Job Title', 'Summary']]
    output_df.to_csv('../output/job_summary.csv', index=False)
    
    print(f"\nSummaries saved to ../output/job_summary.csv")
    
    # Show sample
    print("\n" + "="*50)
    print("Sample Summary:")
    print("="*50)
    sample = result_df.iloc[0]
    print(f"\nJob Title: {sample['Job Title']}")
    print(f"\nOriginal Description:\n{sample['Job Description'][:500]}...")
    print(f"\nSummary:\n{sample['Summary']}")
