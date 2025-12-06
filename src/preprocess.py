"""
Preprocessing Pipeline for NLP Skill Extraction Project
========================================================
This module handles text preprocessing for job descriptions including:
- Lowercasing
- Punctuation removal
- Number removal
- Stopword removal
- Tokenization
- Lemmatization
- POS tagging (optional)
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

download_nltk_data()


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for NLP tasks.
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = True,
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 custom_stopwords: list = None):
        """
        Initialize the preprocessor with configurable options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric characters
            remove_stopwords: Remove common English stopwords
            lemmatize: Apply lemmatization
            custom_stopwords: Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Apply basic cleaning to text.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> list:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords_from_tokens(self, tokens: list) -> list:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered token list
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: list) -> list:
        """
        Apply lemmatization to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized token list
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def get_pos_tags(self, tokens: list) -> list:
        """
        Get POS tags for tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of (token, POS tag) tuples
        """
        return pos_tag(tokens)
    
    def preprocess(self, text: str, return_tokens: bool = False, include_pos: bool = False):
        """
        Full preprocessing pipeline.
        
        Args:
            text: Input text string
            return_tokens: If True, return tokens; otherwise return joined string
            include_pos: If True, include POS tags in output
            
        Returns:
            Preprocessed text (string or tokens with optional POS tags)
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Filter out very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        if include_pos:
            return self.get_pos_tags(tokens)
        
        if return_tokens:
            return tokens
        
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: list, return_tokens: bool = False) -> list:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of text strings
            return_tokens: If True, return tokens for each text
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, return_tokens) for text in texts]


def preprocess_dataframe(df, text_column: str = 'Job Description', 
                         output_column: str = 'cleaned_text',
                         preprocessor: TextPreprocessor = None):
    """
    Preprocess a DataFrame column containing text.
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing text to preprocess
        output_column: Name of output column for cleaned text
        preprocessor: TextPreprocessor instance (creates default if None)
        
    Returns:
        DataFrame with new preprocessed column
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    df = df.copy()
    df[output_column] = df[text_column].apply(preprocessor.preprocess)
    
    return df


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Load data
    df = pd.read_csv('../data/jobs.csv')
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_stopwords=True,
        lemmatize=True
    )
    
    # Preprocess
    df = preprocess_dataframe(df, preprocessor=preprocessor)
    
    # Display sample
    print("Sample preprocessed data:")
    print(df[['Job Title', 'cleaned_text']].head())
