"""
Skill Extraction Module for NLP Job Description Analysis
=========================================================
This module implements two approaches for skill extraction:
1. Rule-Based Extraction: Using predefined skill lists and keyword matching
2. ML-Based Extraction: Using TF-IDF and spaCy NER for candidate extraction
"""

import re
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Set, Dict, Tuple

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')


# Comprehensive predefined skills list
PREDEFINED_SKILLS = {
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'golang',
    'rust', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'dart', 'lua',
    'objective-c', 'shell', 'bash', 'powershell', 'sql', 'plsql', 'tsql',
    
    # Web Development
    'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind', 'jquery', 'ajax',
    'react', 'reactjs', 'react.js', 'angular', 'angularjs', 'vue', 'vuejs', 'vue.js',
    'svelte', 'next.js', 'nextjs', 'nuxt', 'gatsby', 'webpack', 'vite', 'babel',
    'nodejs', 'node.js', 'express', 'expressjs', 'fastapi', 'flask', 'django',
    'spring', 'spring boot', 'asp.net', '.net', 'laravel', 'rails', 'ruby on rails',
    
    # Mobile Development
    'flutter', 'react native', 'xamarin', 'ionic', 'android', 'ios', 'swiftui',
    
    # Databases
    'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch',
    'oracle', 'sql server', 'sqlite', 'cassandra', 'dynamodb', 'firebase',
    'neo4j', 'mariadb', 'couchdb',
    
    # Cloud & DevOps
    'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'docker',
    'kubernetes', 'k8s', 'jenkins', 'gitlab ci', 'github actions', 'circleci',
    'terraform', 'ansible', 'puppet', 'chef', 'vagrant', 'nginx', 'apache',
    'linux', 'unix', 'ci/cd', 'devops', 'microservices',
    
    # Data Science & ML
    'machine learning', 'ml', 'deep learning', 'dl', 'artificial intelligence', 'ai',
    'neural networks', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
    'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'tableau',
    'power bi', 'data analysis', 'data visualization', 'statistics', 'nlp',
    'natural language processing', 'computer vision', 'opencv', 'bert', 'gpt',
    'transformers', 'huggingface', 'spacy', 'nltk',
    
    # Big Data
    'hadoop', 'spark', 'apache spark', 'pyspark', 'hive', 'kafka', 'airflow',
    'data engineering', 'etl', 'data pipeline', 'data warehouse',
    
    # APIs & Protocols
    'rest', 'restful', 'rest api', 'graphql', 'grpc', 'soap', 'websocket',
    'api', 'apis', 'json', 'xml', 'oauth', 'jwt',
    
    # Testing
    'unit testing', 'integration testing', 'selenium', 'cypress', 'jest',
    'pytest', 'junit', 'mocha', 'chai', 'tdd', 'bdd', 'qa', 'automation testing',
    
    # Version Control
    'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial',
    
    # Project Management & Methodologies
    'agile', 'scrum', 'kanban', 'jira', 'confluence', 'trello', 'asana',
    
    # Soft Skills (technical context)
    'problem solving', 'communication', 'teamwork', 'leadership', 'analytical',
    
    # Other Technologies
    'blockchain', 'solidity', 'ethereum', 'web3', 'cybersecurity', 'security',
    'networking', 'tcp/ip', 'http', 'https', 'ssl', 'oop', 'design patterns',
    'solid', 'clean code', 'architecture', 'system design', 'uml',
    'figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator',
    'sap', 'salesforce', 'crm', 'erp', 'sharepoint'
}

# Skill patterns for regex matching
SKILL_PATTERNS = [
    r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|golang?|rust|php|swift|kotlin)\b',
    r'\b(react\.?js?|angular\.?js?|vue\.?js?|node\.?js?|next\.?js?)\b',
    r'\b(django|flask|fastapi|express\.?js?|spring\s?boot?)\b',
    r'\b(aws|azure|gcp|docker|kubernetes|k8s)\b',
    r'\b(mysql|postgresql|mongodb|redis|elasticsearch)\b',
    r'\b(machine\s?learning|deep\s?learning|artificial\s?intelligence|nlp)\b',
    r'\b(tensorflow|pytorch|keras|scikit-learn|pandas|numpy)\b',
    r'\b(git|github|gitlab|ci/?cd|devops)\b',
    r'\b(rest\s?api|graphql|microservices)\b',
    r'\b(flutter|react\s?native|android|ios)\b',
    r'\b(agile|scrum|kanban|jira)\b',
    r'\b(sql|nosql|api|apis|html|css|sass|bootstrap)\b'
]


class RuleBasedExtractor:
    """
    Rule-based skill extractor using predefined skill lists and patterns.
    """
    
    def __init__(self, skills: Set[str] = None, patterns: List[str] = None):
        """
        Initialize with skill list and regex patterns.
        
        Args:
            skills: Set of predefined skills
            patterns: List of regex patterns for skill matching
        """
        self.skills = skills or PREDEFINED_SKILLS
        self.patterns = patterns or SKILL_PATTERNS
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]
    
    def extract_by_keyword(self, text: str) -> Set[str]:
        """
        Extract skills using keyword matching.
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted skills
        """
        text_lower = text.lower()
        found_skills = set()
        
        for skill in self.skills:
            # Use word boundary matching for single words
            if ' ' in skill:
                if skill in text_lower:
                    found_skills.add(skill)
            else:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.add(skill)
        
        return found_skills
    
    def extract_by_pattern(self, text: str) -> Set[str]:
        """
        Extract skills using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted skills
        """
        found_skills = set()
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            found_skills.update([m.lower() for m in matches])
        
        return found_skills
    
    def extract(self, text: str) -> List[str]:
        """
        Extract all skills from text.
        
        Args:
            text: Input text
            
        Returns:
            Sorted list of unique skills
        """
        keyword_skills = self.extract_by_keyword(text)
        pattern_skills = self.extract_by_pattern(text)
        
        all_skills = keyword_skills.union(pattern_skills)
        return sorted(list(all_skills))


class MLBasedExtractor:
    """
    ML-based skill extractor using TF-IDF and spaCy NER.
    """
    
    def __init__(self, known_skills: Set[str] = None):
        """
        Initialize the ML-based extractor.
        
        Args:
            known_skills: Set of known skills for mapping
        """
        self.known_skills = known_skills or PREDEFINED_SKILLS
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words='english'
        )
        self.nlp = nlp
    
    def extract_tfidf_candidates(self, texts: List[str], top_n: int = 50) -> List[str]:
        """
        Extract skill candidates using TF-IDF.
        
        Args:
            texts: List of text documents
            top_n: Number of top terms to consider
            
        Returns:
            List of candidate skill terms
        """
        # Fit TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        
        # Get top terms
        top_indices = avg_scores.argsort()[-top_n:][::-1]
        candidates = [feature_names[i] for i in top_indices]
        
        return candidates
    
    def extract_ner_candidates(self, text: str) -> Set[str]:
        """
        Extract skill candidates using spaCy NER.
        
        Args:
            text: Input text
            
        Returns:
            Set of NER-based candidates
        """
        doc = self.nlp(text)
        candidates = set()
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART']:
                candidates.add(ent.text.lower())
        
        # Extract noun chunks as potential skills
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                candidates.add(chunk.text.lower())
        
        return candidates
    
    def map_to_known_skills(self, candidates: Set[str]) -> Set[str]:
        """
        Map candidates to known skills.
        
        Args:
            candidates: Set of candidate terms
            
        Returns:
            Set of matched known skills
        """
        matched_skills = set()
        
        for candidate in candidates:
            candidate_lower = candidate.lower().strip()
            
            # Direct match
            if candidate_lower in self.known_skills:
                matched_skills.add(candidate_lower)
                continue
            
            # Partial match
            for skill in self.known_skills:
                if skill in candidate_lower or candidate_lower in skill:
                    matched_skills.add(skill)
        
        return matched_skills
    
    def extract(self, text: str) -> List[str]:
        """
        Extract skills from text using ML approaches.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted skills
        """
        # Get NER candidates
        ner_candidates = self.extract_ner_candidates(text)
        
        # Map to known skills
        matched_skills = self.map_to_known_skills(ner_candidates)
        
        # Also do direct matching
        text_lower = text.lower()
        for skill in self.known_skills:
            if skill in text_lower:
                matched_skills.add(skill)
        
        return sorted(list(matched_skills))


class HybridSkillExtractor:
    """
    Hybrid extractor combining rule-based and ML-based approaches.
    """
    
    def __init__(self):
        """Initialize both extractors."""
        self.rule_extractor = RuleBasedExtractor()
        self.ml_extractor = MLBasedExtractor()
    
    def extract(self, text: str, method: str = 'hybrid') -> List[str]:
        """
        Extract skills using specified method.
        
        Args:
            text: Input text
            method: 'rule', 'ml', or 'hybrid'
            
        Returns:
            List of extracted skills
        """
        if method == 'rule':
            return self.rule_extractor.extract(text)
        elif method == 'ml':
            return self.ml_extractor.extract(text)
        else:  # hybrid
            rule_skills = set(self.rule_extractor.extract(text))
            ml_skills = set(self.ml_extractor.extract(text))
            return sorted(list(rule_skills.union(ml_skills)))


def extract_skills_from_dataframe(df: pd.DataFrame, 
                                   text_column: str = 'Job Description',
                                   method: str = 'hybrid') -> pd.DataFrame:
    """
    Extract skills from a DataFrame column.
    
    Args:
        df: Input DataFrame
        text_column: Column containing job descriptions
        method: Extraction method ('rule', 'ml', 'hybrid')
        
    Returns:
        DataFrame with extracted skills
    """
    extractor = HybridSkillExtractor()
    
    df = df.copy()
    df['Extracted Skills'] = df[text_column].apply(
        lambda x: extractor.extract(str(x), method)
    )
    df['Skills Count'] = df['Extracted Skills'].apply(len)
    df['Extracted Skills'] = df['Extracted Skills'].apply(lambda x: ', '.join(x))
    
    return df


def get_skill_statistics(df: pd.DataFrame, 
                         skills_column: str = 'Extracted Skills') -> Dict:
    """
    Get statistics about extracted skills.
    
    Args:
        df: DataFrame with extracted skills
        skills_column: Column containing skills
        
    Returns:
        Dictionary with skill statistics
    """
    all_skills = []
    for skills in df[skills_column]:
        if isinstance(skills, str):
            all_skills.extend([s.strip() for s in skills.split(',')])
        elif isinstance(skills, list):
            all_skills.extend(skills)
    
    skill_counts = Counter(all_skills)
    
    return {
        'total_unique_skills': len(skill_counts),
        'total_skill_mentions': sum(skill_counts.values()),
        'top_10_skills': skill_counts.most_common(10),
        'skill_frequency': dict(skill_counts)
    }


# Main execution
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('../data/jobs.csv')
    
    # Extract skills
    print("Extracting skills from job descriptions...")
    result_df = extract_skills_from_dataframe(df, method='hybrid')
    
    # Save results
    output_df = result_df[['id', 'Job Title', 'Extracted Skills']]
    output_df.to_csv('../output/extracted_skills.csv', index=False)
    
    print(f"Extracted skills saved to ../output/extracted_skills.csv")
    
    # Print statistics
    stats = get_skill_statistics(result_df)
    print(f"\nTotal unique skills found: {stats['total_unique_skills']}")
    print(f"Total skill mentions: {stats['total_skill_mentions']}")
    print("\nTop 10 skills:")
    for skill, count in stats['top_10_skills']:
        print(f"  {skill}: {count}")
