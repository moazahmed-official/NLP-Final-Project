"""
NLP Job Description Analysis - Source Module
============================================
This package contains modules for NLP-based job description analysis.
"""

from .preprocess import TextPreprocessor, preprocess_dataframe
from .skill_extractor import (
    RuleBasedExtractor,
    MLBasedExtractor,
    HybridSkillExtractor,
    extract_skills_from_dataframe,
    PREDEFINED_SKILLS
)
from .summarizer import TextRankSummarizer, TransformerSummarizer, HybridSummarizer
from .evaluate import evaluate_extraction, evaluate_batch, generate_evaluation_report
from .viz import (
    plot_job_title_distribution,
    create_wordcloud,
    plot_skill_frequency,
    create_all_visualizations
)

__version__ = '1.0.0'
__author__ = 'NLP Project'
