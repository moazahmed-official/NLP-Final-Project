"""
Main Pipeline Script for NLP Job Description Analysis
======================================================
This script runs the complete NLP pipeline:
1. Data Loading
2. Preprocessing
3. Skill Extraction
4. Summarization
5. Evaluation
6. Visualization
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import TextPreprocessor, preprocess_dataframe
from skill_extractor import extract_skills_from_dataframe, get_skill_statistics
from summarizer import summarize_dataframe
from evaluate import evaluate_extractor_quality, generate_evaluation_report
from viz import create_all_visualizations


def run_pipeline(data_path: str = 'data/jobs.csv',
                 output_dir: str = 'output',
                 sample_size: Optional[int] = None,
                 verbose: bool = True):
    """
    Run the complete NLP pipeline.
    
    Args:
        data_path: Path to the jobs CSV file
        output_dir: Directory for output files
        sample_size: Number of samples to process (None for all)
        verbose: Print progress information
    """
    start_time = datetime.now()
    
    if verbose:
        print("="*60)
        print("NLP JOB DESCRIPTION ANALYSIS PIPELINE")
        print("="*60)
        print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    
    # ==========================================
    # STEP 1: DATA LOADING
    # ==========================================
    if verbose:
        print("\n" + "-"*40)
        print("STEP 1: Loading Data...")
        print("-"*40)
    
    df = pd.read_csv(data_path)
    
    if sample_size:
        df = df.head(sample_size)
    
    if verbose:
        print(f"Loaded {len(df):,} job postings")
        print(f"Columns: {df.columns.tolist()}")
    
    # ==========================================
    # STEP 2: PREPROCESSING
    # ==========================================
    if verbose:
        print("\n" + "-"*40)
        print("STEP 2: Preprocessing Text...")
        print("-"*40)
    
    preprocessor = TextPreprocessor()
    df = preprocess_dataframe(df, preprocessor=preprocessor)
    
    if verbose:
        print("Preprocessing complete!")
        print(f"Sample cleaned text: {df['cleaned_text'].iloc[0][:100]}...")
    
    # ==========================================
    # STEP 3: SKILL EXTRACTION
    # ==========================================
    if verbose:
        print("\n" + "-"*40)
        print("STEP 3: Extracting Skills...")
        print("-"*40)
    
    skills_df = extract_skills_from_dataframe(df, method='hybrid')
    
    # Save extracted skills
    skills_output = skills_df[['id', 'Job Title', 'Extracted Skills']]
    skills_output.to_csv(f'{output_dir}/extracted_skills.csv', index=False)
    
    if verbose:
        stats = get_skill_statistics(skills_df)
        print(f"Skills extracted successfully!")
        print(f"Total unique skills: {stats['total_unique_skills']}")
        print(f"Saved to: {output_dir}/extracted_skills.csv")
    
    # ==========================================
    # STEP 4: SUMMARIZATION
    # ==========================================
    if verbose:
        print("\n" + "-"*40)
        print("STEP 4: Generating Summaries...")
        print("-"*40)
    
    # Use TextRank for speed
    summary_df = summarize_dataframe(df, method='textrank', num_sentences=2)
    
    # Save summaries
    summary_output = summary_df[['id', 'Job Title', 'Summary']]
    summary_output.to_csv(f'{output_dir}/job_summary.csv', index=False)
    
    if verbose:
        print(f"Summaries generated successfully!")
        print(f"Saved to: {output_dir}/job_summary.csv")
    
    # ==========================================
    # STEP 5: EVALUATION
    # ==========================================
    if verbose:
        print("\n" + "-"*40)
        print("STEP 5: Evaluating Results...")
        print("-"*40)
    
    # Merge for evaluation
    eval_df = skills_df.copy()
    metrics = evaluate_extractor_quality(eval_df)
    
    if verbose:
        print(f"\nEvaluation Metrics:")
        print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"  Macro F1-Score: {metrics['macro_f1']:.4f}")
    
    # Save evaluation report
    report = generate_evaluation_report(metrics)
    with open(f'{output_dir}/evaluation_report.txt', 'w') as f:
        f.write(report)
    
    # ==========================================
    # STEP 6: VISUALIZATION
    # ==========================================
    if verbose:
        print("\n" + "-"*40)
        print("STEP 6: Creating Visualizations...")
        print("-"*40)
    
    try:
        create_all_visualizations(df, skills_df, f'{output_dir}/figures')
        if verbose:
            print(f"Visualizations saved to: {output_dir}/figures/")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not create some visualizations: {e}")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    if verbose:
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"\nTotal time: {duration}")
        print(f"\nOutput files:")
        print(f"  - {output_dir}/extracted_skills.csv")
        print(f"  - {output_dir}/job_summary.csv")
        print(f"  - {output_dir}/evaluation_report.txt")
        print(f"  - {output_dir}/figures/")
    
    return {
        'skills_df': skills_df,
        'summary_df': summary_df,
        'metrics': metrics,
        'duration': duration
    }


if __name__ == "__main__":
    # Run with default settings
    # Use sample_size=1000 for quick testing, None for full dataset
    results = run_pipeline(
        data_path='data/jobs.csv',
        output_dir='output',
        sample_size=1000,  # Set to None for full dataset
        verbose=True
    )
