"""
Evaluation Module for Skill Extraction
=======================================
This module evaluates the skill extraction accuracy using:
- Precision
- Recall
- F1-Score
- Overlap analysis
"""

import pandas as pd
import numpy as np
from typing import List, Set, Dict, Tuple
from collections import Counter


def calculate_precision(extracted: Set[str], ground_truth: Set[str]) -> float:
    """
    Calculate precision: ratio of correctly extracted skills.
    
    Precision = True Positives / (True Positives + False Positives)
    
    Args:
        extracted: Set of extracted skills
        ground_truth: Set of actual/known skills
        
    Returns:
        Precision score (0.0 to 1.0)
    """
    if len(extracted) == 0:
        return 0.0
    
    true_positives = len(extracted.intersection(ground_truth))
    return true_positives / len(extracted)


def calculate_recall(extracted: Set[str], ground_truth: Set[str]) -> float:
    """
    Calculate recall: ratio of actual skills that were extracted.
    
    Recall = True Positives / (True Positives + False Negatives)
    
    Args:
        extracted: Set of extracted skills
        ground_truth: Set of actual/known skills
        
    Returns:
        Recall score (0.0 to 1.0)
    """
    if len(ground_truth) == 0:
        return 0.0
    
    true_positives = len(extracted.intersection(ground_truth))
    return true_positives / len(ground_truth)


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score: harmonic mean of precision and recall.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def evaluate_extraction(extracted: Set[str], ground_truth: Set[str]) -> Dict[str, float]:
    """
    Comprehensive evaluation of skill extraction.
    
    Args:
        extracted: Set of extracted skills
        ground_truth: Set of actual/known skills
        
    Returns:
        Dictionary with precision, recall, F1, and overlap metrics
    """
    precision = calculate_precision(extracted, ground_truth)
    recall = calculate_recall(extracted, ground_truth)
    f1 = calculate_f1_score(precision, recall)
    
    # Additional metrics
    overlap = len(extracted.intersection(ground_truth))
    false_positives = len(extracted - ground_truth)
    false_negatives = len(ground_truth - extracted)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'overlap_count': overlap,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'extracted_count': len(extracted),
        'ground_truth_count': len(ground_truth)
    }


def evaluate_batch(extracted_list: List[Set[str]], 
                   ground_truth_list: List[Set[str]]) -> Dict[str, float]:
    """
    Evaluate extraction across multiple samples.
    
    Args:
        extracted_list: List of extracted skill sets
        ground_truth_list: List of ground truth skill sets
        
    Returns:
        Aggregated metrics (micro and macro averages)
    """
    assert len(extracted_list) == len(ground_truth_list), \
        "Extracted and ground truth lists must have same length"
    
    # Individual scores
    precisions = []
    recalls = []
    f1_scores = []
    
    # Micro-average accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for extracted, ground_truth in zip(extracted_list, ground_truth_list):
        metrics = evaluate_extraction(extracted, ground_truth)
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
        
        tp = metrics['overlap_count']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Macro averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)
    
    # Micro averages
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = calculate_f1_score(micro_precision, micro_recall)
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'total_samples': len(extracted_list),
        'std_precision': np.std(precisions),
        'std_recall': np.std(recalls),
        'std_f1': np.std(f1_scores)
    }


def create_ground_truth_from_keywords(text: str, keywords: Set[str]) -> Set[str]:
    """
    Create a pseudo ground truth by checking which keywords appear in text.
    
    Args:
        text: Input text
        keywords: Set of potential skill keywords
        
    Returns:
        Set of skills found in text (as ground truth)
    """
    text_lower = text.lower()
    found = set()
    
    for keyword in keywords:
        if keyword.lower() in text_lower:
            found.add(keyword.lower())
    
    return found


def evaluate_extractor_quality(df: pd.DataFrame,
                               extracted_col: str = 'Extracted Skills',
                               known_skills: Set[str] = None) -> Dict:
    """
    Evaluate the quality of skill extraction on a DataFrame.
    
    Args:
        df: DataFrame with extracted skills
        extracted_col: Column containing extracted skills
        known_skills: Set of all known/valid skills
        
    Returns:
        Evaluation metrics and analysis
    """
    from src.skill_extractor import PREDEFINED_SKILLS
    
    if known_skills is None:
        known_skills = PREDEFINED_SKILLS
    
    # Parse extracted skills
    all_extracted = []
    for skills in df[extracted_col]:
        if isinstance(skills, str):
            skill_set = set(s.strip().lower() for s in skills.split(',') if s.strip())
        else:
            skill_set = set()
        all_extracted.append(skill_set)
    
    # Create ground truth from job descriptions
    ground_truth_list = []
    for idx, row in df.iterrows():
        text = str(row.get('Job Description', ''))
        gt = create_ground_truth_from_keywords(text, known_skills)
        ground_truth_list.append(gt)
    
    # Evaluate
    metrics = evaluate_batch(all_extracted, ground_truth_list)
    
    # Additional analysis
    all_skills_extracted = set()
    for skills in all_extracted:
        all_skills_extracted.update(skills)
    
    valid_skills = all_skills_extracted.intersection(known_skills)
    unknown_skills = all_skills_extracted - known_skills
    
    metrics['total_unique_extracted'] = len(all_skills_extracted)
    metrics['valid_skills_count'] = len(valid_skills)
    metrics['unknown_skills_count'] = len(unknown_skills)
    metrics['coverage_rate'] = len(valid_skills) / len(known_skills) if known_skills else 0
    
    return metrics


def generate_evaluation_report(metrics: Dict) -> str:
    """
    Generate a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        
    Returns:
        Formatted report string
    """
    report = """
================================================================================
                      SKILL EXTRACTION EVALUATION REPORT
================================================================================

OVERALL METRICS
---------------
Macro Precision:  {macro_precision:.4f} (± {std_precision:.4f})
Macro Recall:     {macro_recall:.4f} (± {std_recall:.4f})
Macro F1-Score:   {macro_f1:.4f} (± {std_f1:.4f})

Micro Precision:  {micro_precision:.4f}
Micro Recall:     {micro_recall:.4f}
Micro F1-Score:   {micro_f1:.4f}

EXTRACTION STATISTICS
---------------------
Total Samples Evaluated:  {total_samples}
Total Unique Skills:      {total_unique_extracted}
Valid Skills Found:       {valid_skills_count}
Unknown Skills Found:     {unknown_skills_count}
Skill Coverage Rate:      {coverage_rate:.2%}

================================================================================
    """.format(**metrics)
    
    return report


def compare_extraction_methods(df: pd.DataFrame,
                               text_column: str = 'Job Description') -> pd.DataFrame:
    """
    Compare different extraction methods.
    
    Args:
        df: Input DataFrame
        text_column: Column containing job descriptions
        
    Returns:
        DataFrame with comparison results
    """
    from src.skill_extractor import RuleBasedExtractor, MLBasedExtractor, HybridSkillExtractor, PREDEFINED_SKILLS
    
    rule_extractor = RuleBasedExtractor()
    ml_extractor = MLBasedExtractor()
    hybrid_extractor = HybridSkillExtractor()
    
    results = []
    
    for idx, row in df.iterrows():
        text = str(row[text_column])
        
        rule_skills = set(rule_extractor.extract(text))
        ml_skills = set(ml_extractor.extract(text))
        hybrid_skills = set(hybrid_extractor.extract(text))
        
        # Ground truth (skills actually in text)
        gt = create_ground_truth_from_keywords(text, PREDEFINED_SKILLS)
        
        results.append({
            'id': row.get('id', idx),
            'rule_count': len(rule_skills),
            'ml_count': len(ml_skills),
            'hybrid_count': len(hybrid_skills),
            'ground_truth_count': len(gt),
            'rule_precision': calculate_precision(rule_skills, gt),
            'ml_precision': calculate_precision(ml_skills, gt),
            'hybrid_precision': calculate_precision(hybrid_skills, gt),
            'rule_recall': calculate_recall(rule_skills, gt),
            'ml_recall': calculate_recall(ml_skills, gt),
            'hybrid_recall': calculate_recall(hybrid_skills, gt)
        })
    
    comparison_df = pd.DataFrame(results)
    
    # Add F1 scores
    for method in ['rule', 'ml', 'hybrid']:
        comparison_df[f'{method}_f1'] = comparison_df.apply(
            lambda x: calculate_f1_score(x[f'{method}_precision'], x[f'{method}_recall']),
            axis=1
        )
    
    return comparison_df


def print_comparison_summary(comparison_df: pd.DataFrame):
    """
    Print summary of method comparison.
    
    Args:
        comparison_df: DataFrame from compare_extraction_methods
    """
    print("\n" + "="*60)
    print("EXTRACTION METHOD COMPARISON")
    print("="*60)
    
    methods = ['rule', 'ml', 'hybrid']
    
    print(f"\n{'Method':<15} {'Avg Precision':<15} {'Avg Recall':<15} {'Avg F1':<15}")
    print("-"*60)
    
    for method in methods:
        avg_p = comparison_df[f'{method}_precision'].mean()
        avg_r = comparison_df[f'{method}_recall'].mean()
        avg_f1 = comparison_df[f'{method}_f1'].mean()
        print(f"{method.upper():<15} {avg_p:<15.4f} {avg_r:<15.4f} {avg_f1:<15.4f}")
    
    print("\nBest performing method by F1-Score:", end=" ")
    avg_f1_scores = {m: comparison_df[f'{m}_f1'].mean() for m in methods}
    best_method = max(avg_f1_scores, key=avg_f1_scores.get)
    print(f"{best_method.upper()} ({avg_f1_scores[best_method]:.4f})")


# Main execution
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    # Load extracted skills
    try:
        df = pd.read_csv('../output/extracted_skills.csv')
        original_df = pd.read_csv('../data/jobs.csv')
        
        # Merge for evaluation
        df = df.merge(original_df[['id', 'Job Description']], on='id', how='left')
        
        # Evaluate
        print("Evaluating skill extraction quality...")
        metrics = evaluate_extractor_quality(df)
        
        # Generate report
        report = generate_evaluation_report(metrics)
        print(report)
        
        # Compare methods (on sample)
        print("\nComparing extraction methods on sample...")
        sample_df = original_df.head(100)
        comparison = compare_extraction_methods(sample_df)
        print_comparison_summary(comparison)
        
    except FileNotFoundError:
        print("Please run skill_extractor.py first to generate extracted_skills.csv")
