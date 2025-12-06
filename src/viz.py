"""
Visualization Module for Job Description NLP Analysis
======================================================
This module creates visualizations including:
- Bar chart of most common job titles
- Word cloud for job descriptions
- Most frequent extracted skills
- Additional analysis charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_job_title_distribution(df: pd.DataFrame, 
                                 title_column: str = 'Job Title',
                                 top_n: int = 15,
                                 figsize: tuple = (12, 6),
                                 save_path: str = None) -> plt.Figure:
    """
    Create bar chart of most common job titles.
    
    Args:
        df: DataFrame with job data
        title_column: Column containing job titles
        top_n: Number of top titles to show
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Count job titles
    title_counts = df[title_column].value_counts().head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(title_counts)), title_counts.values, color=plt.cm.viridis(np.linspace(0, 0.8, len(title_counts))))
    
    # Customize
    ax.set_yticks(range(len(title_counts)))
    ax.set_yticklabels(title_counts.index)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Job Postings', fontsize=12)
    ax.set_title(f'Top {top_n} Most Common Job Titles', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, title_counts.values)):
        ax.text(val + 0.5, i, str(val), va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def create_wordcloud(texts: List[str],
                     figsize: tuple = (14, 7),
                     max_words: int = 100,
                     background_color: str = 'white',
                     colormap: str = 'viridis',
                     save_path: str = None) -> plt.Figure:
    """
    Create word cloud from job descriptions.
    
    Args:
        texts: List of text strings
        figsize: Figure size
        max_words: Maximum number of words
        background_color: Background color
        colormap: Color scheme
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    from wordcloud import WordCloud
    
    # Combine all texts
    combined_text = ' '.join([str(t) for t in texts])
    
    # Create word cloud
    wordcloud = WordCloud(
        width=1400,
        height=700,
        max_words=max_words,
        background_color=background_color,
        colormap=colormap,
        min_font_size=10,
        max_font_size=150,
        random_state=42
    ).generate(combined_text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Job Descriptions', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_skill_frequency(df: pd.DataFrame,
                          skills_column: str = 'Extracted Skills',
                          top_n: int = 20,
                          figsize: tuple = (12, 8),
                          save_path: str = None) -> plt.Figure:
    """
    Create bar chart of most frequent extracted skills.
    
    Args:
        df: DataFrame with extracted skills
        skills_column: Column containing skills
        top_n: Number of top skills to show
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Parse and count skills
    all_skills = []
    for skills in df[skills_column]:
        if isinstance(skills, str):
            all_skills.extend([s.strip().lower() for s in skills.split(',') if s.strip()])
        elif isinstance(skills, list):
            all_skills.extend([s.lower() for s in skills])
    
    skill_counts = Counter(all_skills)
    top_skills = skill_counts.most_common(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    skills, counts = zip(*top_skills)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(skills)))[::-1]
    
    bars = ax.barh(range(len(skills)), counts, color=colors)
    
    # Customize
    ax.set_yticks(range(len(skills)))
    ax.set_yticklabels([s.title() for s in skills])
    ax.invert_yaxis()
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Most Frequently Extracted Skills', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, counts)):
        ax.text(val + 0.5, i, str(val), va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_skills_per_job(df: pd.DataFrame,
                         skills_column: str = 'Extracted Skills',
                         figsize: tuple = (10, 6),
                         save_path: str = None) -> plt.Figure:
    """
    Create histogram of skills count per job.
    
    Args:
        df: DataFrame with extracted skills
        skills_column: Column containing skills
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Count skills per job
    skills_per_job = []
    for skills in df[skills_column]:
        if isinstance(skills, str):
            count = len([s for s in skills.split(',') if s.strip()])
        elif isinstance(skills, list):
            count = len(skills)
        else:
            count = 0
        skills_per_job.append(count)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(skills_per_job, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(np.mean(skills_per_job), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(skills_per_job):.1f}')
    ax.axvline(np.median(skills_per_job), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(skills_per_job):.1f}')
    
    ax.set_xlabel('Number of Skills per Job', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Skills Count per Job Posting', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_skill_category_distribution(df: pd.DataFrame,
                                      skills_column: str = 'Extracted Skills',
                                      figsize: tuple = (10, 10),
                                      save_path: str = None) -> plt.Figure:
    """
    Create pie chart of skill categories.
    
    Args:
        df: DataFrame with extracted skills
        skills_column: Column containing skills
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Define skill categories
    categories = {
        'Programming Languages': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php', 'swift', 'kotlin'],
        'Web Development': ['react', 'angular', 'vue', 'html', 'css', 'nodejs', 'django', 'flask', 'express'],
        'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'firebase'],
        'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'devops'],
        'Data Science & ML': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'ai'],
        'Mobile Development': ['flutter', 'react native', 'android', 'ios', 'swift'],
        'Other Tools': ['git', 'github', 'agile', 'scrum', 'jira', 'api']
    }
    
    # Parse all skills
    all_skills = []
    for skills in df[skills_column]:
        if isinstance(skills, str):
            all_skills.extend([s.strip().lower() for s in skills.split(',') if s.strip()])
    
    skill_counts = Counter(all_skills)
    
    # Categorize
    category_counts = {}
    for category, keywords in categories.items():
        count = sum(skill_counts.get(kw, 0) for kw in keywords)
        if count > 0:
            category_counts[category] = count
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if category_counts:
        labels = list(category_counts.keys())
        sizes = list(category_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           explode=[0.02] * len(labels))
        
        ax.set_title('Distribution of Skills by Category', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No skills data available', ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_description_length_distribution(df: pd.DataFrame,
                                          text_column: str = 'Job Description',
                                          figsize: tuple = (10, 6),
                                          save_path: str = None) -> plt.Figure:
    """
    Create histogram of job description lengths.
    
    Args:
        df: DataFrame with job descriptions
        text_column: Column containing descriptions
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Calculate word counts
    word_counts = df[text_column].apply(lambda x: len(str(x).split()))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(word_counts, bins=50, color='coral', edgecolor='white', alpha=0.7)
    ax.axvline(np.mean(word_counts), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(word_counts):.0f} words')
    ax.axvline(np.median(word_counts), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.median(word_counts):.0f} words')
    
    ax.set_xlabel('Number of Words', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Job Description Lengths', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def create_all_visualizations(jobs_df: pd.DataFrame,
                               skills_df: pd.DataFrame = None,
                               output_dir: str = '../output/figures'):
    """
    Generate all visualizations and save them.
    
    Args:
        jobs_df: DataFrame with job data
        skills_df: DataFrame with extracted skills
        output_dir: Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. Job title distribution
    print("  - Creating job title distribution chart...")
    plot_job_title_distribution(jobs_df, save_path=f'{output_dir}/job_titles.png')
    
    # 2. Word cloud
    print("  - Creating word cloud...")
    create_wordcloud(jobs_df['Job Description'].tolist(), 
                    save_path=f'{output_dir}/wordcloud.png')
    
    # 3. Description length distribution
    print("  - Creating description length distribution...")
    plot_description_length_distribution(jobs_df, 
                                         save_path=f'{output_dir}/description_lengths.png')
    
    # Skills visualizations (if available)
    if skills_df is not None:
        # 4. Skill frequency
        print("  - Creating skill frequency chart...")
        plot_skill_frequency(skills_df, save_path=f'{output_dir}/skill_frequency.png')
        
        # 5. Skills per job
        print("  - Creating skills per job distribution...")
        plot_skills_per_job(skills_df, save_path=f'{output_dir}/skills_per_job.png')
        
        # 6. Skill categories
        print("  - Creating skill category distribution...")
        plot_skill_category_distribution(skills_df, 
                                         save_path=f'{output_dir}/skill_categories.png')
    
    print(f"\nAll visualizations saved to {output_dir}/")
    plt.close('all')


# Main execution
if __name__ == "__main__":
    # Load data
    jobs_df = pd.read_csv('../data/jobs.csv')
    
    # Try to load skills data
    try:
        skills_df = pd.read_csv('../output/extracted_skills.csv')
    except FileNotFoundError:
        print("Note: extracted_skills.csv not found. Running without skills visualizations.")
        skills_df = None
    
    # Generate all visualizations
    create_all_visualizations(jobs_df, skills_df)
    
    # Show sample
    print("\nSample visualizations have been generated!")
    plt.show()
