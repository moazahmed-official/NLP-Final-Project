"""
Visualization helpers for Streamlit pages
"""
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np


def plot_job_title_counts(df, title_col='Job Title', top_n=20):
    counts = df[title_col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3*len(counts))))
    ax.barh(counts.index, counts.values, color='tab:blue')
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title('Top Job Titles')
    plt.tight_layout()
    return fig


def create_wordcloud_from_texts(texts, max_words=150):
    text = ' '.join([str(t) for t in texts if t])
    wc = WordCloud(width=800, height=400, background_color='white', max_words=max_words).generate(text)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig


def plot_skill_freq(skills_series, top_n=20):
    all_skills = []
    for s in skills_series.fillna(''):
        all_skills.extend([t.strip().lower() for t in s.split(',') if t.strip()])
    counts = pd.Series(all_skills).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3*len(counts))))
    ax.barh(counts.index, counts.values, color='tab:green')
    ax.invert_yaxis()
    ax.set_title('Top Extracted Skills')
    plt.tight_layout()
    return fig
