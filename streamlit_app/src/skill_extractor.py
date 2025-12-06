"""
Rule-based + TF-IDF lightweight skill extractor for Streamlit app
Avoids heavy spaCy/torch imports for responsiveness in the demo.
"""
import re
from collections import Counter
from typing import List, Set

from sklearn.feature_extraction.text import TfidfVectorizer

PREDEFINED_SKILLS = {
    'python','java','javascript','typescript','c++','c#','ruby','go','php','swift','kotlin',
    'react','angular','vue','django','flask','fastapi','nodejs','express','spring',
    'flutter','react native','android','ios','sql','mysql','postgresql','mongodb','redis',
    'aws','azure','gcp','docker','kubernetes','jenkins','git','github','gitlab',
    'tensorflow','pytorch','keras','scikit-learn','pandas','numpy','nlp','ai','machine learning'
}

SKILL_PATTERNS = [re.compile(r"\\b" + re.escape(s) + r"\\b", re.IGNORECASE) for s in PREDEFINED_SKILLS]


def extract_rule_based(text: str) -> List[str]:
    found = set()
    for pat in SKILL_PATTERNS:
        if pat.search(text or ""):
            found.add(pat.pattern)
    # Map back to readable skills
    results = []
    for skill in PREDEFINED_SKILLS:
        if re.search(r"\\b" + re.escape(skill) + r"\\b", (text or ""), re.IGNORECASE):
            results.append(skill)
    return sorted(results)


def extract_tfidf_candidates(texts: List[str], top_n: int = 50) -> List[str]:
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=2000)
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    # sum tfidf across docs to rank
    scores = tfidf.sum(axis=0).A1
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [w for w, s in ranked[:top_n]]


def map_candidates_to_skills(candidates: List[str]) -> List[str]:
    matched = set()
    cand = [c.lower() for c in candidates]
    for skill in PREDEFINED_SKILLS:
        for c in cand:
            if skill in c or c in skill:
                matched.add(skill)
    return sorted(matched)


def extract_skills_from_dataframe(df, text_col='Job Description', method='hybrid'):
    df = df.copy()
    df[text_col] = df[text_col].fillna('').astype(str)
    if method == 'rule':
        df['Extracted Skills'] = df[text_col].apply(lambda x: ', '.join(extract_rule_based(x)))
    elif method == 'ml':
        candidates = extract_tfidf_candidates(df[text_col].tolist(), top_n=200)
        mapped = map_candidates_to_skills(candidates)
        # apply mapping by checking if any mapped skill in text
        df['Extracted Skills'] = df[text_col].apply(lambda x: ', '.join([s for s in mapped if re.search(r"\\b"+re.escape(s)+r"\\b", x, re.IGNORECASE)]))
    else:
        # hybrid: combine rule and ml
        candidates = extract_tfidf_candidates(df[text_col].tolist(), top_n=200)
        mapped = map_candidates_to_skills(candidates)
        def hybrid_extract(x):
            rule = set(extract_rule_based(x))
            ml_found = set([s for s in mapped if re.search(r"\\b"+re.escape(s)+r"\\b", x, re.IGNORECASE)])
            return ', '.join(sorted(rule.union(ml_found)))
        df['Extracted Skills'] = df[text_col].apply(hybrid_extract)
    df['Skills Count'] = df['Extracted Skills'].apply(lambda s: len([t for t in s.split(',') if t.strip()]))
    return df


def get_skill_stats(df, skills_col='Extracted Skills'):
    all_skills = []
    for s in df[skills_col].fillna(''):
        all_skills.extend([t.strip().lower() for t in s.split(',') if t.strip()])
    return Counter(all_skills)
