"""
Lightweight summarizers: TextRank + optional Transformer (if transformers available)
"""
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))


class TextRankSummarizer:
    def __init__(self, damping=0.85, min_diff=1e-5, max_iter=100):
        self.damping = damping
        self.min_diff = min_diff
        self.max_iter = max_iter

    def _sentence_similarity(self, s1, s2):
        s1 = [w.lower() for w in s1 if w.lower() not in STOPWORDS]
        s2 = [w.lower() for w in s2 if w.lower() not in STOPWORDS]
        all_words = list(set(s1 + s2))
        if not all_words:
            return 0
        v1 = [1 if w in s1 else 0 for w in all_words]
        v2 = [1 if w in s2 else 0 for w in all_words]
        dot = sum(a*b for a,b in zip(v1,v2))
        mag1 = sum(a*a for a in v1) ** 0.5
        mag2 = sum(b*b for b in v2) ** 0.5
        if mag1*mag2 == 0:
            return 0
        return dot/(mag1*mag2)

    def summarize(self, text, num_sentences=3):
        if not text or len(text) < 50:
            return text
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        tokenized = [word_tokenize(s) for s in sentences]
        n = len(sentences)
        sim = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i!=j:
                    sim[i,j] = self._sentence_similarity(tokenized[i], tokenized[j])
        for i in range(n):
            if sim[i].sum() != 0:
                sim[i] /= sim[i].sum()
        scores = np.ones(n)/n
        for _ in range(self.max_iter):
            prev = scores.copy()
            for i in range(n):
                scores[i] = (1-self.damping) + self.damping * sum(sim[j,i]*prev[j] for j in range(n))
            if np.abs(scores-prev).sum() < self.min_diff:
                break
        ranked = scores.argsort()[-num_sentences:][::-1]
        ranked = sorted(ranked)
        return ' '.join([sentences[i] for i in ranked])


class TransformerSummarizer:
    def __init__(self, model_name='t5-small'):
        self.model_name = model_name
        self.summarizer = None
        try:
            from transformers import pipeline
            self.summarizer = pipeline('summarization', model=self.model_name)
        except Exception:
            self.summarizer = None

    def summarize(self, text, max_length=150, min_length=30):
        if not self.summarizer:
            # fallback
            return TextRankSummarizer().summarize(text)
        # truncate
        words = text.split()
        if len(words) > 500:
            text = ' '.join(words[:500])
        if 't5' in self.model_name.lower():
            text = 'summarize: ' + text
        out = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return out[0]['summary_text']


# convenience
def summarize_text(text, method='textrank', **kwargs):
    if method == 'textrank':
        return TextRankSummarizer().summarize(text, kwargs.get('num_sentences',3))
    else:
        return TransformerSummarizer(kwargs.get('model_name','t5-small')).summarize(text, kwargs.get('max_length',150))
