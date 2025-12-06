# Skill Extraction and Job Description Summarization Using Natural Language Processing (NLP)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Workflow](#workflow)
- [How to Run the Project](#how-to-run-the-project)
- [Project Output](#project-output)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Important Notes](#important-notes)
- [Project Contributors](#project-contributors)
- [Section & Group Details](#section--group-details)

---

## 🎯 Overview

This project provides a complete Natural Language Processing (NLP) pipeline designed to:

1. **Automatically extract technical skills** from job descriptions  
2. **Summarize long job descriptions** for faster and clearer understanding  
3. **Analyze and visualize job market trends** based on skills and roles  

---

## 📊 Dataset Description

### Data File: `data/jobs.csv`

| Field | Description | Type |
|-------|-------------|------|
| `id` | Unique job identifier | Numeric |
| `Job Title` | Name of the job | Text |
| `Job Description` | Full job description | Text |

### Dataset Stats:
- **Total jobs:** ~60,000  
- **Average description length:** ~100 words  
- **Coverage:** Multiple tech-related fields  

---

## 🔄 Workflow

### **Phase 1 — Data Preprocessing**

Text preprocessing includes:

```
Raw text → lowercase → punctuation removal → number removal → stopword removal → tokenization → lemmatization → POS tagging
```

Functions in `src/preprocess.py`:
- `clean_text()`
- `tokenize()`
- `remove_stopwords_from_tokens()`
- `lemmatize_tokens()`
- `get_pos_tags()`

---

### **Phase 2 — Skill Extraction**

#### 1. **Rule-Based Method**
- Uses a predefined skill list  
- Keyword matching  
- High precision for known skills  

#### 2. **ML-Based Method**
- TF-IDF for keyword importance  
- spaCy NER for identifying entities  
- Can detect new/unlisted skills  

**Supported Skills Include:**
- Programming: Python, Java, JavaScript, C++, Go, Rust, etc.  
- Web Dev: React, Angular, Vue, Django, Flask, Node.js  
- Databases: MySQL, PostgreSQL, MongoDB, Redis  
- Cloud: AWS, Azure, GCP, Docker, Kubernetes  
- Data Science: TensorFlow, PyTorch, Pandas, NumPy  
- And **200+ more**  

---

### **Phase 3 — Text Summarization**

#### 1. **TextRank**
- Graph-based ranking  
- Extractive summaries  
- Fast and efficient  

#### 2. **Transformer-Based Models**
- T5-small, BART  
- Abstractive summarization  
- Better quality, slower runtime  

---

### **Phase 4 — Evaluation**

| Metric | Meaning |
|--------|---------|
| **Precision** | % of extracted skills that are correct |
| **Recall** | % of actual skills successfully extracted |
| **F1-Score** | Harmonic mean of precision & recall |

---

### **Phase 5 — Visualization**

Generated charts include:
- Most common job titles  
- Word cloud of job descriptions  
- Most frequent skills  
- Skill count distribution per job  

---

## 🚀 How to Run the Project

### **Requirements**
- Python 3.8+
- pip package manager

---

### **1. Clone the repository**
```bash
git clone <repository-url>
cd final_project
```

### **2. Create a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

---

## ▶️ Running the Pipeline

### **Run Preprocessing**
```bash
cd src
python preprocess.py
```

### **Run Skill Extraction**
```bash
python skill_extractor.py
```
Output: `output/extracted_skills.csv`

### **Run Summarization**
```bash
python summarizer.py
```
Output: `output/job_summary.csv`

### **Run Evaluation**
```bash
python evaluate.py
```

### **Generate Visualizations**
```bash
python viz.py
```
Output directory: `output/figures/`

### **Interactive Notebook**
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 🌐 Streamlit Web App (Interactive UI)

A fully interactive Streamlit web interface is included in:

**`streamlit_app/`**

To run it:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install streamlit

cd streamlit_app
streamlit run app.py
```

Default URL:  
`http://localhost:8501`

To specify a port:

```powershell
streamlit run app.py --server.port 8502
```

### Available Pages:
- **Home** – Upload & preview data  
- **Data Overview** – Stats and tables  
- **Preprocessing** – Clean text & compare before/after  
- **Skill Extraction** – Rule-based, ML, or hybrid  
- **Summarization** – TextRank or Transformer  
- **Visualization** – Word cloud and charts  

---

## 📁 Project Output

### 1. Extracted Skills  
`output/extracted_skills.csv`

| id | Job Title | Extracted Skills |
|----|-----------|------------------|
| 0 | Flutter Developer | flutter, dart, android, ios, git |
| 1 | Python Developer | python, django, sql, api, git |

---

### 2. Summaries  
`output/job_summary.csv`

| id | Job Title | Summary |
|----|-----------|---------|
| 0 | Flutter Developer | We are looking for a professional Flutter Developer... |

---

### 3. Visualizations  
`output/figures/`

- `job_titles.png`  
- `wordcloud.png`  
- `skill_frequency.png`  
- `skills_per_job.png`  

---

### **Example Result**

**Original:**
```
Flutter Developer
We are looking for hire experts flutter developer...
```

**Skills Extracted:**
```
flutter, dart, android, ios, mobile development
```

**Summary:**
```
Seeking a Flutter Developer for full-time or part-time work with a salary range of ₹20,000–₹40,000 per month.
```

---

## 🛠 Technologies Used

### NLP
- NLTK  
- spaCy  
- Transformers (T5, BART)

### Machine Learning
- scikit-learn  
- PyTorch  

### Data Analysis & Visualization
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- WordCloud  

### Tools
- Jupyter Notebook  
- Git  
- Streamlit  

---

## 📂 Project Structure

```
project/
│
├── data/
│   └── jobs.csv
│
├── src/
│   ├── preprocess.py
│   ├── skill_extractor.py
│   ├── summarizer.py
│   ├── evaluate.py
│   └── viz.py
│
├── streamlit_app/
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Data_Overview.py
│   │   ├── 2_Preprocessing.py
│   │   ├── 3_Skill_Extraction.py
│   │   ├── 4_Summarization.py
│   │   └── 5_Visualization.py
│   ├── src/
│   │   ├── preprocess.py
│   │   ├── skill_extractor.py
│   │   ├── summarizer.py
│   │   └── viz.py
│   ├── data/
│   │   └── jobs.csv
│   └── README.md
│
├── notebooks/
│   └── exploration.ipynb
│
├── output/
│   ├── extracted_skills.csv
│   ├── job_summary.csv
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

## ⚠️ Important Notes

### Code Organization
- Each module is standalone  
- All functions include docstrings  
- Follows PEP8 style guidelines  

### Adding New Skills
Inside `src/skill_extractor.py`:

```python
PREDEFINED_SKILLS = {
    "new_skill_1",
    "new_skill_2",
}
```

### Improving Summaries

Use larger models:
```python
summarizer = TransformerSummarizer(model_name='facebook/bart-large-cnn')
```

Control summary length:
```python
summary = summarizer.summarize(text, max_length=200, min_length=50)
```

Increase TextRank sentences:
```python
textrank.summarize(text, num_sentences=5)
```

### Performance Tips
- Use TextRank for large datasets  
- Use hybrid extraction for best accuracy  
- Use sampling during experimentation  

### About `src` Naming Conflict
Both root and Streamlit folders include a `src/` directory, which may cause import conflicts.

Solutions:
- Run Streamlit from its folder  
- Add `streamlit_app/` to `sys.path`  
- Rename one of the directories  

---

### Project Contributors

- Moaz Ahmed 
- Mostafa Rabee
- Mostafa Ahmed
- Hamza Yasser
- Mariam Medhat

---

### Section&Group Details 

- Section : 14
- Group : 3

---