import os
import sys
import streamlit as st
import pandas as pd

# Ensure local streamlit_app/src is prioritized for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.viz import plot_job_title_counts, create_wordcloud_from_texts, plot_skill_freq

st.set_page_config(page_title='5. Visualization')
st.title('📊 التصور والرسوم البيانية')

if 'df' not in st.session_state:
    st.info('حمل مجموعة بيانات أولاً من الصفحة الرئيسية.')
else:
    df = st.session_state['df']
    st.subheader('Word Cloud (أوصاف الوظائف)')
    fig_wc = create_wordcloud_from_texts(df['Job Description'].fillna('').astype(str).tolist())
    st.pyplot(fig_wc)

    st.subheader('Top Job Titles')
    fig_titles = plot_job_title_counts(df)
    st.pyplot(fig_titles)

    if 'skills_df' in st.session_state:
        st.subheader('Top Extracted Skills')
        fig_sk = plot_skill_freq(st.session_state['skills_df']['Extracted Skills'])
        st.pyplot(fig_sk)
    else:
        st.info('شغّل استخراج المهارات في صفحة "استخراج المهارات" لعرض تكرار المهارات.')
