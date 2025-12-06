import os
import sys
import streamlit as st
import pandas as pd

# Ensure local streamlit_app/src is prioritized for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.summarizer import TextRankSummarizer, TransformerSummarizer

st.set_page_config(page_title='4. Summarization')
st.title('📝 تلخيص أوصاف الوظائف')

if 'df' not in st.session_state:
    st.info('حمل مجموعة بيانات أولاً من الصفحة الرئيسية.')
else:
    df = st.session_state['df']
    method = st.selectbox('طريقة التلخيص', ['textrank','transformer'])
    num_sent = st.slider('عدد الجمل (TextRank)', 1, 5, 2)
    model_name = st.text_input('Transformer model (optional)', value='t5-small')

    if st.button('📝 توليد ملخصات على كامل المجموعة'):
        with st.spinner('جارٍ توليد الملخصات...'):
            summaries = []
            if method == 'textrank':
                tr = TextRankSummarizer()
                for t in df['Job Description'].fillna('').astype(str):
                    summaries.append(tr.summarize(t, num_sentences=num_sent))
            else:
                ts = TransformerSummarizer(model_name=model_name)
                for t in df['Job Description'].fillna('').astype(str):
                    summaries.append(ts.summarize(t))
            df['Summary'] = summaries
            st.session_state['df'] = df
            st.session_state['summary_df'] = df[['id','Job Title','Summary']]
        st.success('اكتمل توليد الملخصات')

    if 'summary_df' in st.session_state:
        st.subheader('Sample Summaries')
        st.dataframe(st.session_state['summary_df'].head(20))
        st.download_button('⬇️ تنزيل CSV (job_summary.csv)', st.session_state['summary_df'].to_csv(index=False).encode('utf-8'), file_name='job_summary.csv', mime='text/csv')
    else:
        st.info('لم يتم إنشاء ملخصات بعد — اضغط توليد الملخصات.')
