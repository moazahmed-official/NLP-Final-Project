import os
import sys
import streamlit as st
import pandas as pd

# Ensure local streamlit_app/src is prioritized for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title='1. Data Overview')
st.title('📄 نظرة عامة على البيانات')

if 'df' not in st.session_state:
    st.info('لا توجد بيانات محملة. ارجع إلى الصفحة الرئيسية وارفع ملف CSV.')
else:
    df = st.session_state['df']
    st.subheader('Sample (أول 20 صف)')
    st.dataframe(df.head(20))

    st.subheader('Statistics')
    st.write('عدد الأسطر:', len(df))
    st.write(df.describe(include='all'))

    st.subheader('Jobs per Title')
    st.write(df['Job Title'].value_counts().head(30))

    if st.button('تنظيف أسماء الأعمدة (تحويل لمسافات وstrip)'):
        df.columns = [c.strip() for c in df.columns]
        st.session_state['df'] = df
        st.success('تم تنظيف أسماء الأعمدة')
