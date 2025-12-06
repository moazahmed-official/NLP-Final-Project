import os
import sys
import streamlit as st
from streamlit import expander
import pandas as pd

# Ensure local streamlit_app/src is prioritized for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib.util

# Load the local preprocess module directly to avoid collision with top-level `src` package
local_preprocess_path = os.path.join(ROOT, 'src', 'preprocess.py')
spec = importlib.util.spec_from_file_location('local_preprocess', local_preprocess_path)
local_preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_preprocess)

preprocess_text = local_preprocess.preprocess_text
preprocess_series = local_preprocess.preprocess_series

st.set_page_config(page_title='2. Preprocessing')
st.title('⚙️ المعالجة المسبقة (Preprocessing)')

if 'df' not in st.session_state:
    st.info('لا توجد بيانات محملة. ارفع ملف CSV في الصفحة الرئيسية.')
else:
    df = st.session_state['df']
    st.write('اختر الخيارات ثم اضغط Apply Preprocessing')
    remove_stop = st.checkbox('إزالة Stopwords', value=True)
    do_lemma = st.checkbox('Lemmatization', value=True)
    if st.button('🔧 Apply Preprocessing'):
        with st.spinner('جارٍ المعالجة...'):
            df['cleaned_text'] = preprocess_series(df['Job Description'], remove_stopwords=remove_stop, lemmatize=do_lemma)
            st.session_state['df'] = df
        st.success('تم تطبيق المعالجة')

    # Show before/after for a selected row
    idx = st.number_input('اختر صف لمعاينة قبل/بعد (index)', min_value=0, max_value=max(0,len(df)-1), value=0)
    orig = str(df['Job Description'].iloc[idx])
    cleaned = df.get('cleaned_text', pd.Series(['']*len(df))).iloc[idx]
    with expander('الأصلية'):
        st.write(orig[:2000])
    with expander('بعد المعالجة'):
        st.write(cleaned[:2000])

    if st.button('💾 حفظ cleaned_text إلى ملف output/cleaned.csv'):
        df[['id','Job Title','cleaned_text']].to_csv('output/cleaned.csv', index=False)
        st.success('تم الحفظ: output/cleaned.csv')
