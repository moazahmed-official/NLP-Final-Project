import os
import sys
import streamlit as st
import pandas as pd

# Ensure local streamlit_app/src is prioritized for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.skill_extractor import extract_skills_from_dataframe, get_skill_stats

st.set_page_config(page_title='3. Skill Extraction')
st.title('🎯 استخراج المهارات')

if 'df' not in st.session_state:
    st.info('حمل مجموعة بيانات أولاً من الصفحة الرئيسية.')
else:
    df = st.session_state['df']
    method = st.selectbox('اختر الطريقة', ['hybrid','rule','ml'])
    if st.button('🔍 استخراج المهارات'):
        with st.spinner('جارٍ استخراج المهارات...'):
            skills_df = extract_skills_from_dataframe(df, method=method)
            st.session_state['skills_df'] = skills_df
        st.success('اكتملت عملية الاستخراج')

    if 'skills_df' in st.session_state:
        s_df = st.session_state['skills_df']
        st.subheader('Sample Extracted')
        st.dataframe(s_df[['id','Job Title','Extracted Skills']].head(50))

        st.download_button('⬇️ تنزيل CSV (Extracted Skills)', s_df[['id','Job Title','Extracted Skills']].to_csv(index=False).encode('utf-8'), file_name='extracted_skills.csv', mime='text/csv')

        st.subheader('Top Skills')
        counts = get_skill_stats(s_df)
        top = counts.most_common(30)
        st.table(pd.DataFrame(top, columns=['Skill','Count']))
    else:
        st.info('لا توجد نتائج استخراج بعد — شغّل الاستخراج.')
