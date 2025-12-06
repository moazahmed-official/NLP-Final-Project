import os
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NLP Job Skills & Summarization", layout="wide")

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

st.title("🔥 استخراج المهارات وتلخيص أوصاف الوظائف — واجهة تفاعلية")
st.markdown(
    "تطبيق ويب تفاعلي لتحليل أوصاف الوظائف: التحميل، المعالجة، استخراج المهارات، التلخيص، والتصور. استخدم القوائم الجانبية للتنقل بين الصفحات.")

# Sidebar: dataset upload / load default
with st.sidebar.expander("📥 Dataset"):
    uploaded_file = st.file_uploader("ارفع ملف CSV (الحقول: id, Job Title, Job Description)", type=["csv"] )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            save_path = os.path.join(DATA_DIR, "uploaded_jobs.csv")
            df.to_csv(save_path, index=False)
            st.success(f"تم حفظ الملف إلى: {save_path}")
            st.session_state['df'] = df
        except Exception as e:
            st.error(f"خطأ في قراءة الملف: {e}")
    else:
        default_path = os.path.join("..","data","jobs.csv")
        example_path = os.path.join(DATA_DIR, "jobs.csv")
        # If no uploaded file but repository has data, copy it to streamlit data dir for consistency
        if os.path.exists(default_path) and not os.path.exists(example_path):
            try:
                base_df = pd.read_csv(default_path)
                base_df.to_csv(example_path, index=False)
            except Exception:
                pass
        if 'df' not in st.session_state:
            # load example if present
            if os.path.exists(example_path):
                st.session_state['df'] = pd.read_csv(example_path)
            else:
                st.session_state['df'] = pd.DataFrame(columns=['id','Job Title','Job Description'])

st.sidebar.markdown("---")
st.sidebar.write("💡 صفحة التطبيق متاحة في مجلد `streamlit_app/pages/` — استخدم شريط التصفح الأيسر في Streamlit.")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 إعادة تحميل البيانات من disk"):
    # reload from saved path if exists
    saved = os.path.join(DATA_DIR, "uploaded_jobs.csv")
    if os.path.exists(saved):
        st.session_state['df'] = pd.read_csv(saved)
        st.success("تم إعادة تحميل الملف المحفوظ.")
    else:
        st.warning("لا يوجد ملف محلي محفوظ. قم برفع ملف CSV أولاً.")

st.sidebar.markdown("---")
st.sidebar.write("تلميح: انتقل لصفحات التطبيق عبر واجهة Streamlit Pages (شريط جانبي).")

# Quick summary on main page
st.subheader("ملخص بيانات الحمولة الحالية")
if 'df' in st.session_state and not st.session_state['df'].empty:
    df = st.session_state['df']
    st.write(f"عدد الصفوف: {len(df)}")
    st.dataframe(df.head(10))
else:
    st.info("لا توجد بيانات محملة بعد — ارفع ملف CSV أو ضع ملف `jobs.csv` في `streamlit_app/data/`.")

st.markdown("---")
st.write("إذا رغبت، افتح أي صفحة من صفحات التطبيق لتشغيل المعالجة، الاستخراج، التلخيص، والتصور.")
