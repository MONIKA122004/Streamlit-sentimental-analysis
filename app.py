import os
import re
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import contextlib

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="💬 Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Suppress NLTK info output
# ----------------------------
with contextlib.redirect_stdout(None):
    for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        try:
            if pkg == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

# Ensure stopwords load correctly
try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

LEMMATIZER = WordNetLemmatizer()
LABEL_MAP_INT2STR = {1: "positive", 0: "neutral", -1: "negative"}
PIPE_PATH = "pipeline.pkl"

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [LEMMATIZER.lemmatize(tok) for tok in text.split() if tok not in STOPWORDS]
    return " ".join(tokens)

# ----------------------------
# Parse TXT files
# ----------------------------
def parse_txt(uploaded_file):
    content = uploaded_file.read().decode("utf-8").splitlines()
    texts = [line.strip() for line in content if line.strip()]
    return pd.DataFrame({"text": texts})

# ----------------------------
# Load pipeline
# ----------------------------
def load_pipeline():
    if os.path.exists(PIPE_PATH):
        with open(PIPE_PATH, "rb") as f:
            return pickle.load(f)
    return None

# ----------------------------
# Predict
# ----------------------------
def predict_sentiment(pipe, df, text_col="text"):
    if text_col not in df.columns:
        st.error(f"Column '{text_col}' not found in uploaded file.")
        return None
    df[text_col] = df[text_col].astype(str).map(clean_text)
    df['predicted_label'] = pipe.predict(df[text_col])
    df['sentiment'] = df['predicted_label'].map(LABEL_MAP_INT2STR)
    return df

# ----------------------------
# UI
# ----------------------------
st.title("💬 Sentiment Analysis Dashboard")
st.markdown("""
Welcome! Upload a TXT, CSV, or Excel file and see **sentiment distribution** instantly.  
Enjoy interactive charts and download the labeled results!""")

# Sidebar
st.sidebar.header("Upload & Settings")
user_file = st.sidebar.file_uploader("Upload TXT/CSV/Excel", type=["txt","csv","xlsx"])
text_col_pred = st.sidebar.text_input("Text column for prediction", value="text")
pipe = load_pipeline()

if pipe is None:
    st.warning("⚠️ No trained model found. Make sure 'pipeline.pkl' exists in the app folder.")

elif user_file is not None and st.sidebar.button("Predict Sentiments"):

    # Load file
    if user_file.name.endswith("txt"):
        df_user = parse_txt(user_file)
    elif user_file.name.endswith("csv"):
        df_user = pd.read_csv(user_file)
    else:
        df_user = pd.read_excel(user_file)

    if df_user.empty:
        st.warning("Uploaded file is empty.")
    else:
        df_result = predict_sentiment(pipe, df_user, text_col_pred)

        if df_result is not None:
            # ----------------------------
            # Layout for summary cards
            # ----------------------------
            counts = df_result['sentiment'].value_counts().reindex(["positive","neutral","negative"], fill_value=0)
            col1, col2, col3 = st.columns(3)
            col1.metric("😊 Positive", counts["positive"])
            col2.metric("😐 Neutral", counts["neutral"])
            col3.metric("😞 Negative", counts["negative"])

            st.markdown("---")

            # ----------------------------
            # Interactive charts
            # ----------------------------
            st.subheader("📊 Sentiment Distribution")

            fig_pie = px.pie(df_result, names='sentiment', color='sentiment',
                             color_discrete_map={'positive':'green','neutral':'gray','negative':'red'},
                             title="Pie Chart of Sentiments")
            st.plotly_chart(fig_pie, use_container_width=True)

            # Prepare data for bar chart
            bar_data = df_result['sentiment'].value_counts().reset_index()
            bar_data.columns = ['Sentiment', 'Count']

            fig_bar = px.bar(
                bar_data,
                x='Sentiment',
                y='Count',
                color='Sentiment',
                color_discrete_map={'positive':'green','neutral':'gray','negative':'red'},
                title="Bar Chart of Sentiments"
            )

            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")

            # ----------------------------
            # File preview
            # ----------------------------
            with st.expander("Preview Labeled Data"):
                st.dataframe(df_result.head(10))

            # ----------------------------
            # Download
            # ----------------------------
            buffer = io.BytesIO()
            df_result.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button("⬇️ Download Labeled CSV", buffer,
                               file_name="sentiment_labeled.csv", mime="text/csv")

st.markdown("""
<style>
    .css-18e3th9 {padding-top: 2rem;}
    .css-1d391kg {padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

