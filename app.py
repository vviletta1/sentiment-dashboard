import streamlit as st
import pandas as pd
from transformers import pipeline
import collections

# --- Logo or Brand Image in Sidebar ---
st.sidebar.image(
    "https://i.imgur.com/9b4GdBR.png",  # Change to your own logo URL if you want!
    width=120,
    caption="VeeBot AI"
)

# --- Custom Styles for Modern Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f9fafb;
    }
    h1, h2, h3 {
        color: #1d3557 !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
    }
    .stButton>button {
        color: #fff !important;
        background-color: #457b9d !important;
        border-radius: 8px !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef !important;
        border-radius: 8px 8px 0 0 !important;
        font-size: 16px;
        font-weight: 600;
        padding: 6px 20px !important;
        color: #1d3557 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title and Sidebar Header ---
st.title("🌈 Sentiment & Emotion Dashboard")
st.sidebar.header("📂 Upload or Paste Text Data")

# --- Load Hugging Face pipelines (force PyTorch)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)
emotion_pipe = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    framework="pt",
    return_all_scores=True
)

# --- Sidebar Upload or Paste
uploaded_file = st.sidebar.file_uploader("Upload CSV (column: 'text')", type=["csv"])
input_text = st.sidebar.text_area("Or paste multiple lines of text:", height=120)

# --- Get texts
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    texts = df['text'].astype(str).tolist()
elif input_text.strip():
    texts = [line.strip() for line in input_text.split('\n') if line.strip()]
else:
    st.info("📝 Upload a CSV file or paste some text to begin.")
    st.stop()

# --- Sentiment & Emotion Analysis
with st.spinner("Analyzing..."):
    sentiments = sentiment_pipe(texts)
    emotions_list = [emotion_pipe(t)[0] for t in texts]

# --- Sentiment Summary DataFrame
sentiment_df = pd.DataFrame(sentiments)

# --- Emotion Summary Aggregation
emotion_totals = collections.defaultdict(float)
emotion_counts = collections.defaultdict(int)
for emotion_scores in emotions_list:
    for entry in emotion_scores:
        emotion_totals[entry['label']] += entry['score']
        emotion_counts[entry['label']] += 1
avg_emotion = {k: (emotion_totals[k] / emotion_counts[k]) for k in emotion_totals}

# --- Tabs Layout for Results ---
tab1, tab2, tab3 = st.tabs(["📊 Sentiment", "💡 Emotions", "📝 Recent Messages"])

with tab1:
    st.header("📊 Sentiment Results")
    st.bar_chart(sentiment_df['label'].value_counts())

with tab2:
    st.header("💡 Average Emotion Scores")
    st.bar_chart(pd.Series(avg_emotion))

with tab3:
    st.header("📝 Recent Messages & Results")
    for i, text in enumerate(texts[:10]):
        st.markdown(f"<div style='background-color:#e9ecef; border-radius:12px; padding:10px; margin-bottom:12px;'>"
                    f"<strong>Text:</strong> {text[:150]}{'...' if len(text) > 150 else ''}<br>"
                    f"<strong>Sentiment:</strong> <span style='color:#457b9d'>{sentiments[i]['label']}</span> "
                    f"({sentiments[i]['score']:.2f})<br>"
                    f"<strong>Top Emotion:</strong> <span style='color:#e76f51'>{max(emotions_list[i], key=lambda x: x['score'])['label']}</span> "
                    f"({max(emotions_list[i], key=lambda x: x['score'])['score']:.2f})"
                    f"</div>",
                    unsafe_allow_html=True)
