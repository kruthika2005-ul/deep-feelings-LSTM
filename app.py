import streamlit as st
from src.predict import SentimentPredictor

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Deep Feelings",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_predictor():
    return SentimentPredictor(
        model_path="models/sentiment_model.keras",
        tokenizer_path="models/tokenizer.pkl"
    )

predictor = load_predictor()

# -----------------------------
# Title
# -----------------------------
st.title("🎭 Deep Feelings")
st.subheader("Movie Review Sentiment Analysis using LSTM")

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 About")

st.sidebar.info("""
### Deep Feelings

This application predicts whether a movie review is:

😊 **Positive**

😞 **Negative**

using a Deep Learning **LSTM** model.

**Dataset**
- IMDb Movie Reviews

**Model**
- LSTM Neural Network

**Framework**
- TensorFlow / Keras
""")

# -----------------------------
# User Input
# -----------------------------
review = st.text_area(
    "📝 Enter a Movie Review",
    height=220,
    placeholder="Example: This movie was absolutely fantastic. I loved every minute!"
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        sentiment, confidence = predictor.predict(review)

        st.markdown("---")

        if "Positive" in sentiment:
            st.success(f"### Prediction: {sentiment}")
        else:
            st.error(f"### Prediction: {sentiment}")

        st.metric(
            label="Confidence Score",
            value=f"{confidence*100:.2f}%"
        )

# -----------------------------
# Examples
# -----------------------------
st.markdown("---")

st.subheader("📖 Try These Reviews")

col1, col2 = st.columns(2)

with col1:
    st.success("""
**Positive Review**

This movie was absolutely amazing.
The acting was brilliant and the story
kept me engaged till the end.
""")

with col2:
    st.error("""
**Negative Review**

Worst movie I have ever watched.
The story was boring and the acting
was terrible.
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption("Developed using ❤️ Streamlit, TensorFlow, and LSTM")