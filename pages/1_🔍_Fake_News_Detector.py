import streamlit as st
from transformers import pipeline
import spacy

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Load Hugging Face text classification pipeline
@st.cache_resource
def load_huggingface_pipeline():
    return pipeline("text-classification", model="Subash2580/Bert_model_news_aggregator")

text_classifier = load_huggingface_pipeline()

# Text preprocessing function
def preprocess_text_nlp(text):
    if not text:
        return ""
    doc = nlp(text.lower())
    filtered_text = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return " ".join(filtered_text).strip()

# Prediction function using Hugging Face pipeline
def predict_news(text):
    result = text_classifier(text)[0]
    label = result['label']
    confidence = result['score'] * 100  # Confidence in percentage

    # Map LABEL_1 to "real" and LABEL_2 to "fake"
    prediction = "real" if label == "LABEL_1" else "fake"
    return prediction, confidence

# Main Streamlit app
st.title("Fake News Detector")
st.header("Enter News Headline and Description")

# Text input fields for headline and description
headline = st.text_input("Headline")
description = st.text_area("Description")

if st.button("Analyze"):
    # Combine headline and description
    combined_text = f"{headline} {description}"

    # Preprocess the combined text
    processed_text = preprocess_text_nlp(combined_text)

    # Make a prediction
    prediction, confidence = predict_news(processed_text)

    # Display the results with highlighted confidence
    st.subheader("Analysis Result")
    confidence_color = "green" if prediction == "real" else "red"
    st.markdown(f"**Prediction**: {prediction.capitalize()} - <span style='color:{confidence_color}'>{confidence:.2f}% confidence</span>", unsafe_allow_html=True)
