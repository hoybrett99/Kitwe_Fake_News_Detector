import torch
from transformers import AutoTokenizer
from torch.nn.functional import softmax
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Kitwe Fake News Detector",
    page_icon="📰",
)

# Download the model from Hugging Face if not already downloaded
model_path = hf_hub_download(repo_id="hoybrett99/KitweFakeNewsDetector-BERT", filename="quantized_final_bert_model_complete.pth")

# Load the model and tokenizer
loaded_model = torch.load(model_path)
loaded_model.eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define the prediction function
def predict_news(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        # Calculate confidence score
        confidence = softmax(logits, dim=1).max().item() * 100  # Confidence in percentage
        # Determine prediction label
        prediction = "real" if logits.argmax() == 1 else "fake"

    return prediction, confidence

# Streamlit app layout
st.title("Fake News Detector")
st.write("Enter a news headline or description, and the model will predict if it's real or fake along with a confidence score.")

# Text input
text_input = st.text_input("Enter a news headline or description:")

# Prediction button
if st.button("Predict"):
    if text_input:
        # Get prediction and confidence
        prediction, confidence = predict_news(text_input)

        # Display the results
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        st.write("Please enter some text to get a prediction.")
