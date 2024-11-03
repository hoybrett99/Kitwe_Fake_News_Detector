import pandas as pd
import streamlit as st
from google.cloud import storage
from io import BytesIO  # For handling binary data like Parquet files
import os
import torch
from transformers import AutoTokenizer
from torch.nn.functional import softmax
from huggingface_hub import hf_hub_download

# Streamlit page configuration
st.set_page_config(
    page_title="Kitwe News Today",
    page_icon="📰",
)

# Google Cloud Storage Configuration
BUCKET_NAME = os.getenv("BUCKET")
OBJECT_KEY = os.getenv("OBJECT_KEY_PARQUET")
PROJECT_ID = os.getenv("PROJECT_ID")

# Function to download the data from Google Cloud Storage
@st.cache_data
def download_data_from_gcs(bucket_name, object_key, project_id=PROJECT_ID):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_key)
    data_bytes = blob.download_as_bytes()
    data = pd.read_parquet(BytesIO(data_bytes))
    return data

# Load BERT model for predictions
@st.cache_resource
def load_bert_model():
    model_path = hf_hub_download(repo_id="hoybrett99/KitweFakeNewsDetector-BERT", filename="quantized_final_bert_model_complete.pth")
    model = torch.load(model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

loaded_model, tokenizer = load_bert_model()

# Prediction function using the loaded BERT model
def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        confidence = softmax(logits, dim=1).max().item() * 100  # Confidence in percentage
        prediction = "real" if logits.argmax() == 1 else "fake"
    return prediction, confidence

# Load and process the data
try:
    df = download_data_from_gcs(BUCKET_NAME, OBJECT_KEY)

    # Apply prediction model on the processed descriptions
    df[['prediction', 'confidence']] = df['processed_description'].apply(lambda x: pd.Series(predict_news(x)))

    st.title("Kitwe News Today")
    st.subheader("Latest News")

    # Dropdown for selecting categories
    unique_categories = df['Category'].unique()
    selected_categories = st.multiselect("Filter by Category", unique_categories, default=unique_categories)

    # Filter the DataFrame by the selected categories
    filtered_df = df[df['Category'].isin(selected_categories)]

    # Display the news in a formatted way
    for _, row in filtered_df.iterrows():
        # Display the headline as a clickable hyperlink
        st.markdown(f"### [{row['Headline']}]({row['Link']})")

        # Display Source, Category, and Date
        st.markdown(f"**Source**: {row['Source']}  |  **Category**: {row['Category']} | {row['Date']}")

        # Display the main description paragraph
        st.write(row['Description'])

        # Display prediction label with color-coding
        prediction_color = "green" if row['prediction'] == "real" else "red"
        st.markdown(
            f"**Prediction**: <span style='color:{prediction_color}'>{row['prediction'].capitalize()}</span> "
            f"({row['confidence']:.2f}%)",
            unsafe_allow_html=True
        )

        # Add a horizontal divider between news items
        st.markdown("---")

except Exception as e:
    st.error(f"Failed to load or process data: {e}")
