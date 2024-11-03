import pandas as pd
import streamlit as st
from google.cloud import storage
from io import BytesIO  # For handling binary data like Parquet files
import os
import torch
from transformers import AutoTokenizer
from torch.nn.functional import softmax
from huggingface_hub import hf_hub_download
import spacy
from datetime import datetime, timedelta

# Define keywords for each category
categories_keywords = {
    'sports': ['football', 'soccer', 'basketball', 'tennis', 'cricket', 'olympics', 'athlete', 'sports'],
    'politics': ['government', 'election', 'politician', 'policy', 'parliament', 'minister', 'president', 'vote'],
    'education': ['school', 'university', 'education', 'college', 'students', 'learning', 'teacher', 'scholarship'],
    'health and wellness': ['health', 'hospital', 'doctor', 'wellness', 'mental health', 'fitness', 'medicine', 'disease'],
    'development': ['development', 'infrastructure', 'construction', 'road', 'bridge', 'building', 'urbanization'],
    'narcotics': ['narcotics', 'drug', 'cocaine', 'heroin', 'meth', 'drug trafficking', 'illegal drugs'],
    'fashion': ['fashion', 'clothing', 'designer', 'runway', 'model', 'style', 'apparel', 'trends'],
    'local news': ['local', 'community', 'city', 'town', 'village', 'municipality', 'neighborhood', 'region'],
    'economy news': ['economy', 'economic', 'finance', 'market', 'stocks', 'currency', 'inflation', 'gdp'],
    'business news': ['business', 'company', 'corporation', 'entrepreneur', 'startup', 'industry', 'investment', 'profit']
}

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Kitwe News Today",
    page_icon="📰",
)

# Google Cloud Storage Configuration
BUCKET_NAME = os.getenv('BUCKET')
OBJECT_KEY = os.getenv('OBJECT_KEY_PARQUET')
PROJECT_ID = os.getenv('PROJECT_ID')

# Function to download the data from Google Cloud Storage
@st.cache_data
def download_data_from_gcs(bucket_name, object_key, project_id=PROJECT_ID):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_key)
    data_bytes = blob.download_as_bytes()  # Download as binary for Parquet
    data = pd.read_parquet(BytesIO(data_bytes))  # Read the Parquet data
    return data

# Load spaCy model and BERT model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_bert_model():
    model_path = hf_hub_download(repo_id="hoybrett99/KitweFakeNewsDetector-BERT", filename="quantized_final_bert_model_complete.pth")
    model = torch.load(model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

nlp = load_spacy_model()
loaded_model, tokenizer = load_bert_model()

# Text preprocessing function
def preprocess_texts(texts):
    processed_texts = []
    for doc in nlp.pipe(texts, batch_size=50):
        filtered_text = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
        processed_texts.append(" ".join(filtered_text).strip())
    return processed_texts

# Categorization function based on keywords
def categorize_description(description):
    categories = []
    for category, keywords in categories_keywords.items():
        if any(keyword in description.lower() for keyword in keywords):
            categories.append(category)
    return ', '.join(categories) if categories else 'uncategorized'

# Define the prediction function
def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        confidence = softmax(logits, dim=1).max().item() * 100  # Confidence in percentage
        prediction = "real" if logits.argmax() == 1 else "fake"
    return prediction, confidence

# Load and process the data
try:
    df = download_data_from_gcs(BUCKET_NAME, OBJECT_KEY)

    # Getting today's date
    today = datetime.today()

    # Calculate the date four weeks ago
    one_week_ago = today - timedelta(days=28)

    # Ensure the date format is consistent and filter for recent news
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df[(df['Date'] >= one_week_ago) & (df['Date'] <= today)]
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

    # Create combined_text and processed_description columns
    df['combined_text'] = df['Headline'].fillna('') + ' ' + df['Description'].fillna('')
    df['processed_description'] = preprocess_texts(df['combined_text'])
    df['Category'] = df['Category'].apply(categorize_description)

    # Apply prediction model to generate 'prediction' and 'confidence' columns
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


    # Add a horizontal divider between news items
    st.markdown("---")

except Exception as e:
    st.error(f"Failed to load or process data: {e}")
