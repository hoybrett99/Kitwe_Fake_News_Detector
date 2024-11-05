import pandas as pd
import streamlit as st
from io import StringIO
import os
from transformers import pipeline
import spacy
from datetime import datetime, timedelta
import feedparser
import re

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Kitwe News Today",
    page_icon="📰",
)

# Define keywords for each category
categories_keywords = {
    'sports': ['football', 'soccer', 'basketball', 'tennis', 'cricket', 'olympics', 'athlete', 'sports'],
    'politics': ['government', 'election', 'politician', 'policy', 'parliament', 'minister', 'president', 'vote'],
    'education': ['school', 'university', 'education', 'college', 'students', 'learning', 'teacher', 'scholarship'],
    'health and wellness': ['health', 'hospital', 'doctor', 'wellness', 'mental health', 'fitness', 'medicine', 'disease'],
    'development': ['development', 'infrastructure', 'construction', 'road', 'bridge', 'building', 'urbanization'],
    'narcotics': ['narcotics', 'drug', 'cocaine', 'heroin', 'meth', 'drug trafficking', 'illegal drugs'],
    'fashion': ['fashion', 'clothing', 'designer', 'runway', 'model', 'style', 'apparel', 'trends'],
    'local news': ['local', 'community', 'city', 'town', 'village', 'municipality', 'neighborhood', 'region', ],
    'economy news': ['economy', 'economic', 'finance', 'market', 'stocks', 'currency', 'inflation', 'gdp'],
    'business news': ['business', 'company', 'corporation', 'entrepreneur', 'startup', 'industry', 'investment', 'profit']
}

# Define the RSS feeds
RSS_FEEDS = {
    'Daily Nations Zambia': 'https://dailynationzambia.com/search/kitwe/feed/rss2/',
    'Lusaka Star': 'https://lusakastar.com/search/kitwe/feed/rss2/',
    'Lusaka Voice': 'https://lusakavoice.com/search/kitwe/feed/rss2/',
    'Mwebantu': 'https://www.mwebantu.com/search/kitwe/feed/rss2/',
    'Zambia365': 'https://zambianews365.com/search/kitwe/feed/rss2/',
    'Zambia Eye': 'https://zambianeye.com/search/kitwe/feed/rss2/',
    'Zambia Reports': 'https://zambiareports.news/search/kitwe/feed/rss2/'
}

# Cache the spaCy model loading
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Cache the Hugging Face pipeline
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

# Categorization function based on keywords (ensuring only one category is chosen)
def categorize_description(description):
    for category, keywords in categories_keywords.items():
        if any(keyword in description.lower() for keyword in keywords):
            return category  # Return the first matching category
    return 'uncategorized'  # Default if no match is found

# Prediction function using Hugging Face pipeline
def predict_news(text):
    result = text_classifier(text)[0]
    label = result['label']
    confidence = result['score'] * 100  # Confidence in percentage

    # Map LABEL_1 to "real" and LABEL_2 to "fake"
    prediction = "real" if label == "LABEL_1" else "fake"
    return prediction, confidence


# Helper function to check if a date is within the last 28 days
def is_within_last_28_days(published_date_str):
    try:
        published_date = datetime(*published_date_str[:6])  # Convert to datetime
        today = datetime.today()
        cutoff_date = today - timedelta(days=28)
        return cutoff_date <= published_date <= today
    except Exception as e:
        print(f"Date parsing error: {e}")
        return False

# Text preprocessing function
def preprocess_text(text):
    if not text:
        return ""
    
    # 1. Remove HTML tags
    cleaned_text = re.sub(r'<.*?>', '', text)

    # 2. Extract terms within square brackets
    bracketed_terms = re.findall(r'\[(.*?)\]', cleaned_text)

    # 3. Remove square brackets from the text but keep the terms inside them
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)

    # 4. Combine the main cleaned text with extracted bracketed terms
    combined_text = cleaned_text.strip() + ' ' + ' '.join(bracketed_terms)
    
    return combined_text.strip()

# Cache the RSS feed data collection
@st.cache_data(ttl=3600)  # Cache with a time-to-live of 1 hour
def load_rss_data():
    data = []
    for source_name, feed_url in RSS_FEEDS.items():
        entries = get_feed_entries(feed_url, pages=1)
        for entry in entries:
            if 'published_parsed' in entry and is_within_last_28_days(entry.published_parsed):
                link = entry.link
                date = datetime(*entry.published_parsed[:6]).strftime('%d/%m/%Y')
                description = entry.summary if 'summary' in entry else 'N/A'
                headline = entry.title if 'title' in entry else 'N/A'
                category = ', '.join(tag.term for tag in entry.tags) if 'tags' in entry and entry.tags else 'N/A'
                data.append({
                    'Source': source_name,
                    'Category': category,
                    'Headline': headline,
                    'Link': link,
                    'Description': description,
                    'Date': date,
                })
    return pd.DataFrame(data)

# Function to handle paginated feed entries
def get_feed_entries(feed_url, pages=1):
    all_entries = []
    for page in range(1, pages + 1):
        paged_url = f"{feed_url}?paged={page}"
        parsed_feed = feedparser.parse(paged_url)
        entries = parsed_feed.entries
        if not entries:
            break  # Exit if there are no more entries
        all_entries.extend(entries)
    return all_entries

# Load and process the data
df = load_rss_data()

# Create combined_text and processed_description columns
df['Description'] = df['Description'].apply(preprocess_text)
df['combined_text'] = df['Headline'].fillna('') + ' ' + df['Description'].fillna('')
df['processed_description'] = df['combined_text'].apply(preprocess_text_nlp)
df['Category'] = df['processed_description'].apply(categorize_description)

# Apply prediction model to generate 'prediction' and 'confidence' columns
df[['prediction', 'confidence']] = df['processed_description'].apply(lambda x: pd.Series(predict_news(x)))

# Displaying the data in Streamlit
st.title("Kitwe News Today")
st.subheader("Latest News")

# Dropdown for selecting categories
unique_categories = df['Category'].unique()
selected_categories = st.multiselect("Filter by Category", unique_categories, default=unique_categories)

# Filter the DataFrame by the selected categories
filtered_df = df[df['Category'].isin(selected_categories)]

# Display each news item
for _, row in filtered_df.iterrows():
    st.markdown(f"### [{row['Headline']}]({row['Link']})")
    st.markdown(f"**Source**: {row['Source']}  |  **Category**: {row['Category']} | {row['Date']}")
    st.write(row['Description'])

    prediction_color = "green" if row['prediction'] == "real" else "red"
    st.markdown(
        f"**Prediction**: <span style='color:{prediction_color}'>{row['prediction'].capitalize()}</span> "
        f"({row['confidence']:.2f}%)",
        unsafe_allow_html=True
    )
    st.markdown("---")
