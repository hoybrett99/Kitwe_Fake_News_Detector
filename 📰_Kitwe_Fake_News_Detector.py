import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from transformers import pipeline
import spacy
import base64
from datetime import datetime, timedelta
import feedparser
import re

# Set page title and layout
st.set_page_config(page_title="Kitwe News Today", page_icon="📰", layout="wide")

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Load Hugging Face pipeline
@st.cache_resource
def load_huggingface_pipeline():
    return pipeline("text-classification", model="Subash2580/Bert_model_news_aggregator")

text_classifier = load_huggingface_pipeline()

# Load and encode image for logo
def load_logo(path):
    with open(path, "rb") as image_file:
        encoded_logo = base64.b64encode(image_file.read()).decode()
    return encoded_logo

logo_path = "D:/Data_engineering/kitwe-news-datapipeline/kitwe_logo.png"
encoded_logo = load_logo(logo_path)
logo_html = f"""
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{encoded_logo}" style="width:150px; height:150px; border-radius:50%; object-fit:cover;">
    </div>
"""

# RSS Feeds and categories
RSS_FEEDS = {
    'Daily Nations Zambia': 'https://dailynationzambia.com/search/kitwe/feed/rss2/',
    'Lusaka Star': 'https://lusakastar.com/search/kitwe/feed/rss2/',
    'Lusaka Voice': 'https://lusakavoice.com/search/kitwe/feed/rss2/',
    'Mwebantu': 'https://www.mwebantu.com/search/kitwe/feed/rss2/',
    'Zambia365': 'https://zambianews365.com/search/kitwe/feed/rss2/',
    'Zambia Eye': 'https://zambianeye.com/search/kitwe/feed/rss2/',
    'Zambia Reports': 'https://zambiareports.news/search/kitwe/feed/rss2/'
}
categories_keywords = {
    'sports': [
        'football', 'soccer', 'basketball', 'tennis', 'cricket', 'olympics', 'athlete', 'sports', 'baseball', 'golf', 
        'swimming', 'athletics', 'rugby', 'hockey', 'marathon', 'boxing', 'racing', 'skating', 'tournament', 'league', 
        'competition', 'fitness', 'training', 'coach', 'stadium'
    ],

    'politics': [
        'government', 'election', 'politician', 'policy', 'parliament', 'minister', 'president', 'vote', 'congress', 
        'senate', 'prime minister', 'political party', 'legislation', 'governor', 'mayor', 'ambassador', 'republic', 
        'democracy', 'diplomacy', 'law', 'constitution', 'referendum', 'campaign', 'assembly', 'foreign policy'
    ],

    'local news': [
        'local', 'community', 'city', 'town', 'village', 'municipality', 'neighborhood', 'region', 'council', 
        'development', 'traffic', 'weather', 'public safety', 'crime', 'fire department', 'local government', 
        'district', 'local economy', 'schools', 'events', 'parks', 'transportation', 'housing', 'utilities'
    ],

    'health': [
        'health', 'hospital', 'doctor', 'wellness', 'mental health', 'fitness', 'medicine', 'disease', 'clinic', 
        'nurse', 'pharmacy', 'treatment', 'public health', 'surgery', 'vaccine', 'nutrition', 'exercise', 'therapy', 
        'infection', 'emergency', 'dental', 'vision', 'recovery', 'healthcare', 'epidemic', 'pandemic', 'meditation'
    ],

    'business': [
        'business', 'company', 'corporation', 'entrepreneur', 'startup', 'industry', 'investment', 'profit', 'finance', 
        'stock', 'economy', 'merger', 'acquisition', 'revenue', 'business strategy', 'marketing', 'retail', 'e-commerce', 
        'supply chain', 'management', 'real estate', 'market', 'trade', 'commerce', 'sales', 'customer', 'shareholder'
    ],

    'technology': [
        'technology', 'tech', 'software', 'hardware', 'gadget', 'AI', 'robotics', 'innovation', 'programming', 
        'computing', 'cybersecurity', 'data', 'machine learning', 'internet', 'blockchain', 'virtual reality', 
        'app', 'smartphone', 'cloud', 'electronics', 'coding', 'IoT', 'automation', 'digital', 'VR', 'AR', 'networking'
    ]

}

# Define text preprocessing functions
def preprocess_text(text):
    if not text:
        return ""
    cleaned_text = re.sub(r'<.*?>', '', text)
    bracketed_terms = re.findall(r'\[(.*?)\]', cleaned_text)
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
    return (cleaned_text.strip() + ' ' + ' '.join(bracketed_terms)).strip()

def preprocess_text_nlp(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop]).strip()

# Categorize based on keywords
def categorize_description(description):
    for category, keywords in categories_keywords.items():
        if any(keyword in description.lower() for keyword in keywords):
            return category
    return 'uncategorized'

# Prediction function
def predict_news(text):
    result = text_classifier(text)[0]
    prediction = "real" if result['label'] == "LABEL_1" else "fake"
    return prediction, result['score'] * 100

# Check if date is within last 28 days
def is_within_last_28_days(published_date_str):
    published_date = datetime(*published_date_str[:6])
    return (datetime.today() - timedelta(days=28)) <= published_date <= datetime.today()

# Get paginated feed entries
def get_feed_entries(feed_url, pages=1):
    all_entries = []
    for page in range(1, pages + 1):
        parsed_feed = feedparser.parse(f"{feed_url}?paged={page}")
        entries = parsed_feed.entries
        if not entries:
            break
        all_entries.extend(entries)
    return all_entries

# Load and process RSS feed data
@st.cache_data(ttl=3600)
def load_rss_data():
    data = []
    for source_name, feed_url in RSS_FEEDS.items():
        entries = get_feed_entries(feed_url, pages=1)
        for entry in entries:
            if 'published_parsed' in entry and is_within_last_28_days(entry.published_parsed):
                link = entry.link
                date = datetime(*entry.published_parsed[:6])  # Convert directly to datetime
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

df = load_rss_data()
df['Description'] = df['Description'].apply(preprocess_text)
df['combined_text'] = df['Headline'].fillna('') + ' ' + df['Description'].fillna('')
df['processed_description'] = df['combined_text'].apply(preprocess_text_nlp)
df['Category'] = df['processed_description'].apply(categorize_description)
df[['prediction', 'confidence']] = df['processed_description'].apply(lambda x: pd.Series(predict_news(x)))

# Streamlit sidebar with logo and menu
with st.sidebar:
    st.markdown(logo_html, unsafe_allow_html=True)
    selected = option_menu(
        menu_title="Navigation",
        options=["Real-time News Aggregator", "News Classifier"],
        icons=["newspaper", "robot"],
        menu_icon="menu-up",
        default_index=0,
    )
    if selected == "Real-time News Aggregator":
        st.subheader("Sort by")
        
        # Category filter
        category_options = ["All", "Politics", "Sports", "Technology", "Business", "Health", "Local News"]
        category_sort = st.selectbox("Category", category_options, index=0)
        
        # News Source filter
        source_options = ["All"] + list(RSS_FEEDS.keys())
        source_sort = st.selectbox("News Source", source_options, index=0)

    elif selected == "News Classifier":
        st.markdown("<h3 style='color: #FF6F61;'>Addressing Kitwe's Fake News Problem</h3>", unsafe_allow_html=True)
        st.write("""
            This tool helps Kitwe residents identify credible news. Let's reduce misinformation and promote a well-informed community.
        """)

# Content display for each page
if selected == "Real-time News Aggregator":
    st.title("Welcome to the Real-time News Aggregator")
    st.subheader("Latest News")

    # Apply category filter if selected
    if category_sort != "All":
        filtered_df = df[df['Category'] == category_sort.lower()]
    else:
        filtered_df = df

    # Apply news source filter if selected
    if source_sort != "All":
        filtered_df = filtered_df[filtered_df['Source'] == source_sort]

    # Sort by Date in descending order
    filtered_df = filtered_df.sort_values(by='Date', ascending=False)
    
    for _, row in filtered_df.iterrows():
        st.markdown(f"### [{row['Headline']}]({row['Link']})")
        st.markdown(f"**Source**: {row['Source']}  |  **Category**: {row['Category']} | {row['Date'].strftime('%d/%m/%Y')}")
        st.write(row['Description'])
        prediction_color = "green" if row['prediction'] == "real" else "red"
        st.markdown(
            f"**Prediction**: <span style='color:{prediction_color}'>{row['prediction'].capitalize()}</span> "
            f"({row['confidence']:.2f}%)",
            unsafe_allow_html=True
        )
        st.markdown("---")

elif selected == "News Classifier":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔍 News Classifier</h1>", unsafe_allow_html=True)
    st.subheader("📝 Input News Details")
    headline = st.text_input("Enter News Headline", key="headline_input")
    if headline:
        description = st.text_input("Enter News Description (Optional)", key="description_input")
        if st.button("Classify News"):
            prediction, confidence = predict_news(headline + " " + description)
            prediction_color = "green" if prediction == "real" else "red"
            st.markdown(
                f"**Prediction**: <span style='color:{prediction_color}'>{prediction.capitalize()}</span> "
                f"({confidence:.2f}%)",
                unsafe_allow_html=True
            )

# Footer
footer_html = """
    <style>
        .footer-line { width: 100%; height: 2px; background-color: #cccccc; margin-top: 20px; }
        .footer { text-align: center; padding: 10px; font-size: 14px; color: white; }
    </style>
    <div class="footer-line"></div>
    <div class="footer">
        <p>© 2024 Kitwe_News_aggregator. All rights reserved | 
        <a href="https://example.com" target="_blank" style="color: #ffffff; text-decoration: underline;">Privacy Policy</a></p>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
