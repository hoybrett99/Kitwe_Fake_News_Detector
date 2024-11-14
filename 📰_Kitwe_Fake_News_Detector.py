import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from transformers import pipeline
import spacy
import base64
from datetime import datetime, timedelta
import feedparser
import re
import requests
import string
from PIL import Image
import numpy as np
from scipy.stats import entropy
from nltk.corpus import words
import nltk
import os
import csv
from time import sleep
from streamlit_lottie import st_lottie
import requests
import subprocess

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Download NLTK word list if not already available
nltk.download('words')
word_list = set(words.words())  # Load English dictionary words

# URL of the logo image in the GitHub repository
logo_url = "https://raw.githubusercontent.com/hoybrett99/Kitwe_Fake_News_Detector/main/kitwe_logo.png"

# Set page title and layout
st.set_page_config(page_title="Kitwe News Today", page_icon=logo_url, layout="wide")

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

# Function to fetch and encode image from GitHub
def load_logo_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    encoded_logo = base64.b64encode(response.content).decode()
    return encoded_logo

# Function to load Lottie animation from URL
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
# Load Lottie animation
lottie_animation_url = "https://lottie.host/7af44218-fb3e-4086-8ae8-f0bcae862821/VTk1cCimDv.json" #About animation
lottie_animation = load_lottie_url(lottie_animation_url)
aggregator_animation_url = "https://lottie.host/5e2852a0-10dd-47d2-ab7b-8009014263d6/a0sVUtL3po.json"  # Real time aggregator animation
aggregator_animation = load_lottie_url(aggregator_animation_url)
news_classifier_animation_url="https://lottie.host/c1acdc05-2226-4e50-86b6-5a0e276915b3/b97nRlohu6.json" #News classifier animation
news_classifier_animation=load_lottie_url(news_classifier_animation_url)
footer_animation_url = "https://lottie.host/3a39d0a5-c0c4-4f57-895b-d3653e0c58b1/nuq0AM5tcQ.json"  # footer animation
footer_animation = load_lottie_url(footer_animation_url)
sidebar_animation_url2 = "https://lottie.host/b208c76c-d287-4d92-a935-45b1ac457cf5/fTKhwkukNS.json"  # Sidebar  animation
sidebar_animation2 = load_lottie_url(sidebar_animation_url2)



# Load and encode the logo image
encoded_logo = load_logo_from_url(logo_url)

# HTML to display the Base64-encoded logo image
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
    'Zambia Reports': 'https://zambiareports.news/search/kitwe/feed/rss2/',
    'Lusaka Times': 'https://www.lusakatimes.com/search/kitwe/feed/rss2/'
}
categories_keywords = {
    'sports': ['football', 'soccer', 'basketball', 'tennis', 'cricket', 'olympics', 'athlete', 'sports', 'baseball', 'golf', 
               'swimming', 'athletics', 'rugby', 'hockey', 'marathon', 'boxing', 'racing', 'skating', 'tournament', 'league', 
               'competition', 'fitness', 'training', 'coach', 'stadium'],
    'politics': ['government', 'election', 'politician', 'policy', 'parliament', 'minister', 'president', 'vote', 'congress', 
                 'senate', 'prime minister', 'political party', 'legislation', 'governor', 'mayor', 'ambassador', 'republic', 
                 'democracy', 'diplomacy', 'law', 'constitution', 'referendum', 'campaign', 'assembly', 'foreign policy'],
    'local news': ['local', 'community', 'city', 'town', 'village', 'municipality', 'neighborhood', 'region', 'council', 
                   'development', 'traffic', 'weather', 'public safety', 'crime', 'fire department', 'local government', 
                   'district', 'local economy', 'schools', 'events', 'parks', 'transportation', 'housing', 'utilities'],
    'health': ['health', 'hospital', 'doctor', 'wellness', 'mental health', 'fitness', 'medicine', 'disease', 'clinic', 
               'nurse', 'pharmacy', 'treatment', 'public health', 'surgery', 'vaccine', 'nutrition', 'exercise', 'therapy', 
               'infection', 'emergency', 'dental', 'vision', 'recovery', 'healthcare', 'epidemic', 'pandemic', 'meditation'],
    'business': ['business', 'company', 'corporation', 'entrepreneur', 'startup', 'industry', 'investment', 'profit', 'finance', 
                 'stock', 'economy', 'merger', 'acquisition', 'revenue', 'business strategy', 'marketing', 'retail', 'e-commerce', 
                 'supply chain', 'management', 'real estate', 'market', 'trade', 'commerce', 'sales', 'customer', 'shareholder'],
    'technology': ['technology', 'tech', 'software', 'hardware', 'gadget', 'AI', 'robotics', 'innovation', 'programming', 
                   'computing', 'cybersecurity', 'data', 'machine learning', 'internet', 'blockchain', 'virtual reality', 
                   'app', 'smartphone', 'cloud', 'electronics', 'coding', 'IoT', 'automation', 'digital', 'VR', 'AR', 'networking']
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

# Enhanced garbage input check based on heuristics, dictionary matching, and entropy
def is_garbage_input(text):
    words_in_text = text.translate(str.maketrans('', '', string.punctuation)).split()
    if len(words_in_text) < 3:
        return True
    avg_word_length = np.mean([len(word) for word in words_in_text])
    if avg_word_length > 12:
        return True
    non_alpha_ratio = sum(1 for char in text if not char.isalpha()) / len(text)
    if non_alpha_ratio > 0.5:
        return True

    meaningful_words = sum(1 for word in words_in_text if word.lower() in word_list)
    if meaningful_words / len(words_in_text) < 0.3:
        return True

    char_freqs = np.array([text.count(char) for char in set(text)])
    text_entropy = entropy(char_freqs)
    if text_entropy < 3.0 or text_entropy > 5.0:
        return True

    return False

def save_prediction_to_csv(headline, description, prediction, confidence):
    # Define the CSV file path
    csv_file = "predictions_log.csv"
    
    # Check if the file exists and load existing data if it does
    if os.path.isfile(csv_file):
        try:
            # 'ISO-8859-1' encoding to avoid UnicodeDecodeError
            df = pd.read_csv(csv_file, encoding="ISO-8859-1")
            
            # Check if 'Headline' and 'Description' columns exist in the DataFrame
            if 'Headline' in df.columns and 'Description' in df.columns:
                # Check for duplicacy: either headline or description already exists
                duplicate_entry = ((df['Headline'] == headline).any() or (df['Description'] == description).any())
            else:
                duplicate_entry = False  # No duplicate if the required columns are missing
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            return
    else:
        duplicate_entry = False

    # Proceed only if it's not a duplicate entry
    if not duplicate_entry:
        try:
            # Prepare the data to be saved
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = [timestamp, headline, description, prediction, confidence]
            
            # Open the file in append mode and write data
            with open(csv_file, mode="a", newline="", encoding="ISO-8859-1") as file:
                writer = csv.writer(file)
                
                # Write headers if the file does not exist
                if not os.path.isfile(csv_file) or os.stat(csv_file).st_size == 0:
                    writer.writerow(["Timestamp", "Headline", "Description", "Prediction", "Confidence"])
                
                # Write the prediction data
                writer.writerow(data)
            
            print(f"Data saved successfully to {csv_file}.")  # Success message for confirmation
        except Exception as e:
            print(f"An error occurred while saving data to CSV: {e}")  # Error handling
    else:
        print("Duplicate entry found. Data not saved to avoid redundancy.")

        
# Prediction function with garbage detection for headline, description, and combined text
def predict_news(headline, description=""):
    with st.spinner("Predicting news authenticity..."):
        headline_is_garbage = is_garbage_input(headline)
        description_is_garbage = is_garbage_input(description) if description else False
        combined_text = headline + " " + description if description else headline
        combined_is_garbage = is_garbage_input(combined_text)

        if headline_is_garbage or description_is_garbage or combined_is_garbage:
            prediction = "fake"
            confidence = 99.0
        else:
            result = text_classifier(combined_text)[0]
            prediction = "real" if result['label'] == "LABEL_1" else "fake"
            confidence = result['score'] * 100
    
    # Save the prediction data to CSV
    save_prediction_to_csv(headline, description, prediction, confidence)
    
    return prediction, confidence


# Check if date is within last 28 days
def is_within_last_28_days(published_date_str):
    published_date = datetime(*published_date_str[:6])
    return (datetime.today() - timedelta(days=28)) <= published_date <= datetime.today()

# Get paginated feed entries
def get_feed_entries(feed_url, pages=3):
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
                date = datetime(*entry.published_parsed[:6])
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

# Function to categorize descriptions based on keywords
def categorize_description(description):
    for category, keywords in categories_keywords.items():
        if any(keyword in description.lower() for keyword in keywords):
            return category
    return 'uncategorized'


df = load_rss_data()
df['Description'] = df['Description'].apply(preprocess_text)
df['combined_text'] = df['Headline'].fillna('') + ' ' + df['Description'].fillna('')
df['processed_description'] = df['combined_text'].apply(preprocess_text_nlp)
df['Category'] = df['processed_description'].apply(categorize_description)
df[['prediction', 'confidence']] = df['processed_description'].apply(lambda x: pd.Series(predict_news(x)))

# Remaining Streamlit layout code (sidebar, menu, and footer) should follow here
# Define icons (or image URLs) for each news source
source_icons = {
    "All": "🌐 All Sources",
    "Daily Nations Zambia": "news_sources_images/dailynationszambia.png",
    "Lusaka Star": "news_sources_images/lusakastar.jpeg",
    "Lusaka Voice": "news_sources_images/lusakavoice.png",
    "Mwebantu": "news_sources_images/mwebantu.jpeg",
    "Zambia365": "news_sources_images/zambia365.jpg",
    "Zambia Eye": "news_sources_images/zambiaeye.jpeg",
    "Zambia Reports": "news_sources_images/zambiareports.jpeg",
    "Lusaka Times": "news_sources_images/lusakatimes.jpeg"
}

# Define category options with icons
category_options = {
    "All": "🌐 All",
    "Politics": "🏛️ Politics",
    "Sports": "🏅 Sports",
    "Technology": "💻 Technology",
    "Business": "💼 Business",
    "Health": "⚕️ Health",
    "Local News": "🏘️ Local News"
}

# Streamlit sidebar with logo and menu
with st.sidebar:
    if sidebar_animation2:
        st_lottie(sidebar_animation2,height=100,key="sidebar animation2")
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
        # Display category dropdown with icons
        category_keys = list(category_options.keys())
        category_sort = st.selectbox(
            "Category",
            options=category_keys,
            format_func=lambda x: category_options[x]  
        )

        # Display news source dropdown
        source_options = ["All"] + list(RSS_FEEDS.keys())
        source_sort = st.selectbox("News Source", source_options, index=0)

        with st.expander("❓ Help"):
            st.write("**Real-time News Aggregator** shows the latest news from various sources, categorized by topic.")
            st.write("**News Classifier** allows you to check the credibility of news based on headlines and descriptions.")
            st.write("Use the options in the sidebar to filter news by category or source.")

        # Display the selected source icon if available
        if source_sort != "All" and source_icons.get(source_sort):
            icon_path = source_icons[source_sort]
            try:
                source_icon_image = Image.open(icon_path)
                st.image(source_icon_image, width=50, caption=source_sort)  # Display icon with source name as caption
            except FileNotFoundError:
                st.write("Icon not found for the selected source.")


    elif selected == "News Classifier":
        st.markdown("<h3 style='color: #FF6F61;'>Addressing Kitwe's Fake News Problem</h3>", unsafe_allow_html=True)
        st.write("This tool helps Kitwe residents identify credible news. Let's reduce misinformation and promote a well-informed community.")

# Content display for each page
if selected == "Real-time News Aggregator":
    if aggregator_animation:
        st_lottie(aggregator_animation, height=200, key="aggregator_animation")
    st.title("Welcome to the Real-time News Aggregator")
    st.subheader("Latest News")
    if category_sort != "All":
        filtered_df = df[df['Category'] == category_sort.lower()]
    else:
        filtered_df = df
    if source_sort != "All":
        filtered_df = filtered_df[filtered_df['Source'] == source_sort]
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

# News Classifier interface
elif selected == "News Classifier":
    if news_classifier_animation:
        st_lottie(news_classifier_animation, height=200, key="news_classifier_animation")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔍 News Classifier</h1>", unsafe_allow_html=True)
    st.write("Welcome to the News Classifier! Enter a headline and an optional description to get insights on the type of news.")
    st.subheader("📝 Input News Details")
    headline = st.text_input("Enter News Headline", key="headline_input")
    
    if headline:
        # Optional description input
        description = st.text_input("Enter News Description (Optional)", key="description_input")
        st.markdown("""
        <div style="background-color: #f0f9ff; padding: 10px; border-radius: 8px; display: flex; align-items: center; margin-top: 5px; margin-bottom: 15px;">
            <img src="https://img.icons8.com/color/48/000000/info--v1.png" alt="info icon" width="24" height="24" style="margin-right: 10px;">
            <p style="margin: 0; font-size: 14px; color: #007BFF;">Enter a detailed description for better classification results.</p>
        </div>
        """, unsafe_allow_html=True)

        # Prediction only with headline or headline + description
        if st.button("Classify News"):
            # Display a loading spinner during model processing
            with st.spinner("Evaluating the input... Please wait."):
                sleep(2)  
                
                # Perform the prediction
                prediction, confidence = predict_news(headline, description)
                
            # Display prediction results
            prediction_color = "green" if prediction == "real" else "red"
            st.markdown(
                f"**Prediction**: <span style='color:{prediction_color}'>{prediction.capitalize()}</span> "
                f"({confidence:.2f}%)",
                unsafe_allow_html=True
            )

# Add About section at the bottom of the page
st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line separator
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>About Kitwe News Classifier & Aggregator</h2>", unsafe_allow_html=True)

# Display Lottie animation
if lottie_animation:
    st_lottie(lottie_animation, height=200, key="news_animation")

# Description and interactive icons for About section
st.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
        <p style="font-size: 16px; max-width: 600px; color: #555;">
            The <strong>Kitwe News Classifier & Aggregator</strong> serves as a real-time resource to help citizens stay updated on 
            authentic and relevant news. By classifying news accurately, our goal is to help the community discern real news from fake, 
            ensuring a trustworthy news experience.
        </p>
        <div style="display: flex; gap: 30px; margin-top: 15px;">
            <div style="text-align: center;">
                <img src="https://img.icons8.com/color/48/000000/news.png" alt="Real-Time Icon" />
                <p style="font-size: 14px; color: #4CAF50;">Real-Time</p>
            </div>
            <div style="text-align: center;">
                <img src="https://img.icons8.com/color/48/000000/artificial-intelligence.png" alt="AI-Powered Icon" />
                <p style="font-size: 14px; color: #4CAF50;">AI-Powered</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Footer Section
st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line separator

with st.container():
    col1, col2 = st.columns([1, 3])

    # Footer Animation on the left
    with col1:
        if footer_animation:
            st_lottie(footer_animation, height=120, key="footer_animation")

    # Footer Text Content on the right
    with col2:
        st.markdown(
            """
            <div style="text-align: left; color: #555;">
                <p><strong>Disclaimer:</strong> This model may occasionally misclassify news. It is designed as a supportive tool, and we encourage users to apply personal judgment.</p>
                <p>&copy; 2024 Kitwe News Aggregator. All rights reserved.</p>
                <p>Contact: support@kitwenews.com | Privacy Policy | Terms of Use</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
