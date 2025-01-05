import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Define the preprocess_text function
stop_words = set(stopwords.words('english'))


def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Check for missing or NaN values in 'clean_text' and 'category' columns
    df = df.dropna(subset=['clean_text', 'category'])

    # Ensure all values in 'clean_text' are strings and clean invalid data
    df['clean_text'] = df['clean_text'].astype(str)
    df['category'] = df['category'].astype(float)

    # Apply preprocessing to the 'clean_text' column
    df['cleaned_text'] = df['clean_text'].apply(preprocess_text)

    return df


def preprocess_text(text):
    # Remove URLs, mentions, hashtags, and special characters using regular expressions
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)  # Remove special characters
    text = text.lower()  # Convert text to lowercase
    text = word_tokenize(text)  # Tokenize the text
    text = [word for word in text if word not in stop_words]  # Remove stop words
    return ' '.join(text)
