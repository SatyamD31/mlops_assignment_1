import optuna
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
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

def extract_features(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    y = df['category']
    return X, y, tfidf_vectorizer

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, alpha=1.0):
    nb_classifier = MultinomialNB(alpha=alpha)
    nb_classifier.fit(X_train, y_train)
    return nb_classifier

def objective(trial):
    # Hyperparameter search space
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e1)  # log-uniform distribution for alpha

    # Load and preprocess the data
    filepath = 'C:/Users/vidis/Downloads/mlops_assignment_1-feat-satyam/mlops_assignment_1-feat-satyam/data/Twitter_Data.csv'
    df = load_and_preprocess_data(filepath)
    X, y, tfidf_vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model with current alpha
    model = train_model(X_train, y_train, alpha)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics with MLflow
    with mlflow.start_run():
        mlflow.log_params({'alpha': alpha})
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        # Log model
        mlflow.sklearn.log_model(model, "naive_bayes_model")
        mlflow.log_artifact(filepath, artifact_path="tables")
    
    return accuracy  # Optuna optimizes the objective function (maximize accuracy)

if __name__ == "__main__":
    # Create an Optuna study for optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  # Run optimization for 10 trials

    # Print best hyperparameters and accuracy
    print(f"BEST HYPERPARAMS: {study.best_params}")
    print(f"BEST ACCURACY: {study.best_value}")

