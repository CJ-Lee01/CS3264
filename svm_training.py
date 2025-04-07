import os
import numpy as np
import pandas as pd
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
from dataloading import get_hasib18_fns

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_data():
    train_df, test_df = get_hasib18_fns()
    
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    test_df['processed_text'] = test_df['text'].apply(preprocess_text)
    
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
    test_df['label_encoded'] = label_encoder.transform(test_df['label'])
    
    return train_df, test_df, label_encoder

def main():
    np.random.seed(42)
    
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print("Loading and preprocessing data...")
    train_df, test_df, label_encoder = load_data()
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Labels: {label_encoder.classes_}")
    
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
    
    print("\n--- Training SVM model ---")
    start_time = time.time()
    
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7)),
        ('classifier', LinearSVC())
    ])
    
    svm_pipeline.fit(train_df['processed_text'], train_df['label_encoded'])
    
    joblib.dump(svm_pipeline, os.path.join(model_dir, 'svm_model.pkl'))
    
    y_pred = svm_pipeline.predict(test_df['processed_text'])
    
    accuracy = accuracy_score(test_df['label_encoded'], y_pred)
    precision = precision_score(test_df['label_encoded'], y_pred, average='weighted')
    recall = recall_score(test_df['label_encoded'], y_pred, average='weighted')
    f1 = f1_score(test_df['label_encoded'], y_pred, average='weighted')
    
    svm_time = time.time() - start_time
    
    print(f"SVM Training Time: {svm_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    results_file = os.path.join(model_dir, 'svm_evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"SVM Training Time: {svm_time:.2f} seconds\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()