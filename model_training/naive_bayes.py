import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloading import get_hasib18_fns

def naive_bayes_text_classifier(texts, labels,new_texts,actual_labels):
    """
    Perform text classification using Multinomial Naive Bayes
    
    Args:
        texts (list): Input text strings
        labels (list): Corresponding labels for texts
    
    Returns:
        tuple: Trained model, classification report, and confusion matrix
    """
    # Split the data into training and testing sets
    
    # Create a pipeline that combines vectorization and Naive Bayes
    text_clf = Pipeline([
        # Convert text to word count vectors
        ('vectorizer', CountVectorizer(
            # Optional parameters to tune feature extraction
            stop_words='english',  # Remove common English stop words
            max_features=5000,     # Limit number of features
            ngram_range=(1, 2)     # Consider both unigrams and bigrams
        )),
        # Use Multinomial Naive Bayes classifier
        ('classifier', MultinomialNB(
            # Optional parameters to tune the classifier
            alpha=1.0,  # Smoothing parameter
        ))
    ])
    
    # Train the model
    text_clf.fit(texts, labels)
    
    # Make predictions
    y_pred = text_clf.predict(new_texts)
    
    # Generate classification report
    report = classification_report(actual_labels, y_pred)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(actual_labels, y_pred)
    
    return text_clf, report, conf_matrix

def predict_new_text(model, new_texts):
    """
    Predict labels for new texts using the trained model
    
    Args:
        model: Trained Naive Bayes pipeline
        new_texts (list): New text strings to classify
    
    Returns:
        list: Predicted labels
    """
    return model.predict(new_texts)

# Example usage
if __name__ == "__main__":
    # Sample text classification dataset
    train_df, test_df = get_hasib18_fns(include_instruction=False)
    texts = train_df['text'].values
    labels = train_df['label'].values
   
    new_texts = test_df['text'].values
    actual_labels = test_df['label'].values
    
    # Train the Naive Bayes model
    model, classification_report, confusion_matrix = naive_bayes_text_classifier(texts, labels,new_texts,actual_labels)
    
    # Print classification metrics
    print("Classification Report:")
    print(classification_report)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    
    # Example of predicting new texts
    
    predictions = predict_new_text(model, new_texts)
    
    # print("\nNew Text Predictions:")
    # for text, prediction, actual in zip(new_texts, predictions, actual_labels):
    #     print(f"Text: '{text}' -> Predicted Label: {prediction}, Actual Label: {actual}")