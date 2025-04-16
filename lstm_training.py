import os
import numpy as np
import pandas as pd
import torch
import joblib
import argparse
from nltk.tokenize import word_tokenize
import re
import nltk
from nltk.corpus import stopwords
from dataloading import get_hasib18_fns

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Text preprocessing function (same as in training)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back to string
    return ' '.join(tokens)

def load_dataset():
    """Load the original dataset used for training"""
    train_df, test_df = get_hasib18_fns()
    
    # Preprocess text
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    test_df['processed_text'] = test_df['text'].apply(preprocess_text)
    
    # Concatenate for full dataset
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    return full_df

def run_inference(model_type, index=None):
    """
    Run inference on a specific entry or all entries
    
    Args:
        model_type: 'gru', 'lstm', or 'svm'
        index: Specific index to run inference on (optional)
    """
    model_dir = 'models'
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset()
    
    # Check if index is valid
    if index is not None:
        if index < 0 or index >= len(dataset):
            print(f"Error: Index {index} is out of bounds. Dataset has {len(dataset)} entries.")
            return
    
    # Load appropriate model based on model_type
    if model_type.lower() == 'svm':
        # Load SVM model
        try:
            # Try loading GPU model first
            model_package = joblib.load(os.path.join(model_dir, 'gpu_svm_model.pkl'))
            vectorizer = model_package['vectorizer']
            model = model_package['model']
            print("Loaded GPU SVM model")
        except:
            # Fall back to CPU model
            model = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
            print("Loaded CPU SVM model")
        
        # Load label encoder
        label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        
        # Function to get prediction
        def get_prediction(text):
            processed = preprocess_text(text)
            
            # Handle different model types
            if isinstance(model, dict):  # GPU model package
                features = vectorizer.transform([processed])
                pred_idx = model['model'].predict(features)[0]
            else:  # Pipeline or other sklearn model
                pred_idx = model.predict([processed])[0]
            
            return label_encoder.inverse_transform([pred_idx])[0]
    
    elif model_type.lower() in ['gru', 'lstm']:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load appropriate files for the model type
        model_name = model_type.lower()
        
        # Load label encoder
        label_encoder = joblib.load(os.path.join(model_dir, f'label_encoder{"_lstm" if model_name == "lstm" else ""}.pkl'))
        
        # Load vocabulary
        vocab = joblib.load(os.path.join(model_dir, f'vocab{"_lstm" if model_name == "lstm" else ""}.pkl'))
        
        # Load model
        try:
            # Try loading complete model
            model = torch.load(os.path.join(model_dir, f'complete_{model_name}_model.pt'))
            print(f"Loaded complete {model_name.upper()} model")
        except:
            # Fall back to loading config and weights
            config = joblib.load(os.path.join(model_dir, f'{model_name}_config.pkl'))
            
            if model_name == 'gru':
                from torch import nn
                
                class GRUClassifier(nn.Module):
                    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
                        super().__init__()
                        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, 
                                        bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0)
                        self.fc = nn.Linear(hidden_dim * 2, output_dim)
                        self.dropout = nn.Dropout(dropout)
                        
                    def forward(self, text, text_lengths):
                        embedded = self.dropout(self.embedding(text))
                        
                        # Pack sequence
                        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), 
                                                                        batch_first=True, enforce_sorted=False)
                        
                        packed_output, hidden = self.gru(packed_embedded)
                        
                        # Concatenate the final forward and backward hidden states
                        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
                        
                        return self.fc(hidden)
                
                model = GRUClassifier(
                    config['INPUT_DIM'], 
                    config['EMBEDDING_DIM'],
                    config['HIDDEN_DIM'],
                    config['OUTPUT_DIM'],
                    config['N_LAYERS'],
                    config['DROPOUT']
                )
            else:  # LSTM
                from torch import nn
                
                class LSTMClassifier(nn.Module):
                    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
                        super().__init__()
                        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                                        bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0)
                        self.fc = nn.Linear(hidden_dim * 2, output_dim)
                        self.dropout = nn.Dropout(dropout)
                        
                    def forward(self, text, text_lengths):
                        embedded = self.dropout(self.embedding(text))
                        
                        # Pack sequence
                        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), 
                                                                        batch_first=True, enforce_sorted=False)
                        
                        packed_output, (hidden, cell) = self.lstm(packed_embedded)
                        
                        # Concatenate the final forward and backward hidden states
                        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
                        
                        return self.fc(hidden)
                
                model = LSTMClassifier(
                    config['INPUT_DIM'], 
                    config['EMBEDDING_DIM'],
                    config['HIDDEN_DIM'],
                    config['OUTPUT_DIM'],
                    config['N_LAYERS'],
                    config['DROPOUT']
                )
            
            # Load weights
            model.load_state_dict(torch.load(os.path.join(model_dir, f'best_{model_name}_model.pt')))
            print(f"Loaded {model_name.upper()} model from config and weights")
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Function to get prediction
        def get_prediction(text):
            processed = preprocess_text(text)
            tokens = word_tokenize(processed)
            
            # Convert tokens to indices
            indices = [vocab.get(token, 1) for token in tokens]  # 1 is <UNK>
            
            # Convert to tensor
            text_tensor = torch.tensor(indices).unsqueeze(0).to(device)
            length_tensor = torch.tensor([len(indices)]).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(text_tensor, length_tensor)
                predicted_idx = torch.argmax(output, dim=1).item()
            
            return label_encoder.inverse_transform([predicted_idx])[0]
    
    else:
        print(f"Error: Unknown model type '{model_type}'. Use 'gru', 'lstm', or 'svm'.")
        return
    
    # Run inference on specific entry or all entries
    if index is not None:
        # Get the specified entry
        entry = dataset.iloc[index]
        
        # Get prediction
        prediction = get_prediction(entry['text'])
        
        # Display result
        print("\n" + "="*80)
        print(f"Entry #{index}:")
        print("="*80)
        print(f"Text: {entry['text']}")
        print("-"*80)
        print(f"Actual label: {entry['label']}")
        print(f"Predicted label: {prediction}")
        print(f"Prediction {'correct' if prediction == entry['label'] else 'incorrect'}")
        print("="*80)
    else:
        # Run inference on all entries (first 5 for demo)
        print("\nRunning inference on first 5 entries:")
        for i in range(min(5, len(dataset))):
            entry = dataset.iloc[i]
            prediction = get_prediction(entry['text'])
            
            print("\n" + "-"*80)
            print(f"Entry #{i}:")
            print(f"Text: {entry['text'][:100]}..." if len(entry['text']) > 100 else entry['text'])
            print(f"Actual label: {entry['label']}")
            print(f"Predicted label: {prediction}")
            print(f"Prediction {'correct' if prediction == entry['label'] else 'incorrect'}")
        
        print("\nUse --index parameter to specify a particular entry.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on trained models')
    parser.add_argument('--model', type=str, required=True, choices=['gru', 'lstm', 'svm'], 
                        help='Model type to use for inference')
    parser.add_argument('--index', type=int, help='Index of the entry to run inference on')
    
    args = parser.parse_args()
    
    run_inference(args.model, args.index)