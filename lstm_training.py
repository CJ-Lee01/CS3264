import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
from torch.nn.utils.rnn import pad_sequence
from dataloading import get_hasib18_fns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = word_tokenize(self.texts[idx])[:self.max_length]
        indices = [self.vocab.get(token, 1) for token in tokens]  # 1 is <UNK>
        return torch.tensor(indices), self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.tensor(labels), lengths

# LSTM model
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
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), 
                                                         batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        return self.fc(hidden)

def train_lstm(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        texts, labels, lengths = batch
        texts = texts.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts, lengths)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == labels).float()
        accuracy = correct.sum() / len(correct)
        epoch_acc += accuracy.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_lstm(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in iterator:
            texts, labels, lengths = batch
            texts = texts.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            predictions = model(texts, lengths)
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            
            _, predicted = torch.max(predictions, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct = (predicted == labels).float()
            accuracy = correct.sum() / len(correct)
            epoch_acc += accuracy.item()
            
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), accuracy, precision, recall, f1

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print("Loading and preprocessing data...")
    train_df, test_df, label_encoder = load_data()
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Labels: {label_encoder.classes_}")
    
    # Save label encoder
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder_lstm.pkl'))
    
    print("\n--- Training LSTM model ---")
    start_time = time.time()
    
    # Build vocabulary
    tokens = []
    for text in train_df['processed_text']:
        tokens.extend(word_tokenize(text))
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token in set(tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    
    # Save vocabulary
    joblib.dump(vocab, os.path.join(model_dir, 'vocab_lstm.pkl'))
    
    train_dataset = NewsDataset(train_df['processed_text'].tolist(), 
                               train_df['label_encoded'].tolist(), 
                               vocab)
    test_dataset = NewsDataset(test_df['processed_text'].tolist(), 
                              test_df['label_encoded'].tolist(), 
                              vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(label_encoder.classes_)
    N_LAYERS = 2
    DROPOUT = 0.5
    
    model_config = {
        'INPUT_DIM': INPUT_DIM,
        'EMBEDDING_DIM': EMBEDDING_DIM,
        'HIDDEN_DIM': HIDDEN_DIM,
        'OUTPUT_DIM': OUTPUT_DIM,
        'N_LAYERS': N_LAYERS,
        'DROPOUT': DROPOUT
    }
    joblib.dump(model_config, os.path.join(model_dir, 'lstm_config.pkl'))
    
    model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    N_EPOCHS = 10
    best_val_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_lstm(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, accuracy, precision, recall, f1 = evaluate_lstm(model, test_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_lstm_model.pt'))
        
        print(f"Epoch: {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_lstm_model.pt'))
    
    torch.save(model, os.path.join(model_dir, 'complete_lstm_model.pt'))
    
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_lstm_model.pt')))
    
    val_loss, val_acc, accuracy, precision, recall, f1 = evaluate_lstm(model, test_loader, criterion, device)
    
    lstm_time = time.time() - start_time
    
    print(f"\nLSTM Training Time: {lstm_time:.2f} seconds")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final Precision: {precision:.4f}")
    print(f"Final Recall: {recall:.4f}")
    print(f"Final F1 Score: {f1:.4f}")

    results_file = os.path.join(model_dir, 'lstm_evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"LSTM Training Time: {lstm_time:.2f} seconds\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()