import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloading import get_hasib18_fns

# 1. Add custom Dataset class
class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
         
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_mlp(texts, labels, test_size=0.2, epochs=50, learning_rate=0.001):
    # 2. Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(texts).toarray()
    
    # 3. Encode string labels to integers
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # 4. Proper train/test split
    X_train, y_train = features, encoded_labels
    
    # 5. Create DataLoaders
    train_dataset = TextDataset(X_train, y_train)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Get model parameters
    input_size = X_train.shape[1]
    num_classes = len(le.classes_)
    
    model = MultiLayerPerceptron(
        input_size=input_size, 
        hidden_layers=[64, 32], 
        num_classes=num_classes
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for _ in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    
    return model, vectorizer, le

def evaluate_model(model, test_loader, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(batch_labels.numpy())
    
    # Convert numerical labels to strings
    class_names = [str(cls) for cls in label_encoder.classes_]  # Convert to strings
    y_true = label_encoder.inverse_transform(all_labels)
    y_pred = label_encoder.inverse_transform(all_preds)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix (convert to strings here too)
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    return y_true, y_pred


# Example usage
# Modified evaluation section in main block
if __name__ == "__main__":
    train_df, test_df = get_hasib18_fns(include_instruction=False)
    
    # Training data processing
    texts = train_df['text'].values
    labels = train_df['label'].values
    trained_model, vectorizer, label_encoder = train_mlp(texts, labels)

    # Save components
    torch.save(trained_model.state_dict(), 'mlp_model.pth')
    np.save('vectorizer.npy', vectorizer)
    np.save('label_encoder.npy', label_encoder)

    # Evaluation with numerical test labels
    test_texts = test_df['text'].values
    
    # Convert test labels directly to integers
    test_labels = test_df['label'].values.astype(int)  # Already 0/1
    
    # Process test features
    test_features = vectorizer.transform(test_texts).toarray()
    
    # Create dataset with numerical labels
    test_dataset = TextDataset(test_features, test_labels)  # Use raw labels
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate with numerical class mapping
    class_names = label_encoder.classes_  # Original string labels
    
    # Manually verify label alignment
    print("\nLabel verification:")
    print(f"Training label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    print(f"Test labels found: {np.unique(test_labels)}")
    
    # Modified evaluation call
    y_true, y_pred = evaluate_model(
        trained_model, 
        test_loader,
        label_encoder
    )