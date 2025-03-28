
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloading import get_hasib18_fns


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        """
        Multi-Layer Perceptron for text classification
        
        Args:
            input_size (int): Number of input features
            hidden_layers (list): List of neuron counts for hidden layers
            num_classes (int): Number of output classes
        """
        super(MultiLayerPerceptron, self).__init__()
        
        # Create layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # Prevent overfitting
            prev_size = hidden_size
         
        # Final classification layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_mlp(texts, labels, test_size=0.2, epochs=50, learning_rate=0.001):
    """
    Train a Multi-Layer Perceptron for text classification
    
    Args:
        texts (list): Input text strings
        labels (list): Corresponding labels
        test_size (float): Proportion of data for testing
        epochs (int): Number of training epochs
        learning_rate (float): Optimizer learning rate
    """
    # Split data
    train_df, test_df = get_hasib18_fns(include_instruction=True) 
    
    # Determine input and output sizes
    input_size = train_dataset.features.shape[1]
    num_classes = len(np.unique(labels))
    
    # Initialize model
    model = MultiLayerPerceptron(
        input_size=input_size, 
        hidden_layers=[64, 32], 
        num_classes=num_classes
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Loss: {total_loss/len(train_loader):.4f}, '
              f'Accuracy: {100 * correct / total:.2f}%')
    
    return model

# Example usage
if __name__ == "__main__":
    # Sample text classification dataset
    texts = [
        "I love this movie", 
        "This film is amazing", 
        "Terrible movie, waste of time",
        "Horrible film, did not enjoy",
        "Great acting and direction"
    ]
    labels = [
        "positive", 
        "positive", 
        "negative", 
        "negative", 
        "positive"
    ]
    
    # Train the model
    trained_model = train_mlp(texts, labels)
