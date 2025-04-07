import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloading import get_multilingual_dataset

# -----------------------
# Load HuggingFace tokenizer
print("🔤 Loading AutoTokenizer...")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# -----------------------
# Custom Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        return input_ids, torch.tensor(self.labels[idx])

# -----------------------
# RNN model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        out = self.fc(hidden.squeeze(0))
        return out

# -----------------------
# Training function

def train_rnn(total_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    train_df, test_df, val_df = get_multilingual_dataset()

    # Set tokenizer vocab size
    vocab_size = tokenizer.vocab_size

    # Create datasets
    train_dataset = NewsDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len=50)
    test_dataset = NewsDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len=50)
    val_dataset = NewsDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_len=50)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model, loss, optimizer
    model = RNNClassifier(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint if exists
    checkpoint_path = "checkpoint.pt"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("🔁 Found checkpoint! Resuming training...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resuming from epoch {start_epoch}")

    # Training loop
    val_f1 = []
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        for x_batch, y_batch in loop:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        # Training metrics
        train_acc = accuracy_score(all_labels, all_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

        print(f"\n🟩 Epoch {epoch+1} Results:")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Train Acc: {train_acc:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")

        # Evaluation
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                preds = torch.argmax(out, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(y_batch.cpu().numpy())

        test_acc = accuracy_score(test_labels, test_preds)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')
        print(f"🧪 Test Acc: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
        val_f1.append(test_f1)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"💾 Checkpoint saved at epoch {epoch+1}")

    # Save validation F1 scores
    with open("val_f1.pkl", "wb") as f:
        pickle.dump(val_f1, f)

    # Final evaluation
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            preds = torch.argmax(out, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y_batch.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')
    print(f"Final Test Acc: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
    print("✅ Training complete!")

    # Save final model
    torch.save(model.state_dict(), "rnn_model.pth")

# -----------------------
# Run training
if __name__ == "__main__":
    train_rnn(total_epochs=30)