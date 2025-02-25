#!/usr/bin/env python3
"""
Script to train a CNN sentiment classifier on a CSV dataset
with columns: 'processed_comment' and 'sentiment' (0=negative, 1=positive).
Uses embeddings + CNN architecture optimized for text classification.

Usage:
  python cnn.py --csv_path dataset.csv --model_out custom_cnn_sentiment.pt --vocab_out vocab.json
"""

import argparse
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added this import
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Arguments
parser = argparse.ArgumentParser(description="Train a CNN sentiment model.")
parser.add_argument('--csv_path', type=str, required=True, help="Path to CSV with processed_comment and sentiment columns")
parser.add_argument('--model_out', type=str, default='custom_cnn_sentiment.pt', help="Where to save the trained model")
parser.add_argument('--vocab_out', type=str, default='vocab.json', help="Where to save the vocabulary")
parser.add_argument('--max_vocab', type=int, default=20000, help="Max vocabulary size")
parser.add_argument('--max_len', type=int, default=100, help="Max sequence length")
parser.add_argument('--embed_dim', type=int, default=300, help="Embedding dimension")
parser.add_argument('--num_filters', type=int, default=100, help="Number of filters per size")
parser.add_argument('--filter_sizes', type=str, default='3,4,5', help="Comma-separated CNN filter sizes")
parser.add_argument('--batch_size', type=int, default=64, help="Training batch size")
parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
parser.add_argument('--train_split', type=float, default=0.7, help="Proportion of data for training")
parser.add_argument('--val_split', type=float, default=0.15, help="Proportion of data for validation")
parser.add_argument('--test_split', type=float, default=0.15, help="Proportion of data for testing")
parser.add_argument('--results_out', type=str, default='CNN_results.json', help="Where to save evaluation results")

args = parser.parse_args()

class SentimentDataset(Dataset):
    def __init__(self, df, word2index, max_len=100):
        self.texts = df['processed_comment'].tolist()
        self.labels = df['sentiment'].tolist()
        self.word2index = word2index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        label = self.labels[idx]
        tokens = txt.split()

        # Convert tokens to indices
        indices = [self.word2index.get(t, 0) for t in tokens]  # 0 = <unk>
        # Pad or truncate
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [1] * (self.max_len - len(indices))  # 1 = <pad>

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def build_vocab(texts, max_vocab=20000):
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and take top max_vocab-2 words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab-2]
    
    # Create word2index dictionary
    word2index = {'<unk>': 0, '<pad>': 1}  # Reserve 0 for unknown, 1 for padding
    for i, (word, _) in enumerate(sorted_words):
        word2index[word] = i + 2
    
    return word2index

class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, output_dim=1, dropout=0.5):
        super(SentimentCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        
        # Multiple parallel convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) 
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, max_len)
        
        embedded = self.embedding(x)
        # embedded shape: (batch_size, max_len, embed_dim)
        
        embedded = embedded.unsqueeze(1)
        # embedded shape: (batch_size, 1, max_len, embed_dim)
        
        # Apply convolutions and max-pooling
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate pooled features
        cat = torch.cat(pooled, dim=1)
        
        # Apply dropout and classification layer
        dropped = self.dropout(cat)
        return self.sigmoid(self.fc(dropped))

def train_one_epoch(model, loader, criterion, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    preds = []
    trues = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = (outputs >= 0.5).float()
        preds.extend(pred.cpu().numpy())
        trues.extend(labels.cpu().numpy())

    epoch_acc = accuracy_score(trues, preds)
    epoch_f1 = f1_score(trues, preds)
    return total_loss / len(loader), epoch_acc, epoch_f1

def eval_one_epoch(model, loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0
    preds = []
    trues = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            pred = (outputs >= 0.5).float()
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    epoch_acc = accuracy_score(trues, preds)
    epoch_f1 = f1_score(trues, preds)
    return total_loss / len(loader), epoch_acc, epoch_f1

def test_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'classification_report': classification_report(all_labels, all_preds, output_dict=True),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'loss': total_loss / len(test_loader)
    }
    
    # Print classification report
    print("\nTest Set Classification Report:")
    print(classification_report(all_labels, all_preds))
    
    return metrics

def plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_results(results, filepath):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    df = pd.read_csv(args.csv_path)
    
    # Split data
    train_val_df, test_df = train_test_split(df, test_size=args.test_split, random_state=42)
    train_size = args.train_split / (args.train_split + args.val_split)
    train_df, val_df = train_test_split(train_val_df, train_size=train_size, random_state=42)
    
    print(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Build vocabulary
    print("Building vocabulary...")
    word2index = build_vocab(train_df['processed_comment'].tolist(), max_vocab=args.max_vocab)
    vocab_size = len(word2index)
    print(f"Vocab size = {vocab_size}")

    # Save vocabulary
    with open(args.vocab_out, 'w', encoding='utf-8') as f:
        json.dump(word2index, f, ensure_ascii=False)

    # Create datasets and dataloaders
    train_dataset = SentimentDataset(train_df, word2index, max_len=args.max_len)
    val_dataset = SentimentDataset(val_df, word2index, max_len=args.max_len)
    test_dataset = SentimentDataset(test_df, word2index, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
    model = SentimentCNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        filter_sizes=filter_sizes,
        output_dim=1,
        dropout=args.dropout
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize results dictionary
    results = {
        'training': {
            'epochs': [],
            'best_val_f1': None
        },
        'test': {}
    }

    # Training loop
    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device=device
        )
        val_loss, val_acc, val_f1 = eval_one_epoch(model, val_loader, criterion, device=device)

        # Store epoch results
        results['training']['epochs'].append({
            'epoch': epoch + 1,
            'train': {'loss': train_loss, 'accuracy': train_acc, 'f1': train_f1},
            'val': {'loss': val_loss, 'accuracy': val_acc, 'f1': val_f1}
        })

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} "
              f"|| Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.model_out)
            print(f"  [*] Saved new best model to {args.model_out} (val_f1={val_f1:.4f})")
    
    results['training']['best_val_f1'] = best_val_f1

    print("Training complete!")
    
    # Testing phase
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(args.model_out))
    metrics = test_model(model, test_loader, criterion, device)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        save_path=args.results_out.replace('.json', '_confusion_matrix.png')
    )
    
    # Store test results
    results['test'] = metrics
    
    # Save results
    save_results(results, args.results_out)
    print(f"\nResults saved to {args.results_out}")

if __name__ == "__main__":
    main()