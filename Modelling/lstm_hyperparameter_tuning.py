#!/usr/bin/env python3
"""
Hyperparameter tuning script for LSTM sentiment classifier using Random Search.
Uses a subset of data (10%) to speed up the tuning process.

Usage:
  python hyperparameter_tuning.py --csv_path dataset.csv --results_out tuning_results.json
"""

import argparse
import pandas as pd
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
from lstm import CustomSentimentLSTM, SentimentDataset, build_vocab
from sklearn.metrics import f1_score
import time
from datetime import datetime

# Argument parsing
parser = argparse.ArgumentParser(description="Tune hyperparameters for LSTM sentiment model.")
parser.add_argument('--csv_path', type=str, required=True, help="Path to CSV dataset")
parser.add_argument('--results_out', type=str, default='lstm_tuning_results.json', help="Where to save tuning results")
parser.add_argument('--n_trials', type=int, default=20, help="Number of random trials")
parser.add_argument('--sample_size', type=float, default=0.1, help="Fraction of data to use for tuning")
args = parser.parse_args()

# Define hyperparameter search space
param_space = {
    'embed_dim': [128, 256],
    'hidden_dim': [256, 512],
    'num_layers': [2, 3],
    'batch_size': [32, 64],
    'learning_rate': [1e-4, 2e-4, 5e-4, 1e-3],
    'dropout': [0.2, 0.3, 0.4, 0.5],
    'max_len': [50, 100, 150],
}

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    """Train the model and return the best validation F1 score"""
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)
                preds = (outputs >= 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds)
        best_val_f1 = max(best_val_f1, val_f1)
    
    return best_val_f1

def random_search():
    # Set random seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and sample data
    df = pd.read_csv(args.csv_path)
    df_sample = df.sample(frac=args.sample_size, random_state=42)
    print(f"Using {len(df_sample)} samples for tuning (original size: {len(df)})")
    
    # Split data
    train_df, val_df = train_test_split(df_sample, test_size=0.2, random_state=42)
    
    # Results storage
    results = {
        'trials': [],
        'best_params': None,
        'best_f1': 0.0,
        'tuning_info': {
            'sample_size': args.sample_size,
            'n_trials': args.n_trials,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_samples': len(df_sample)
        }
    }
    
    # Build vocabulary
    word2index = build_vocab(train_df['processed_comment'].tolist())
    vocab_size = len(word2index)
    
    for trial in range(args.n_trials):
        # Randomly sample hyperparameters
        params = {
            'embed_dim': random.choice(param_space['embed_dim']),
            'hidden_dim': random.choice(param_space['hidden_dim']),
            'num_layers': random.choice(param_space['num_layers']),
            'batch_size': random.choice(param_space['batch_size']),
            'learning_rate': random.choice(param_space['learning_rate']),
            'dropout': random.choice(param_space['dropout']),
            'max_len': random.choice(param_space['max_len'])
        }
        
        print(f"\nTrial {trial + 1}/{args.n_trials}")
        print("Parameters:", params)
        
        # Create datasets
        train_dataset = SentimentDataset(train_df, word2index, max_len=params['max_len'])
        val_dataset = SentimentDataset(val_df, word2index, max_len=params['max_len'])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Initialize model
        model = CustomSentimentLSTM(
            vocab_size=vocab_size,
            embed_dim=params['embed_dim'],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
        
        # Train and evaluate
        start_time = time.time()
        val_f1 = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device)
        training_time = time.time() - start_time
        
        # Store results
        trial_results = {
            'trial': trial + 1,
            'parameters': params,
            'val_f1': float(val_f1),
            'training_time': training_time
        }
        results['trials'].append(trial_results)
        
        print(f"Validation F1: {val_f1:.4f}")
        print(f"Training time: {training_time:.2f}s")
        
        # Update best results
        if val_f1 > results['best_f1']:
            results['best_f1'] = float(val_f1)
            results['best_params'] = params
            print("New best model found!")
    
    # Save results
    with open(args.results_out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTuning results saved to {args.results_out}")
    
    print("\nBest parameters found:")
    for param, value in results['best_params'].items():
        print(f"{param}: {value}")
    print(f"Best validation F1: {results['best_f1']:.4f}")

if __name__ == "__main__":
    random_search()