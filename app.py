import os
import torch
import numpy as np
import json
from joblib import load
from flask import Flask, render_template, request, jsonify
from torch.utils.data import Dataset

# Models path
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trained Models')

# Flask app
app = Flask(__name__)

# Load models
def load_models():
    # Load the vocabulary for neural models
    with open('vocab.json', 'r', encoding='utf-8') as f:
        word2index = json.load(f)
    
    # Load TF-IDF vectorizer (shared by LinearSVC and RF)
    vectorizer = load(os.path.join(MODELS_PATH, 'tfidf_vectorizer.joblib'))
    
    # Load LinearSVC
    linearsvc_model = load(os.path.join(MODELS_PATH, 'linearsvc_model.joblib'))
    
    # Load Random Forest
    rf_model = load(os.path.join(MODELS_PATH, 'rf_model.joblib'))
    
    # Device configuration for PyTorch models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CNN model
    cnn_model = SentimentCNN(
        vocab_size=len(word2index), 
        embed_dim=300, 
        num_filters=100,
        filter_sizes=[3, 4, 5],
        output_dim=1, 
        dropout=0.5
    )
    cnn_model.load_state_dict(torch.load(os.path.join(MODELS_PATH, 'custom_cnn_sentiment.pt'), map_location=device))
    cnn_model.to(device)
    cnn_model.eval()
    
    # Load LSTM model
    lstm_model = CustomSentimentLSTM(
        vocab_size=len(word2index),
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        output_dim=1,
        dropout=0.5
    )
    lstm_model.load_state_dict(torch.load(os.path.join(MODELS_PATH, 'custom_lstm_sentiment_tuned.pt'), map_location=device))
    lstm_model.to(device)
    lstm_model.eval()
    
    return {
        'word2index': word2index,
        'vectorizer': vectorizer,
        'linearsvc': linearsvc_model,
        'rf': rf_model,
        'cnn': cnn_model,
        'lstm': lstm_model,
        'device': device
    }

# Dataset for neural models (CNN, LSTM)
class SentimentDataset(Dataset):
    def __init__(self, text, word2index, max_len=100):
        self.text = text
        self.word2index = word2index
        self.max_len = max_len

    def __len__(self):
        return 1  # We're only processing one sample at a time

    def __getitem__(self, idx):
        tokens = self.text.split()
        
        # Convert tokens to indices
        indices = [self.word2index.get(t, 0) for t in tokens]  # 0 = <unk>
        
        # Pad or truncate
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [1] * (self.max_len - len(indices))  # 1 = <pad>
            
        return torch.tensor(indices, dtype=torch.long)

# CNN Model definition
class SentimentCNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, output_dim=1, dropout=0.5):
        super(SentimentCNN, self).__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        
        # Multiple parallel convolutional layers
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(1, num_filters, (fs, embed_dim)) 
            for fs in filter_sizes
        ])
        
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, max_len)
        
        embedded = self.embedding(x)
        # embedded shape: (batch_size, max_len, embed_dim)
        
        embedded = embedded.unsqueeze(1)
        # embedded shape: (batch_size, 1, max_len, embed_dim)
        
        # Apply convolutions and max-pooling
        conved = [torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate pooled features
        cat = torch.cat(pooled, dim=1)
        
        # Apply dropout and classification layer
        dropped = self.dropout(cat)
        return self.sigmoid(self.fc(dropped))

# LSTM Model with attention
class CustomSentimentLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, output_dim=1, dropout=0.3):
        super(CustomSentimentLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm = torch.nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = torch.nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (h, c) = self.lstm(embedded)
        
        # Apply attention
        context = self.attention_net(lstm_out)
        
        # Dense layers with dropout
        out = self.dropout(self.relu(self.fc1(context)))
        out = self.sigmoid(self.fc2(out))
        return out

# Load all models 
models = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form.get('comment', '')
        processed_comment = comment.lower()  # Simple preprocessing
        
        # Make predictions with Linear SVC
        X_svc = models['vectorizer'].transform([processed_comment])
        linearsvc_pred = int(models['linearsvc'].predict(X_svc)[0])
        linearsvc_score = float(models['linearsvc'].decision_function(X_svc)[0])
        
        # Make predictions with Random Forest
        X_rf = models['vectorizer'].transform([processed_comment])
        rf_pred = int(models['rf'].predict(X_rf)[0])
        rf_prob = models['rf'].predict_proba(X_rf)[0]
        rf_score = float(rf_prob[1])  # Probability of positive class
        
        # Make prediction with CNN
        cnn_dataset = SentimentDataset(processed_comment, models['word2index'])
        cnn_input = next(iter(cnn_dataset)).unsqueeze(0).to(models['device'])
        with torch.no_grad():
            cnn_output = models['cnn'](cnn_input).item()
        cnn_pred = 1 if cnn_output >= 0.5 else 0
        
        # Make prediction with LSTM
        lstm_dataset = SentimentDataset(processed_comment, models['word2index'])
        lstm_input = next(iter(lstm_dataset)).unsqueeze(0).to(models['device'])
        with torch.no_grad():
            lstm_output = models['lstm'](lstm_input).item()
        lstm_pred = 1 if lstm_output >= 0.5 else 0
        
        # Format results
        results = {
            'comment': comment,
            'linearsvc': {
                'prediction': 'Positive' if linearsvc_pred == 1 else 'Negative',
                'score': linearsvc_score
            },
            'rf': {
                'prediction': 'Positive' if rf_pred == 1 else 'Negative',
                'score': rf_score
            },
            'cnn': {
                'prediction': 'Positive' if cnn_pred == 1 else 'Negative',
                'score': cnn_output
            },
            'lstm': {
                'prediction': 'Positive' if lstm_pred == 1 else 'Negative',
                'score': lstm_output
            }
        }
        
        return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)