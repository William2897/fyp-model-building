# FairEval - Multi-Model Sentiment Analysis System

## Overview
FairEval is a sentiment analysis system that evaluates comments using four different machine learning models. It provides a comparative analysis of sentiment predictions to ensure fair and robust evaluation of text.

## Features
- **Multiple Model Architecture**: Utilizes four different models for sentiment analysis:
  - Linear SVC with TF-IDF features
  - Random Forest with TF-IDF features
  - CNN with word embeddings
  - LSTM with attention mechanism
- **Web Interface**: User-friendly UI for inputting comments and viewing analysis results
- **Confidence Scores**: Displays prediction confidence from each model
- **Comparative Analysis**: Shows sentiment predictions across all models for better decision making

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FairEval.git
cd FairEval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
.
├── app.py                      # Main Flask application
├── labeling_pipeline.py        # Pipeline for labeling data with transformer models
├── vocab.json                  # Vocabulary file for neural models
├── Modelling/                  # Model implementation files
│   ├── LinearSVC_TF-IDF.py     # LinearSVC model implementation
│   ├── RF_TF-IDF.py            # Random Forest model implementation 
│   ├── lstm.py                 # LSTM model implementation
│   ├── cnn.py                  # CNN model implementation
│   └── ...                     # Other model files
├── Eval Results/               # Evaluation metrics and results
│   ├── linearsvc_results.json  # LinearSVC evaluation results
│   ├── rf_results.json         # Random Forest evaluation results
│   ├── LSTM_results.json       # LSTM evaluation results
│   └── CNN_results.json        # CNN evaluation results
├── templates/                  # Web interface templates
│   └── index.html              # Main UI template
├── Trained Models/             # Directory for saved model files
└── Results Images/             # Visualizations of model performance
```

## Usage

### Running the Web Application
```bash
python app.py
```
Then open your browser and go to http://localhost:5000/

### Training Models
To train the LinearSVC model:
```bash
python Modelling/LinearSVC_TF-IDF.py --csv_path path/to/data.csv --model_out model.joblib --results_out results.json
```

To train the Random Forest model:
```bash
python Modelling/RF_TF-IDF.py --csv_path path/to/data.csv --model_out rf_model.joblib --results_out results.json
```

To train the LSTM model:
```bash
python Modelling/lstm.py --csv_path path/to/data.csv --model_out lstm_model.pt --vocab_out vocab.json
```

### Labeling New Data
```bash
python labeling_pipeline.py
```

## Model Details

### Traditional Machine Learning Models
- **LinearSVC**: Uses TF-IDF features with n-grams (1-3) for text representation
- **Random Forest**: Ensemble method with TF-IDF features for robust predictions

### Deep Learning Models
- **CNN**: Convolutional Neural Network with word embeddings
- **LSTM**: Bidirectional LSTM with attention mechanism for sequential understanding

## Data Processing
The system processes raw comments through:
1. Text preprocessing
2. Feature extraction (TF-IDF or embeddings)
3. Model prediction
4. Confidence score calculation

## Evaluation Metrics
Each model is evaluated using:
- Accuracy
- F1 Score
- Confusion Matrix
- Feature importance (for traditional ML models)

## Credits
Developed as part of the FairEval Model Development project (Final Year Project for Asia Pacific University of Technology and Innovation).