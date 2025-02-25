#!/usr/bin/env python3

import argparse
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import psutil
import warnings
from joblib import dump, load

# Arguments
parser = argparse.ArgumentParser(description="Train a LinearSVC with TF-IDF for sentiment analysis.")
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--model_out', type=str, default='linearsvc_model.joblib')
parser.add_argument('--vectorizer_out', type=str, default='tfidf_vectorizer.joblib')
parser.add_argument('--max_features', type=int, default=3000)
parser.add_argument('--results_out', type=str, default='linearsvc_results.json')
parser.add_argument('--batch_size', type=int, default=10000)

args = parser.parse_args()

def batch_predict(model, X, batch_size=10000):
    """Make predictions in batches to avoid memory issues"""
    predictions = []
    n_samples = X.shape[0]
    
    for i in tqdm(range(0, n_samples, batch_size), desc="Predicting"):
        batch_end = min(i + batch_size, n_samples)
        batch_pred = model.predict(X[i:batch_end])
        predictions.extend(batch_pred)
    
    return np.array(predictions)

def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix using seaborn"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_feature_importance(model, vectorizer, top_n=20):
    """Analyze and plot feature coefficients"""
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]  # For binary classification
    abs_coef = np.abs(coef)
    indices = np.argsort(abs_coef)[::-1][:top_n]
    
    plt.figure(figsize=(12, 6))
    plt.title('Top Feature Coefficients')
    plt.bar(range(top_n), coef[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_coefficients.png')
    plt.close()
    
    # Print top features and their coefficients
    print("\nTop important features:")
    for i in indices:
        print(f"{feature_names[i]}: {coef[i]:.4f}")

def main():
    print("Loading data...")
    df = pd.read_csv(args.csv_path)
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    print("\nPerforming TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=5,
        max_df=0.9,
        dtype=np.float32
    )
    
    # Fit and transform training data
    X_train = vectorizer.fit_transform(train_df['processed_comment'])
    y_train = train_df['sentiment'].values
    
    # Transform validation and test data
    X_val = vectorizer.transform(val_df['processed_comment'])
    y_val = val_df['sentiment'].values
    X_test = vectorizer.transform(test_df['processed_comment'])
    y_test = test_df['sentiment'].values

    print("\nTraining model...")
    svc = LinearSVC(random_state=42, max_iter=2000)
    svc.fit(X_train, y_train)
    
    # Analyze feature importance
    analyze_feature_importance(svc, vectorizer)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_val_pred = batch_predict(svc, X_val, args.batch_size)
    val_metrics = {
        'accuracy': float(accuracy_score(y_val, y_val_pred)),
        'f1': float(f1_score(y_val, y_val_pred))
    }
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation F1 Score: {val_metrics['f1']:.4f}")
    
    print("\nMaking predictions on test set...")
    y_pred = batch_predict(svc, X_test, args.batch_size)

    # Calculate metrics
    test_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    # Print results
    print("\nTest Results:")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(test_metrics['confusion_matrix']),
        save_path=args.results_out.replace('.json', '_confusion_matrix.png')
    )

    # Save results and models
    print("\nSaving results and models...")
    metrics = {
        'validation': val_metrics,
        'test': test_metrics,
        'feature_coefficients': {
            'features': vectorizer.get_feature_names_out().tolist(),
            'coefficients': svc.coef_[0].tolist()
        }
    }
    
    with open(args.results_out, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    dump(svc, args.model_out)
    dump(vectorizer, args.vectorizer_out)

    print(f"Results saved to {args.results_out}")
    print(f"Model saved to {args.model_out}")
    print(f"Vectorizer saved to {args.vectorizer_out}")
    print(f"Feature coefficients plot saved to feature_coefficients.png")

if __name__ == "__main__":
    main()