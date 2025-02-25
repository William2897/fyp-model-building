#!/usr/bin/env python3

import argparse
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import numpy as np
from joblib import dump
import warnings
from tqdm import tqdm

# Arguments
parser = argparse.ArgumentParser(description="Hyperparameter tuning for Random Forest with TF-IDF")
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--best_model_out', type=str, default='rf_best_model.joblib')
parser.add_argument('--vectorizer_out', type=str, default='tfidf_best_vectorizer.joblib')
parser.add_argument('--results_out', type=str, default='rf_tuning_results.json')
parser.add_argument('--max_features', type=int, default=3000)
args = parser.parse_args()

def main():
    print("Loading data...")
    df = pd.read_csv(args.csv_path)
    
    # Get class distribution in original data
    original_dist = df['sentiment'].value_counts(normalize=True)
    print("\nOriginal class distribution:")
    print(original_dist)
    
    # First take 10% stratified random sample of the data for tuning
    print("\nTaking 10% stratified random sample for hyperparameter tuning...")
    df_sample = pd.DataFrame()
    for sentiment in df['sentiment'].unique():
        # Sample 10% from each class
        class_data = df[df['sentiment'] == sentiment]
        class_sample = class_data.sample(frac=0.1, random_state=42)
        df_sample = pd.concat([df_sample, class_sample])
    
    # Shuffle the combined sample
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify class distribution in sample
    sample_dist = df_sample['sentiment'].value_counts(normalize=True)
    print("\nSampled data class distribution:")
    print(sample_dist)
    print(f"Sample size: {len(df_sample)} rows")
    
    # Split sampled data for tuning (using stratified split)
    train_df, val_df = train_test_split(
        df_sample, 
        test_size=0.2, 
        random_state=42,
        stratify=df_sample['sentiment']
    )
    print(f"Data splits: Train={len(train_df)}, Val={len(val_df)}")
    
    # Verify class distribution in splits
    print("\nTraining set class distribution:")
    print(train_df['sentiment'].value_counts(normalize=True))
    print("\nValidation set class distribution:")
    print(val_df['sentiment'].value_counts(normalize=True))

    print("\nPerforming TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=5,
        max_df=0.9,
        dtype=np.float32
    )
    
    X_train = vectorizer.fit_transform(train_df['processed_comment'])
    y_train = train_df['sentiment'].values
    X_val = vectorizer.transform(val_df['processed_comment'])
    y_val = val_df['sentiment'].values

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [20, 30, 40, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }

    # Initialize base model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Create scorers
    scorers = {
        'f1': make_scorer(f1_score),
        'accuracy': make_scorer(accuracy_score)
    }

    # Initialize GridSearchCV
    print("\nStarting GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorers,
        refit='f1',  # Use F1 score to select best model
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # Fit GridSearchCV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train, y_train)

    # Get best parameters and scores
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_

    # Validate best model on validation set
    best_model = grid_search.best_estimator_
    val_accuracy = accuracy_score(y_val, best_model.predict(X_val))
    val_f1 = f1_score(y_val, best_model.predict(X_val))

    # Prepare results
    results = {
        'best_parameters': best_params,
        'best_cv_score': float(grid_search.best_score_),
        'validation_scores': {
            'accuracy': float(val_accuracy),
            'f1': float(val_f1)
        },
        'cv_results': {
            'params': [str(p) for p in cv_results['params']],
            'mean_test_f1': cv_results['mean_test_f1'].tolist(),
            'mean_test_accuracy': cv_results['mean_test_accuracy'].tolist(),
            'std_test_f1': cv_results['std_test_f1'].tolist(),
            'std_test_accuracy': cv_results['std_test_accuracy'].tolist(),
        }
    }

    # Save results and best model
    print("\nSaving results and best model...")
    with open(args.results_out, 'w') as f:
        json.dump(results, f, indent=2)
    
    dump(best_model, args.best_model_out)
    dump(vectorizer, args.vectorizer_out)

    # Print summary
    print("\nTuning Results Summary:")
    print(f"Best parameters: {best_params}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"\nResults saved to {args.results_out}")
    print(f"Best model saved to {args.best_model_out}")
    print(f"Vectorizer saved to {args.vectorizer_out}")

if __name__ == "__main__":
    main()