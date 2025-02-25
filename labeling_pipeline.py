# labeling_pipeline.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd

# Configuration
MODEL_NAME = "siebert/sentiment-roberta-large-english"
BATCH_SIZE = 32  # Increased for better GPU utilization
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_dataset(file_path):
    df = pd.read_csv(file_path)
    # Ensure the required column exists
    if 'rating_comment' not in df.columns:
        raise ValueError("CSV file must contain a 'rating_comment' column")
    return Dataset.from_pandas(df)

def process_function(examples, tokenizer, model):
    # Tokenize the texts
    encodings = tokenizer(
        examples['rating_comment'],  # Changed from 'comment' to 'rating_comment'
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Move to GPU and get predictions
    with torch.no_grad():
        inputs = {k: v.to(DEVICE) for k, v in encodings.items()}
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get labels and scores
    labels = predictions.argmax(dim=-1)
    scores = predictions.max(dim=-1).values
    
    return {
        'sentiment': labels.cpu().numpy(),
        'confidence': scores.cpu().numpy()
    }

def label_data(input_file, output_file):
    # Load model with optimizations
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map='auto'
    ).to(DEVICE)
    model.eval()
    
    # Create and process dataset
    dataset = create_dataset(input_file)
    
    # Process in batches
    processed_dataset = dataset.map(
        lambda x: process_function(x, tokenizer, model),
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=dataset.column_names
    )
    
    # Save results
    df = pd.DataFrame({
        'rating_comment': dataset['rating_comment'],  # Changed from 'comment'
        'sentiment': processed_dataset['sentiment'],
        'confidence': processed_dataset['confidence']
    })
    df.to_csv(output_file, index=False)

# Run labeling
if __name__ == "__main__":
    label_data(
        input_file="C:/Users/User/OneDrive/Desktop/topic_results.csv",
        output_file="C:/Users/User/OneDrive/Desktop/FairEval/RMP SET - Labelled SA.csv"
    )