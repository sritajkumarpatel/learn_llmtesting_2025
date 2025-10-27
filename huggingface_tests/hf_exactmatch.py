# RUNS WITH HUGGING FACE - Free models, no API keys required

"""
Exact Match Evaluation Utility (Hugging Face)
=============================================

What is Exact Match Evaluation?
- Evaluates model performance using exact match accuracy
- Measures the proportion of predictions that exactly match the references
- Uses IMDB movie review dataset for evaluation
- Demonstrates exact match evaluation for text classification tasks

How It Works:
- Loads the BART large MNLI zero-shot classification model
- Uses the test split of the IMDB dataset (Internet Movie Database reviews)
- Generates predictions for each test example using zero-shot with POSITIVE/NEGATIVE labels
- Computes exact match using string labels (POSITIVE/NEGATIVE)
- Prints the exact match score

Score Interpretation:
- Exact Match Range: 0.0 to 1.0 (higher is better)
- 0.0 = No predictions match exactly
- 1.0 = All predictions match exactly

Threshold: N/A (raw score, no predefined pass/fail criteria)

Use Cases:
- Evaluating sentiment analysis model performance on exact matches
- Benchmarking text classification models
- Understanding model accuracy on real-world review data

Reference: Hugging Face Evaluate
https://huggingface.co/docs/evaluate/index
"""

from transformers import pipeline
from datasets import load_dataset
from evaluate import load

def test_model_exactmatch():
    # Load the zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["POSITIVE", "NEGATIVE"]
    
    # Load the IMDB dataset
    dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(1000))
    
    # Load exact match metric
    exact_match_metric = load("exact_match")
    
    predictions_strings = []
    references_strings = []
    
    # Process the subset
    for example in dataset:
        text = example["text"]
        true_label = example["label"]  # 0 or 1
        
        # Truncate text to avoid sequence length errors
        text = text[:1024]
        
        # Get prediction
        pred = classifier(text, candidate_labels=candidate_labels)
        pred_label = pred["labels"][0]  # Highest scoring label
        
        # Keep strings for exact match
        pred_string = pred_label
        ref_string = "POSITIVE" if true_label == 1 else "NEGATIVE"
        
        predictions_strings.append(pred_string)
        references_strings.append(ref_string)
    
    # Compute exact match
    exact_match_result = exact_match_metric.compute(predictions=predictions_strings, references=references_strings)
    
    print(f"Model Exact Match: {exact_match_result['exact_match']:.4f}")
    
    return exact_match_result

if __name__ == "__main__":
    test_model_exactmatch()