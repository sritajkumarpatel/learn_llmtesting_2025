# RUNS WITH HUGGING FACE - Free models, no API keys required

"""
Model Accuracy Evaluation Utility (Hugging Face)
================================================

What is Model Accuracy Evaluation?
- Measures how well a text classification model performs on a test dataset
- Calculates the proportion of correct predictions out of total predictions
- Uses accuracy metric from Hugging Face evaluate library for standardized computation

How It Works:
- Takes: model name, dataset name, and split (e.g., "validation" or "test")
- Loads the specified dataset split and creates a text classification pipeline
- Generates predictions for each text example in the dataset
- Maps string predictions to integers (1 for "POSITIVE", 0 for "NEGATIVE")
- Computes accuracy by comparing predictions to ground truth labels

Score Interpretation:
- Score Range: 0.0 to 1.0 (proportion of correct predictions)
- 0.0 = All predictions incorrect (worst performance)
- 0.5 = Half predictions correct (random guessing baseline)
- 1.0 = All predictions correct (perfect performance)

Threshold: N/A (raw accuracy score, no predefined pass/fail criteria)

Use Cases:
- Benchmarking model performance on standard datasets
- Comparing different model architectures or fine-tuning approaches
- Validating model generalization on held-out test data
- Quality assurance for deployed models

Reference: Hugging Face Evaluate
https://huggingface.co/docs/evaluate/index
"""

from datasets import load_dataset
from transformers import pipeline
from evaluate import load

def test_model_accuracy(model_name: str, dataset_name: str, split: str = "test"):
    """
    Evaluate model accuracy on a dataset using Hugging Face evaluate.
    """
    # Load the accuracy metric
    accuracy_metric = load("accuracy")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Create a classification pipeline
    classifier = pipeline("text-classification", model=model_name)
    
    # Get predictions and references
    predictions = []
    references = []
    
    for example in dataset:
        pred = classifier(example["sentence"])[0]
        predictions.append(1 if pred["label"] == "POSITIVE" else 0)
        references.append(example["label"])
    
    # Compute accuracy
    results = accuracy_metric.compute(predictions=predictions, references=references)
    
    return results

# Example usage
if __name__ == "__main__":
    results = test_model_accuracy(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        dataset_name="sst2",
        split="validation"
    )
    print(f"Accuracy: {results['accuracy']}")