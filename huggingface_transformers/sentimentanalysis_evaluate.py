"""
Sentiment Analysis Model Evaluation
===================================

What is Sentiment Analysis Model Evaluation?
- Evaluates the performance of a sentiment analysis model using multiple metrics
- Measures accuracy, precision, recall, and F1-score for binary sentiment classification
- Uses a fine-tuned RoBERTa model trained for sentiment analysis
- Demonstrates model evaluation workflow with custom test data

How It Works:
- Takes: test data with text and true labels (0=negative, 1=positive)
- Uses RoBERTa sentiment pipeline to classify sentiment (POSITIVE/NEGATIVE)
- Maps predictions to integers (POSITIVE=1, NEGATIVE=0)
- Computes accuracy, precision, recall, and F1-score metrics

Score Interpretation:
- All Scores Range: 0.0 to 1.0 (higher is better)
- Accuracy: Proportion of correct predictions
- Precision: Proportion of positive predictions that are correct (TP/(TP+FP))
- Recall: Proportion of actual positives correctly identified (TP/(TP+FN))
- F1-Score: Harmonic mean of precision and recall (2*P*R/(P+R))

Threshold: N/A (raw scores, no predefined pass/fail criteria)

Use Cases:
- Comprehensive benchmarking of sentiment analysis models
- Validating model performance across multiple evaluation metrics
- Comparing different sentiment classification approaches
- Quality assurance for deployed sentiment models

Reference: Hugging Face Transformers & Evaluate
https://huggingface.co/docs/transformers/main_classes/pipelines
https://huggingface.co/docs/evaluate/index
"""

from transformers import pipeline
import evaluate

sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def evaluate_model(pipeline, test_data):
    """
    Evaluate model performance on test data using multiple sentiment analysis metrics.

    Processes test data containing text and labels, generates predictions,
    and computes accuracy, precision, recall, and F1-score.

    Parameters:
    pipeline: Hugging Face pipeline for sentiment analysis
    test_data: List of dicts with 'text' and 'label' keys, or Hugging Face dataset

    Returns:
    dict: Dictionary containing accuracy, precision, recall, and f1 scores
    """
    texts = [item["text"] for item in test_data]
    true_labels = [item["label"] for item in test_data]
    preds = pipeline(texts)
    preds_int = [1 if p['label'] == 'POSITIVE' else 0 for p in preds]
    
    accuracy = accuracy_metric.compute(predictions=preds_int, references=true_labels)
    precision = precision_metric.compute(predictions=preds_int, references=true_labels, average="binary")
    recall = recall_metric.compute(predictions=preds_int, references=true_labels, average="binary")
    f1 = f1_metric.compute(predictions=preds_int, references=true_labels, average="binary")
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }

if __name__ == "__main__":
    test_data = [
        {"text": "I love using large language models for natural language processing tasks!", "label": 1},
        {"text": "I hate my life", "label": 0},
        {"text": "i guess the food was okay", "label": 1},
        {"text": "The movie had stunning visuals but the plot was dull and uninteresting.", "label": 0}
    ]
    results = evaluate_model(sentiment_pipeline, test_data)
    print("Custom Test Data Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1']:.4f}")