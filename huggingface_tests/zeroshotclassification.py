"""
Zero-Shot Text Classification Utility (Hugging Face Transformers)
===============================================================

What is Zero-Shot Text Classification?
- Classifies text into user-provided categories without task-specific training
- Uses natural language inference (NLI) to determine text-label relationships
- Reformulates classification as entailment: "Does this text entail 'This is about [label]'?"

How It Works:
- Takes: text input string and list of candidate labels
- Loads BART-large-MNLI model fine-tuned for natural language inference
- For each label, creates hypothesis "This text is about [label]" and checks entailment
- Returns: dictionary with labels ranked by entailment confidence scores

Use Cases:
- Topic classification with custom categories
- Sentiment analysis with domain-specific labels
- Content categorization for news/articles
- Intent detection in chatbots
- Multi-label document tagging

Reference: Hugging Face Transformers Zero-Shot Classification
https://huggingface.co/docs/transformers/tasks/zero_shot_classification
"""

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_text(text: str, labels: list) -> dict:
    return classifier(text, candidate_labels=labels)

if __name__ == "__main__":
    text = "The new movie is a thrilling adventure that takes place in a dystopian future."
    labels = ["action", "drama", "comedy", "sci-fi"]
    result = classify_text(text, labels)
    print("Classification Result:")
    print(result)
