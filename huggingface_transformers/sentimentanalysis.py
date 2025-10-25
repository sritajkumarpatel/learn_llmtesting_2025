"""
Sentiment Analysis Utility (Hugging Face Transformers)
======================================================

What is Sentiment Analysis?
- Analyzes the emotional tone and attitude expressed in text input
- Classifies text as Positive or Negative sentiment using pre-trained transformer models
- Uses DistilBERT fine-tuned on SST-2 dataset for binary sentiment classification

How It Works:
- Takes: text input string
- Loads default sentiment-analysis pipeline (distilbert-base-uncased-finetuned-sst-2-english)
- Processes text through transformer model to predict sentiment probabilities
- Returns: sentiment classification ("Positive" or "Negative") based on highest probability

Use Cases:
- Content moderation and filtering
- Customer feedback analysis
- Social media sentiment monitoring
- Brand reputation management
- Automated review processing

Reference: Hugging Face Transformers Sentiment Analysis
https://huggingface.co/docs/transformers/tasks/sequence_classification
"""

from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text: str) -> str:
    result = classifier(text)[0]
    return result['label'] + " " + round(result['score'], 4).__str__()

if __name__ == "__main__":
    positive_text = "I love using large language models for natural language processing tasks!"
    sentiment = analyze_sentiment(positive_text)
    print(f"Sentiment Analysis Result: {sentiment}")

    negative_text = "I hate my life"
    sentiment = analyze_sentiment(negative_text)
    print(f"Sentiment Analysis Result: {sentiment}")

    neutral_text = "i guess the food was okay"
    sentiment = analyze_sentiment(neutral_text)
    print(f"Sentiment Analysis Result: {sentiment}")

    complicated_text = "The movie had stunning visuals but the plot was dull and uninteresting."
    sentiment = analyze_sentiment(complicated_text)
    print(f"Sentiment Analysis Result: {sentiment}")