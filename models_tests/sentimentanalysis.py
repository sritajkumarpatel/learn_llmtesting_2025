# RUNS WITH HUGGING FACE - Free models, no API keys required

"""
Sentiment Analysis Utility
==========================

What is Sentiment Analysis?
- Analyzes the emotional tone and attitude expressed in text input
- Classifies text as Positive, Negative, or Neutral sentiment
- Uses LLM to perform natural language understanding and classification

How It Works:
- Takes: text input string
- Constructs prompt asking LLM to analyze sentiment
- Returns: sentiment classification ("Positive", "Negative", or "Neutral")

Use Cases:
- Content moderation and filtering
- Customer feedback analysis
- Social media sentiment monitoring
- Brand reputation management
- Automated review processing

Reference: Ollama Models
https://ollama.ai/library
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import config, setup_ollama, generate_ollama_response
from utils.config import get_ollama_model, get_ollama_url, ModelType

def sentiment_analysis(text: str) -> str:
    """
    Perform sentiment analysis on the given text using an Ollama model.

    Parameters:
    text (str): The input text to analyze.

    Returns:
    str: The sentiment analysis result (e.g., "Positive", "Negative", "Neutral").
    """
    
    sample_text = f"Analyze the sentiment of the following text and respond with 'Positive', 'Negative', or 'Neutral':\n\n{text}"

    model_name = get_ollama_model(ModelType.ACTUAL_OUTPUT)
    response = generate_ollama_response(sample_text, model_name=model_name)
    return response

if __name__ == "__main__":
    setup_ollama()
    positive_text = "I love using large language models for natural language processing tasks!"
    sentiment = sentiment_analysis(positive_text)
    print(f"Sentiment Analysis Result: {sentiment}")

    negative_text = "I hate my life"
    sentiment = sentiment_analysis(negative_text)
    print(f"Sentiment Analysis Result: {sentiment}")

    neutral_text = "The food was okay"
    sentiment = sentiment_analysis(neutral_text)
    print(f"Sentiment Analysis Result: {sentiment}")