"""
Text Summarization Utility
==========================

What is Text Summarization?
- Condenses long text into shorter, coherent summaries
- Maintains key information and main ideas while reducing length
- Uses LLM to perform natural language understanding and generation

How It Works:
- Takes: text input string and desired word count limit
- Constructs prompt asking LLM to summarize within word limit
- Returns: concise summary text within specified word count

Use Cases:
- Content summarization for articles and documents
- Meeting notes and transcript condensation
- Research paper abstract generation
- News article summarization
- Educational content simplification

Reference: Ollama Models
https://ollama.ai/library
"""

from pathlib import Path
import sys
from xml.parsers.expat import model

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import config, setup_ollama, generate_ollama_response
from utils.config import get_ollama_model, get_ollama_url, ModelType

def text_summarization(text: str, count: int) -> str:
    """
   Summarize the following text in within {count} words using ollama.

    Parameters:
    text (str): The input text to summarize.

    Returns:
    str: The summarization result.
    """

    sample_text = f"Summarize the following text in within {count} words using ollama:\n\n{text}"

    model_name = get_ollama_model(ModelType.ACTUAL_OUTPUT)
    response = generate_ollama_response(sample_text, model_name=model_name)
    return response

if __name__ == "__main__":
    setup_ollama()

    # Example 1: Summarize within 50 words
    text_long = """
    Artificial intelligence is transforming industries worldwide. Machine learning models can now 
    analyze vast amounts of data and make predictions with unprecedented accuracy. Companies are 
    investing heavily in AI research and development to gain competitive advantages. From healthcare 
    to finance, AI applications are revolutionizing how businesses operate and serve customers.
    """
    
    prompt_50 = f"Summarize the following text in within 50 words:\n\n{text_long}"
    response50words = text_summarization(text_long, 50)
    print("Summary (within 50 words):")
    print(response50words)
    print()

   # Example 2: Summarize within 150 words
    text_longer = """
    Artificial intelligence is transforming industries worldwide by enabling machines to learn from data 
    and make intelligent decisions. Machine learning models can now analyze vast amounts of data and make 
    predictions with unprecedented accuracy. Companies are investing heavily in AI research and development 
    to gain competitive advantages and improve operational efficiency. From healthcare to finance, AI 
    applications are revolutionizing how businesses operate and serve customers. Natural language processing 
    allows computers to understand and generate human language. Computer vision enables machines to interpret 
    visual information from images and videos. These technologies are creating new opportunities for innovation 
    and solving complex problems that were previously impossible to tackle.
    """
    response150words = text_summarization(text_longer, 150)
    print("Summary (within 150 words):")
    print(response150words)
    

