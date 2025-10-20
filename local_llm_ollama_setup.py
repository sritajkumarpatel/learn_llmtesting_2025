import ollama
from deepeval.models import OllamaModel

def setup_ollama():
    try:
        # Test if Ollama is already running by listing models
        ollama.list()
    except Exception:
        # If not running, start Ollama server
        ollama.serve()

def setup_custom_ollama_model_for_evaluation(model="deepseek-r1:8b", base_url="http://localhost:11434", temperature=0):
    """Sets up and returns a custom Ollama model for evaluation purposes."""
    return OllamaModel(model=model, base_url=base_url, temperature=temperature)