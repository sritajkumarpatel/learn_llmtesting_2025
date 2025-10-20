import ollama

def setup_ollama():
    try:
        # Test if Ollama is already running by listing models
        ollama.list()
    except Exception:
        # If not running, start Ollama server
        ollama.serve()