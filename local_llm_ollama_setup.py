"""
Local LLM Ollama Setup Module
=============================
Provides utility functions to initialize and manage local Ollama LLM instances.

Functions:
- setup_ollama(): Start Ollama server if not running
- setup_custom_ollama_model_for_evaluation(): Create evaluator model instance
- generate_ollama_response(): Generate response from specified model
"""

import ollama
from deepeval.models import OllamaModel

def setup_ollama():
    """
    Initialize and ensure Ollama server is running.
    
    Attempts to connect to Ollama. If connection fails, starts the server.
    No parameters or return value.
    
    Example:
        setup_ollama()  # Ensures Ollama is ready to use
    """
    try:
        # Test if Ollama is already running by listing models
        ollama.list()
    except Exception:
        # If not running, start Ollama server
        ollama.serve()

def setup_custom_ollama_model_for_evaluation(model="deepseek-r1:8b", base_url="http://localhost:11434", temperature=0):
    """
    Create a custom Ollama model instance for evaluation/judging.
    
    Args:
        model (str): Ollama model name. Default: "deepseek-r1:8b"
        base_url (str): Ollama server URL. Default: "http://localhost:11434"
        temperature (int): Model temperature (0=deterministic, 1+=creative). Default: 0
    
    Returns:
        OllamaModel: Configured model instance for DeepEval evaluation
    
    Example:
        evaluator = setup_custom_ollama_model_for_evaluation(
            model="deepseek-r1:8b",
            temperature=0  # Consistent results for evaluation
        )
    """
    return OllamaModel(model=model, base_url=base_url, temperature=temperature)

def generate_ollama_response(model_name, query):
    """
    Generate a response from specified Ollama model for a given query.
    
    Args:
        model_name (str): Name of the Ollama model to use (e.g., "llama3.2:3b")
        query (str): The question/prompt to send to the model
    
    Returns:
        str: The model's response text
    
    Example:
        response = generate_ollama_response('llama3.2:3b', 'What is AI?')
        print(response)  # Outputs model's answer
    """
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': query
        }
    ])
    return response['message']['content']