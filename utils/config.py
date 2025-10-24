import json
from enum import Enum
from pathlib import Path


class ModelType(Enum):
    """Enum for model types to avoid string errors"""
    EVALUATION = "evaluation"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    BASE_URL = "base_url"
    EVALUATION_ADVANCED = "evaluation_advanced"
    EMBEDDING_MODEL = "embedding_model"


# Load configuration once
def _load_config(config_path='config/models.json'):
    path = Path(__file__).parent.parent / config_path
    with open(path, 'r') as f:
        return json.load(f)


_config = _load_config()


def get_ollama_model(model_type: ModelType) -> str:
    """Get Ollama model by enum type"""
    if not isinstance(model_type, ModelType):
        raise TypeError(f"Expected ModelType enum, got {type(model_type).__name__}")
    return _config['ollama']['models'].get(model_type.value)


def get_ollama_url() -> str:
    """Get Ollama base URL"""
    return _config['ollama'][ModelType.BASE_URL.value]
