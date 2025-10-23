"""
RAGAS NoiseSensitivity Metric
=============================

What is NoiseSensitivity?
- Measures how much a response is affected by irrelevant/noisy context injection
- Tests robustness by adding irrelevant documents to retrieved contexts
- Answers: "Does the response degrade when irrelevant context is added?"

How It Works:
- Takes: query, response, reference, and retrieved_contexts
- Injects irrelevant noise (perturbations) into the context systematically
- Measures: How many errors/inconsistencies appear in the response
- Outputs: Score from 0.0 to 1.0 (lower is better)

Score Interpretation (RAGAS Standard):
- 0.0       = Perfect (✅ PASS) - No incorrect claims detected, response robust to noise
- 0.0-0.3   = Good (✅ PASS) - ≤30% of claims become incorrect when noise added
- 0.3-0.5   = Fair (⚠️ PARTIAL) - 30-50% of claims become incorrect when noise added
- 0.5-1.0   = Poor (❌ FAIL) - >50% of claims become incorrect, low robustness to noise

Threshold: 0.5 (50%)
- Maximum acceptable: Noise sensitivity should be ≤0.5 (50% of claims can handle noise)
- Lower scores are better: 0.0 = perfect robustness, 1.0 = all claims fail with noise
- Interpretation: Score represents PROPORTION of claims that become incorrect when irrelevant context is injected

Use Cases:
- Evaluating response robustness to noisy/irrelevant context
- Quality assurance for RAG pipeline retrieval quality
- Detecting systems vulnerable to prompt injection or context confusion
- Comparing robustness across different response generation strategies

Reference: RAGAS Documentation
https://docs.ragas.io/en/latest/concepts/metrics/
"""

import sys
from pathlib import Path
from ragas import SingleTurnSample
from ragas.metrics import NoiseSensitivity
from ragas.llms.base import LangchainLLMWrapper
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import config, setup_ollama, generate_ollama_response
from utils.config import get_ollama_model, get_ollama_url, ModelType
from utils.wikipedia_retriever import retrieve_context_from_wiki

def test_noise_sensitivity(user_query, expected_output, context):
    """Test query with NoiseSensitivity metric."""

    # Generate response from LLM under test
    actual_output = generate_ollama_response(user_query, model_name=get_ollama_model(ModelType.ACTUAL_OUTPUT))
    
    test_data = {
        "user_input": user_query,
        "response": actual_output,
        "reference": expected_output,
        "retrieved_contexts": [context]
    }

    print(f"\n📝 Query: {user_query}")
    print(f"💬 Response: {actual_output[:150]}...")
    print(f"� Reference: {expected_output[:150]}...")
    print(f"📚 Context: {context[:150]}...")

    ollama_chat = ChatOllama(model=get_ollama_model(ModelType.EVALUATION), base_url=get_ollama_url())
    evaluator_model = LangchainLLMWrapper(ollama_chat)
    # NoiseSensitivity with mode='irrelevant' tests robustness to irrelevant/noisy contexts
    # According to RAGAS framework documentation, mode parameter specifies noise type:
    # - 'irrelevant': Tests sensitivity to irrelevant context (default behavior)
    noise_sensitivity = NoiseSensitivity(mode="irrelevant", llm=evaluator_model)

    test_data = SingleTurnSample(**test_data)
    finalscore = noise_sensitivity.single_turn_score(test_data)

    # Determine pass/fail based on NoiseSensitivity score threshold
    # Lower scores are better: 0.0 = no noise impact, higher = more sensitive to noise
    # Range: 0.0 (perfect robustness) to 1.0 (very sensitive to noise)
    threshold = 0.5
    if finalscore <= threshold:
        status = "✅ PASS"
    else:
        status = "❌ FAIL"
    
    print(f"NoiseSensitivity Score: {finalscore:.4f} | Threshold: {threshold} | {status}")

if __name__ == "__main__":
    print("RAGAS NoiseSensitivity Score Evaluation")
    print("=" * 50)
    
    setup_ollama()

    # Example test case
    user_query = "What is Harry Potter book series?"
    expected_output = generate_ollama_response(user_query, model_name=get_ollama_model(ModelType.EXPECTED_OUTPUT))
    context = retrieve_context_from_wiki("Harry Potter book series")

    test_noise_sensitivity(user_query, expected_output, context)