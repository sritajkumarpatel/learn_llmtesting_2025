"""
⚠️ NON-LLM METRICS - IMPORTANT NOTES:
- BLEU Score is a surface-level metric based on n-gram overlap
- Does NOT understand semantic meaning or context
- Scores tend to be lower compared to LLM-based metrics
- NOT recommended for production evaluation
- Use only for quick testing and prototyping
- For reliable evaluation, use LLM-based metrics (FaithfulnessMetric, AnswerRelevancy, etc.)
"""

import sys
from pathlib import Path
from ragas import SingleTurnSample
from ragas.metrics import BleuScore

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, setup_custom_ollama_model_for_evaluation, generate_ollama_response


def test_single_query_blue_score(user_query, expected_output, use_custom_response=False, custom_response=None):
    """Test query with Blue Score metric."""
    
    # Generate response from LLM under test
    ollama_response = generate_ollama_response(user_query, model_name="llama3.1:8b")
    
    test_data = {
        "input": user_query,
        "response": ollama_response,
        "reference": expected_output
    }

    print(f"\n📝 Query: {user_query}")
    print(f"💬 Response: {ollama_response[:150]}...")
    print(f"💬 Reference: {expected_output[:150]}...")

    metric = BleuScore()
    test_data = SingleTurnSample(**test_data)
    finalscore = metric.single_turn_score(test_data)
    
    # Determine pass/fail based on BLEU score threshold
    threshold = 0.5
    if finalscore >= threshold:
        status = "✅ PASS"
    else:
        status = "❌ FAIL"
    
    print(f"BLEU Score: {finalscore:.4f} | Threshold: {threshold} | {status}")

if __name__ == "__main__":
    print("RAGAS BLEU Score Evaluation")
    print("=" * 50)
    
    setup_ollama()

    # Example test case
    user_query = "What is the capital of France?"
    expected_output = generate_ollama_response(user_query, model_name="deepseek-r1:8b")
    
    test_single_query_blue_score(user_query, expected_output)
    
    print("=" * 50)