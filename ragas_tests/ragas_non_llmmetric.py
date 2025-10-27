# RUNS LOCALLY - Uses local metrics, no API keys required

"""
RAGAS BLEU Score Metric (Non-LLM Based)
========================================

⚠️ IMPORTANT: Non-LLM metrics are LESS RELIABLE than LLM-based metrics

What is BLEU Score?
- Surface-level metric based on n-gram overlap (1-grams, 2-grams, etc.)
- Compares response with reference using string matching
- Does NOT understand semantic meaning or context
- Originally designed for machine translation evaluation

How It Works:
- Counts matching n-grams between response and reference
- Calculates precision for each n-gram size
- Outputs: Score from 0.0 to 1.0 (proportion of matching n-grams)

Score Interpretation (RAGAS Standard):
- Score Range: 0.0 to 1.0 (PROPORTION of n-grams matching)
- 0.0         = No overlap - Response completely different from reference
- 0.0-0.2     = Poor match (❌ FAIL) - ≤20% overlap, very different
- 0.2-0.4     = Fair match (⚠️ PARTIAL) - 20-40% overlap
- 0.4-0.6     = Good match (✅ PASS) - 40-60% overlap, significant similarity
- 0.6-1.0     = Excellent match (✅ PASS) - ≥60% overlap, very similar

Threshold: 0.5 (50%)
- Minimum acceptable for BLEU Score: ≥50% n-gram overlap
- Higher scores are better: 1.0 = identical, 0.0 = completely different
- Interpretation: Score represents PROPORTION of n-grams matching between response and reference

Limitations:
❌ No semantic understanding (synonyms treated as different)
❌ Penalizes valid paraphrasing
❌ Can give high scores to grammatically incorrect but similar text
❌ Not suitable for evaluating response quality alone

Use Cases (Limited):
- Quick sanity checks only
- Prototyping and testing
- Baseline comparison
- NOT for production evaluation

Best Practice:
Use LLM-based metrics (Faithfulness, AnswerRelevancy, ContextRecall) for reliable evaluation

Reference: RAGAS Documentation
https://docs.ragas.io/en/latest/concepts/metrics/
"""

import sys
from pathlib import Path
from ragas import SingleTurnSample
from ragas.metrics import BleuScore

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, generate_ollama_response
from utils.config import get_ollama_model, get_ollama_url, ModelType


def test_single_query_blue_score(user_query, expected_output):
    """Test query with Blue Score metric."""
    
    # Generate response from LLM under test
    actual_output = generate_ollama_response(user_query, model_name=get_ollama_model(ModelType.ACTUAL_OUTPUT))
    test_data = {
        "user_input": user_query,
        "response": actual_output,
        "reference": expected_output
    }

    print(f"\n📝 Query: {user_query}")
    print(f"💬 Response: {actual_output[:150]}...")
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
    expected_output = generate_ollama_response(user_query, model_name=get_ollama_model(ModelType.EXPECTED_OUTPUT))
    
    test_single_query_blue_score(user_query, expected_output)
    
    print("=" * 50)