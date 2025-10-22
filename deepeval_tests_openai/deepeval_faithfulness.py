"""
DeepEval FaithfulnessMetric (LLM-Based with OpenAI GPT-4)
==========================================================

What is FaithfulnessMetric?
- Checks what PROPORTION of LLM response is FACTUALLY CONSISTENT with provided context
- Prevents hallucinations and made-up information
- Uses OpenAI GPT-4 as judge
- Answers: "What percentage of response claims are supported by context?"

How It Works:
- Takes: query, actual_output (LLM response), retrieval_context
- GPT-4 evaluates: "What proportion of claims in response are supported by context?"
- Outputs: Score from 0.0 to 1.0 (proportion of supported claims)

Score Interpretation (DeepEval Standard):
- Score Range: 0.0 to 1.0 (PROPORTION of claims supported by context)
- 0.0         = No faithful claims - All contradicts or ignores context
- 0.0-0.3     = Low faithfulness (❌ FAIL) - ≤30% of claims supported
- 0.3-0.5     = Partial faithfulness (⚠️ PARTIAL) - 30-50% of claims supported
- 0.5-0.7     = Good faithfulness (✅ PASS) - 50-70% of claims supported
- 0.7-1.0     = High faithfulness (✅ PASS) - ≥70% of claims supported by context

Threshold: 0.5 (50%)
- Minimum acceptable: At least 50% of claims must be supported by context
- Higher scores are better: 1.0 = all claims faithful, 0.0 = all unsupported
- Interpretation: Score represents PROPORTION of response claims supported by provided context

Use Cases:
- RAG system validation
- Fact-checking accuracy
- Hallucination detection
- Knowledge base consistency
- Production LLM evaluation

Requires: OPENAI_API_KEY environment variable

Reference: DeepEval Documentation
https://docs.depevalai.com/docs/metrics/faithfulness/
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, generate_ollama_response


def test_faithfulness(query, retrieval_context, threshold=0.5):
    """
    Test FaithfulnessMetric - checks if the LLM output is factually consistent with the provided context.
    
    Scoring:
    - Score ranges from 0 to 1
    - Score 1.0 = Fully faithful (output is consistent with context) ✅ PASS
    - Score >= threshold = PASS (meets minimum faithfulness requirement)
    - Score < threshold = FAIL (insufficient faithfulness)
    - Default threshold = 0.5
    """
    
    # Generate response using local Ollama LLM
    response = generate_ollama_response(query)
    
    # Initialize FaithfulnessMetric
    faithfulness_metric = FaithfulnessMetric(
        threshold=threshold,
        include_reason=True,
        async_mode=True,
        verbose_mode=False
    )
    
    # Create test case with required parameters
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=retrieval_context
    )
    
    print(f"Query: {query}")
    print(f"LLM Output: {response}")
    print(f"Context: {retrieval_context}")
    print("=" * 80)
    
    # Measure faithfulness
    faithfulness_metric.measure(test_case)
    
    # Determine pass/fail based on faithfulness score
    # DeepEval FaithfulnessMetric: Score 1.0 = Fully faithful (best), Score 0.0 = Not faithful (worst)
    # threshold=0.5 is MINIMUM passing threshold (score >= 0.5 passes, < 0.5 fails)
    if faithfulness_metric.score >= threshold:
        print(f"✅ Test PASSED - Faithfulness Score: {faithfulness_metric.score:.2f} (Output is faithful to context)")
    else:
        print(f"❌ Test FAILED - Faithfulness Score: {faithfulness_metric.score:.2f} (Output lacks faithfulness)")
        print(f"   Reason: {faithfulness_metric.reason}")


def test_faithfulness_with_factual_errors(threshold=0.5):
    """
    Test FaithfulnessMetric with output that contains factual errors.
    This should fail because the output contradicts the retrieval context.
    """
    
    # Define retrieval context with correct facts (concise)
    retrieval_context = [
        "The Great Wall of China is in northern China.",
        "It was built during the Ming Dynasty."
    ]
    
    # Create LLMTestCase with intentionally inaccurate output
    test_case = LLMTestCase(
        input="Where is the Great Wall of China?",
        actual_output="The Great Wall of China is located in southern China and was built by the Qin Dynasty.",
        retrieval_context=retrieval_context
    )
    
    # Initialize FaithfulnessMetric
    faithfulness_metric = FaithfulnessMetric(
        threshold=threshold,
        include_reason=True,
        async_mode=True,
        verbose_mode=False
    )
    
    print(f"Query: {test_case.input}")
    print(f"LLM Output: {test_case.actual_output}")
    print(f"Context: {retrieval_context}")
    print("=" * 80)
    
    # Measure faithfulness
    faithfulness_metric.measure(test_case)
    
    # Determine pass/fail based on faithfulness score
    if faithfulness_metric.score >= threshold:
        print(f"✅ Test PASSED - Faithfulness Score: {faithfulness_metric.score:.2f}")
    else:
        print(f"❌ Test FAILED - Faithfulness Score: {faithfulness_metric.score:.2f} (Output contains factual errors)")
        print(f"   Reason: {faithfulness_metric.reason}")


def test_faithfulness_partial_consistency(threshold=0.5):
    """
    Test FaithfulnessMetric with output that is partially consistent with context.
    Some claims are faithful, some are missing or exaggerated.
    """
    
    # Define retrieval context about Python (concise)
    retrieval_context = [
        "Python is a high-level programming language.",
        "It was created by Guido van Rossum in 1989."
    ]
    
    # Output has correct facts but adds information not in context
    test_case = LLMTestCase(
        input="What is Python?",
        actual_output="Python is a high-level programming language created by Guido van Rossum. It is the most popular programming language in the world.",
        retrieval_context=retrieval_context
    )
    
    # Initialize FaithfulnessMetric
    faithfulness_metric = FaithfulnessMetric(
        threshold=threshold,
        include_reason=True,
        async_mode=True,
        verbose_mode=False
    )
    
    print(f"Query: {test_case.input}")
    print(f"LLM Output: {test_case.actual_output}")
    print(f"Context: {retrieval_context}")
    print("=" * 80)
    
    # Measure faithfulness
    faithfulness_metric.measure(test_case)
    
    # Determine pass/fail based on faithfulness score
    if faithfulness_metric.score >= threshold:
        print(f"✅ Test PASSED - Faithfulness Score: {faithfulness_metric.score:.2f}")
    else:
        print(f"❌ Test FAILED - Faithfulness Score: {faithfulness_metric.score:.2f} (Threshold: {threshold})")
        print(f"   Reason: {faithfulness_metric.reason}")


if __name__ == "__main__":
    print("=" * 80)
    print("DEEPEVAL FAITHFULNESS METRIC TEST - OpenAI GPT-4 Evaluator")
    print("=" * 80)
    print("\nFaithfulnessMetric Scoring:")
    print("  Score 1.0 = Fully faithful ✅ PASS")
    print("  Score >= 0.5 = Meets threshold ✅ PASS")
    print("  Score < 0.5 = Below threshold ❌ FAIL")
    print("  Default threshold = 0.5 (minimum score needed to pass)")
    print("\n" + "=" * 80)
    
    # Check if Ollama is running and start if needed
    setup_ollama()
    
    # Test 1: Faithful output (LLM-generated)
    print("\n📝 Test 1: Faithful Output (LLM-generated from Ollama)")
    print("-" * 80)
    retrieval_context_1 = [
        "Paris is the capital of France.",
        "The Eiffel Tower is in Paris."
    ]
    test_faithfulness("What is Paris?", retrieval_context_1, threshold=0.5)
    
    # Test 2: Intentionally inaccurate output (should fail)
    print("\n📝 Test 2: Factually Incorrect Output (Should FAIL)")
    print("-" * 80)
    test_faithfulness_with_factual_errors(threshold=0.5)
    
    # Test 3: Partially faithful output
    print("\n📝 Test 3: Partially Faithful Output (Some facts missing/added)")
    print("-" * 80)
    test_faithfulness_partial_consistency(threshold=0.5)
    
    # Test 4: Same test with higher threshold
    print("\n📝 Test 4: Partially Faithful Output with Higher Threshold (0.8)")
    print("-" * 80)
    test_faithfulness_partial_consistency(threshold=0.8)
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
