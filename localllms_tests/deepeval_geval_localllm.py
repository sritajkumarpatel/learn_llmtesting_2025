from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, generate_ollama_response, setup_custom_ollama_model_for_evaluation


def test_correctness(evaluationModel, threshold=1.0):
    """
    Test GEval Metric with Local LLM Evaluator - Custom evaluation using a local Ollama model as judge.
    
    Scoring:
    - Score ranges from 0 to 1 (can be customized with rubrics to 0-10)
    - Higher score = Better match to criteria ✅ PASS
    - Lower score = Worse match to criteria ❌ FAIL
    - Threshold is MAXIMUM acceptable score (score <= threshold passes, > threshold fails)
    
    Local vs OpenAI:
    - Uses local Ollama model instead of OpenAI GPT-4
    - Cost-effective and runs offline
    - May have lower evaluation accuracy than GPT-4
    """

    # Generate response using local Ollama LLM (generation model)
    response = generate_ollama_response('What is the capital of India?')

    # Initialize GEval with local LLM as evaluator
    correctness_metric = GEval(
        name="test_correctness",
        criteria="Determine if the actual output matches the expected output",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=threshold,
        model=evaluationModel
    )
    
    # Create test case
    test_case = LLMTestCase(
        input="What is the capital of India?",
        actual_output=response,
        expected_output="The capital of India is New Delhi.",
    )
    
    print(f"Query: {test_case.input}")
    print(f"LLM Output: {response}")
    print(f"Expected Output: {test_case.expected_output}")
    print(f"Evaluator Model: {evaluationModel}")
    print("=" * 80)
    
    try:
        assert_test(test_case, [correctness_metric])
        print(f"✅ Test PASSED! (Score: {correctness_metric.score:.2f} <= Threshold: {correctness_metric.threshold})")
    except AssertionError as e:
        print(f"❌ Test FAILED! (Score: {correctness_metric.score:.2f} > Threshold: {correctness_metric.threshold})")
        print(f"   Reason: {correctness_metric.reason}")

if __name__ == "__main__":
    print("=" * 80)
    print("DEEPEVAL GEVAL METRIC TEST - Local Ollama Evaluator")
    print("=" * 80)
    print("\nGEval Metric Scoring (with Local LLM):")
    print("  Score 1.0 = Perfect match ✅ BEST")
    print("  Score 0.5 = Partial match ⚠️ BORDERLINE")
    print("  Score 0.0 = No match ❌ WORST")
    print("  Threshold is MAXIMUM acceptable score (score <= threshold passes)")
    print("\nNote: Using local Ollama for evaluation instead of OpenAI GPT-4")
    print("\n" + "=" * 80)
    
    # Check if Ollama is running and start if needed
    setup_ollama()
    
    # Set local LLM as evaluation judge model
    evaluationModel = setup_custom_ollama_model_for_evaluation()

    # Test 1: Threshold 1.0 (Very strict - Will likely FAIL)
    print("\n📝 Test 1: Very Strict Threshold (1.0) - Expected: ❌ FAIL")
    print("-" * 80)
    test_correctness(evaluationModel, threshold=1.0)
    
    # Test 2: Threshold 0.8 (Strict - Will likely FAIL)
    print("\n📝 Test 2: Strict Threshold (0.8) - Expected: ❌ FAIL")
    print("-" * 80)
    test_correctness(evaluationModel, threshold=0.8)
    
    # Test 3: Threshold 0.5 (Moderate - Will likely FAIL)
    print("\n📝 Test 3: Moderate Threshold (0.5) - Expected: ❌ FAIL")
    print("-" * 80)
    test_correctness(evaluationModel, threshold=0.5)
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
