from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import ollama
import os
from dotenv import load_dotenv
from local_llm_ollama_setup import setup_ollama

# Load environment variables from .env file
load_dotenv()


# Set OpenAI API key in environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

# Set OpenAI API key for deepeval's GEval metric (which uses OpenAI for evaluation)
os.environ["OPENAI_API_KEY"] = openai_api_key

def test_correctness(threshold=1.0):
    # Check if Ollama is running and start if needed
    setup_ollama()

    response = ollama.chat(model='llama3.2:3b', messages=[
        {
            'role': 'user',
            'content': 'Who is the president of the United States as of 2024?'
        }
    ])
    actual_output = response['message']['content']

    correctness_metric = GEval(
        name="test_correctness",
        criteria="Determine if the actual output matches the expected output",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=threshold,
    )
    test_case = LLMTestCase(
        input="Who is the president of the United States as of 2024?",
        actual_output=actual_output,
        expected_output="As of 2024, the president of the United States is Joe Biden.",
    )
    print("Actual Output:", actual_output)
    
    try:
        assert_test(test_case, [correctness_metric])
        print("✅ Test Passed!")
    except AssertionError as e:
        print("\n❌ Test Failed!")
        print(f"\nError Details:\n{str(e)}")
        
        # Extract and display metric details
        for metric in [correctness_metric]:
            print(f"\n📊 Metric: {metric.name}")
            print(f"   Score: {metric.score}")
            print(f"   Threshold: {metric.threshold}")
            print(f"   Reason: {metric.reason}")

if __name__ == "__main__":
    # Test 1: Threshold 1.0 - Will FAIL (score is typically ~0.28-0.3, needs 1.0)
    # LLM output includes extra context, not exact match
    print("=" * 60)
    print("Test 1: Strict Match (threshold=1.0)")
    print("Expected Result: ❌ WILL FAIL - LLM adds extra context")
    print("=" * 60)
    test_correctness()
    
    # Test 2: Threshold 0.8 - Will FAIL (score is typically ~0.28-0.3, needs 0.8)
    # Still won't meet the 0.8 threshold requirement
    print("\n" + "=" * 60)
    print("Test 2: High Threshold (threshold=0.8)")
    print("Expected Result: ❌ WILL FAIL - Score doesn't reach 0.8")
    print("=" * 60)
    test_correctness(threshold=0.8)
    
    # Test 3: Threshold 0.5 - Will FAIL (score is typically ~0.28-0.3, needs 0.5)
    # Score still below threshold
    print("\n" + "=" * 60)
    print("Test 3: Medium Threshold (threshold=0.5)")
    print("Expected Result: ❌ WILL FAIL - Score doesn't reach 0.5")
    print("=" * 60)
    test_correctness(threshold=0.5)
    
    # Test 4: Threshold 0.0 - Will PASS (any positive score passes)
    # Any non-zero score will pass this threshold
    print("\n" + "=" * 60)
    print("Test 4: No Threshold (threshold=0.0)")
    print("Expected Result: ✅ WILL PASS - Any score above 0 passes")
    print("=" * 60)
    test_correctness(threshold=0.0)