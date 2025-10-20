from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import ollama
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, generate_ollama_response

# Load environment variables from .env file
load_dotenv()


# Set OpenAI API key in environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

# Set OpenAI API key for deepeval's GEval metric (which uses OpenAI for evaluation)
os.environ["OPENAI_API_KEY"] = openai_api_key

def test_correctness(threshold=1.0):

    # Generate response using local Ollama LLM
    response = generate_ollama_response('Who is the president of the United States as of 2024?')

    correctness_metric = GEval(
        name="test_correctness",
        criteria="Determine if the actual output matches the expected output",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=threshold,
        openai_api_key=openai_api_key,
    )
    test_case = LLMTestCase(
        input="Who is the president of the United States as of 2024?",
        actual_output=response,
        expected_output="As of 2024, the president of the United States is Joe Biden.",
    )
    print("Actual Output:", response)
    
    try:
        assert_test(test_case, [correctness_metric])
        print(f"✅ Test Passed! (Score: {correctness_metric.score} >= Threshold: {correctness_metric.threshold})")
    except AssertionError as e:
        print(f"❌ Test Failed! (Score: {correctness_metric.score} < Threshold: {correctness_metric.threshold})")
        print(f"   Reason: {correctness_metric.reason}")

if __name__ == "__main__":

    # Check if Ollama is running and start if needed
    setup_ollama()

    # Test 1: Threshold 1.0 - Will FAIL
    print("Test 1 (threshold=1.0) - Expected: ❌ FAIL")
    test_correctness()
    
    # Test 2: Threshold 0.8 - Will FAIL
    print("\nTest 2 (threshold=0.8) - Expected: ❌ FAIL")
    test_correctness(threshold=0.8)
    
    # Test 3: Threshold 0.5 - Will FAIL
    print("\nTest 3 (threshold=0.5) - Expected: ❌ FAIL")
    test_correctness(threshold=0.5)
    
    # Test 4: Threshold 0.0 - Will PASS
    print("\nTest 4 (threshold=0.0) - Expected: ✅ PASS")
    test_correctness(threshold=0.0)
