"""
DeepEval GEval (Generalized Evaluation - LLM-Based with OpenAI GPT-4)
====================================================================

What is GEval?
- CUSTOM, FLEXIBLE LLM-based evaluation metric
- Define your OWN evaluation criteria in natural language
- Uses OpenAI GPT-4 as judge
- Answers: "Does the response match my custom evaluation criteria?"

How It Works:
- You define evaluation criteria (e.g., "Is the answer factually correct?")
- You specify evaluation parameters (what to compare)
- GPT-4 scores response based on YOUR criteria
- Returns score 0.0-1.0 (or custom scale 0-10 with rubrics)

Score Interpretation (Flexible - depends on criteria):
- 0.0-0.3   = Does not meet criteria (❌ FAIL)
- 0.3-0.5   = Partially meets criteria (⚠️ PARTIAL)
- 0.5-0.7   = Mostly meets criteria (✅ PASS)
- 0.7-1.0   = Fully meets criteria (✅ PASS)

Threshold: 0.5 (50% - MINIMUM passing threshold)
- Score must be >= 0.5 to PASS
- Rationale: Custom criteria should have at least 50% satisfaction
- Adjustable per evaluation need

Key Features:
- Custom criteria definition (your business rules)
- Multiple evaluation parameters (input, output, expected, context)
- Flexible scoring (0-1 default, 0-10 with rubrics)
- Rubric support for detailed scoring guides
- Best for domain-specific evaluations

Use Cases:
- Domain-specific quality checks
- Business rule validation
- Custom compliance requirements
- Specialized domain evaluations
- Multi-criteria scoring
- Complex reasoning assessment

Requires: OPENAI_API_KEY environment variable

Reference: DeepEval Documentation
https://docs.depevalai.com/docs/metrics/g-eval/
"""

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
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
    """
    Test GEval Metric - Custom LLM-based evaluation with configurable criteria.
    
    Scoring:
    - Score ranges from 0 to 1 (can be customized with rubrics to 0-10)
    - Higher score = Better match to criteria ✅ PASS
    - Lower score = Worse match to criteria ❌ FAIL
    - Threshold is MAXIMUM acceptable score (score <= threshold passes, > threshold fails)
    
    GEval allows:
    - Custom evaluation criteria (natural language description)
    - Multiple evaluation parameters (input, actual_output, expected_output, etc.)
    - Flexible scoring system with custom rubrics
    """
    
    # Generate response using local Ollama LLM
    response = generate_ollama_response('Who is the president of the United States as of 2024?')

    # Initialize GEval with evaluation criteria
    correctness_metric = GEval(
        name="test_correctness",
        criteria="Determine if the actual output matches the expected output",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=threshold
    )
    
    # Create test case with required parameters
    test_case = LLMTestCase(
        input="Who is the president of the United States as of 2024?",
        actual_output=response,
        expected_output="As of 2024, the president of the United States is Joe Biden.",
    )
    
    print(f"Query: {test_case.input}")
    print(f"LLM Output: {response}")
    print(f"Expected Output: {test_case.expected_output}")
    print("=" * 80)
    
    try:
        assert_test(test_case, [correctness_metric])
        print(f"✅ Test Passed! (Score: {correctness_metric.score:.2f} <= Threshold: {correctness_metric.threshold})")
    except AssertionError as e:
        print(f"❌ Test Failed! (Score: {correctness_metric.score:.2f} > Threshold: {correctness_metric.threshold})")
        print(f"   Reason: {correctness_metric.reason}")

if __name__ == "__main__":
    print("=" * 80)
    print("DEEPEVAL GEVAL METRIC TEST - OpenAI GPT-4 Evaluator")
    print("=" * 80)
    print("\nGEval Metric Scoring:")
    print("  Score 1.0 = Perfect match ✅ BEST")
    print("  Score 0.5 = Partial match ⚠️ BORDERLINE")
    print("  Score 0.0 = No match ❌ WORST")
    print("  Threshold is MAXIMUM acceptable score (score <= threshold passes)")
    print("\n" + "=" * 80)
    
    # Check if Ollama is running and start if needed
    setup_ollama()

    # Test 1: Threshold 1.0 (Very strict - Will likely FAIL)
    print("\n📝 Test 1: Very Strict Threshold (1.0) - Expected: ❌ FAIL")
    print("-" * 80)
    test_correctness(threshold=1.0)
    
    # Test 2: Threshold 0.8 (Strict - Will likely FAIL)
    print("\n📝 Test 2: Strict Threshold (0.8) - Expected: ❌ FAIL")
    print("-" * 80)
    test_correctness(threshold=0.8)
    
    # Test 3: Threshold 0.5 (Moderate - Will likely FAIL)
    print("\n📝 Test 3: Moderate Threshold (0.5) - Expected: ❌ FAIL")
    print("-" * 80)
    test_correctness(threshold=0.5)
    
    # Test 4: Threshold 0.0 (Very lenient - Will likely PASS)
    print("\n📝 Test 4: Lenient Threshold (0.0) - Expected: ✅ PASS")
    print("-" * 80)
    test_correctness(threshold=0.0)
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
