# RUNS LOCALLY - Uses Ollama for local LLM inference, no API keys required

"""
DeepEval AnswerRelevancyMetric (LLM-Based with Local Ollama)
===========================================================

What is AnswerRelevancyMetric (Local)?
- Measures what PROPORTION of the LLM response directly addresses the input query
- Detects off-topic or tangential answers
- Uses LOCAL Ollama model (config-defined evaluation model) as judge
- Answers: "What percentage of the response addresses the query?"

How It Works:
- Takes: query, actual_output (LLM response)
- Local Ollama evaluator analyzes:
  - What proportion of the response directly addresses the question?
  - How much of the answer contains relevant information?
  - What percentage is on-topic vs completely unrelated?
- Outputs: Score from 0.0 to 1.0 (proportion of relevant content)

Score Interpretation (DeepEval Standard):
- Score Range: 0.0 to 1.0 (PROPORTION of response addressing query)
- 0.0         = Completely irrelevant - Off-topic or wrong subject entirely
- 0.0-0.3     = Mostly irrelevant (❌ FAIL) - ≤30% addresses query
- 0.3-0.5     = Partially relevant (⚠️ PARTIAL) - 30-50% addresses query
- 0.5-0.7     = Mostly relevant (✅ PASS) - 50-70% addresses query
- 0.7-1.0     = Highly relevant (✅ PASS) - ≥70% directly addresses query

Threshold: 0.5 (50% - MINIMUM passing threshold)
- Score must be >= 0.5 to PASS (at least half relevant)
- Higher scores are better: 1.0 = perfect relevance, 0.0 = irrelevant
- Interpretation: Score represents PROPORTION of response addressing the query

Local vs OpenAI:
- Uses local Ollama model for evaluation (free, offline)
- No API costs or network dependency
- May have lower accuracy than GPT-4
- Runs completely in your environment

Use Cases:
- Q&A system validation (offline)
- Chatbot accuracy assessment (private)
- Search result relevance
- Cost-effective evaluation
- Privacy-critical evaluations
- Testing before OpenAI deployment

Requires: Ollama running with config-defined evaluation model

Reference: DeepEval Documentation
https://docs.depevalai.com/docs/metrics/answer-relevancy/
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, setup_custom_ollama_model_for_evaluation, generate_ollama_response


def test_answer_relevancy(evaluationModel, query):
    """
    Test AnswerRelevancyMetric with Local LLM Evaluator - measures relevance using local Ollama.
    
    Scoring:
    - Score ranges from 0 to 1
    - Score 1.0 = Highly relevant to query ✅ PASS
    - Score >= 0.5 = Reasonably relevant ✅ PASS
    - Score < 0.5 = Not relevant to query ❌ FAIL
    - Default threshold = 0.5
    
    Local vs OpenAI:
    - Uses local Ollama model for evaluation instead of OpenAI GPT-4
    - Runs completely offline
    - Free to use but may have lower accuracy
    """

    # Generate response using local Ollama LLM
    ollama_response = generate_ollama_response(query)
    
    # Initialize AnswerRelevancyMetric with local evaluator
    answer_relevancy_metric = AnswerRelevancyMetric(model=evaluationModel)
    
    # Create test case
    test_case = LLMTestCase(
        input=query,
        actual_output=ollama_response,
    )

    print(f"Query: {query}")
    print(f"LLM Output: {ollama_response}")
    print(f"Evaluator Model: {evaluationModel}")
    print("=" * 80)
    
    # Measure relevancy
    answer_relevancy_metric.measure(test_case)
    
    # Determine pass/fail based on relevancy score
    if answer_relevancy_metric.score >= 0.5:
        print(f"✅ Test PASSED - Relevancy Score: {answer_relevancy_metric.score:.2f}")
    else:
        print(f"❌ Test FAILED - Relevancy Score: {answer_relevancy_metric.score:.2f}")
        print(f"   Reason: {answer_relevancy_metric.reason}")


def test_answer_relevancy_custom(evaluationModel, query, custom_answer):
    """
    Test with a custom answer for testing specific scenarios.
    Useful for testing failure cases or specific output patterns with local evaluator.
    """
    
    # Initialize AnswerRelevancyMetric with local evaluator
    answer_relevancy_metric = AnswerRelevancyMetric(model=evaluationModel)
    
    # Create test case with custom answer
    test_case = LLMTestCase(
        input=query,
        actual_output=custom_answer,
    )

    print(f"Query: {query}")
    print(f"Custom Output: {custom_answer}")
    print(f"Evaluator Model: {evaluationModel}")
    print("=" * 80)
    
    # Measure relevancy
    answer_relevancy_metric.measure(test_case)
    
    # Determine pass/fail based on relevancy score
    if answer_relevancy_metric.score >= 0.5:
        print(f"✅ Test PASSED - Relevancy Score: {answer_relevancy_metric.score:.2f}")
    else:
        print(f"❌ Test FAILED - Relevancy Score: {answer_relevancy_metric.score:.2f}")
        print(f"   Reason: {answer_relevancy_metric.reason}")


if __name__ == "__main__":
    print("=" * 80)
    print("DEEPEVAL ANSWER RELEVANCY METRIC TEST - Local Ollama Evaluator")
    print("=" * 80)
    print("\nAnswerRelevancyMetric Scoring (with Local LLM):")
    print("  Score 1.0 = Highly relevant ✅ PASS")
    print("  Score >= 0.5 = Reasonably relevant ✅ PASS")
    print("  Score < 0.5 = Not relevant ❌ FAIL")
    print("  Default threshold = 0.5")
    print("\nNote: Using local Ollama for evaluation instead of OpenAI GPT-4")
    print("\n" + "=" * 80)

    # Check if Ollama is running and start if needed
    setup_ollama()
    
    # Set local LLM as evaluation judge model
    evaluationModel = setup_custom_ollama_model_for_evaluation()

    # Test 1: Direct factual question (Expected: ✅ PASS)
    print("\n📝 Test 1: Direct Factual Question")
    print("-" * 80)
    test_answer_relevancy(evaluationModel, "What is the capital of France?")
    
    # Test 2: Future event question (Expected: ✅ PASS)
    print("\n📝 Test 2: Future Event Question")
    print("-" * 80)
    test_answer_relevancy(evaluationModel, "Who won the FIFA World Cup in 2099?")
    
    # Test 3: Completely off-topic answer (Expected: ❌ FAIL)
    print("\n📝 Test 3: Off-Topic Answer (Should FAIL)")
    print("-" * 80)
    test_answer_relevancy_custom(
        evaluationModel,
        "What is the capital of France?",
        "Pizza is a delicious Italian dish made with dough, tomato sauce, and cheese. Popular toppings include pepperoni, mushrooms, and olives."
    )
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
