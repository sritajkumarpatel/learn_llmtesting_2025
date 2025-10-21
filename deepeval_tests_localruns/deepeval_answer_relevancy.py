"""
DeepEval AnswerRelevancyMetric (LLM-Based with Local Ollama)
===========================================================

What is AnswerRelevancyMetric (Local)?
- Measures how RELEVANT the LLM response is to the input query
- Detects off-topic or tangential answers
- Uses LOCAL Ollama model (deepseek-r1:8b) as judge
- Answers: "Does the response directly address the query?"

How It Works:
- Takes: query, actual_output (LLM response)
- Local Ollama evaluator analyzes:
  - Is the response directly addressing the question?
  - Does the answer contain relevant information?
  - Is the output on-topic or completely unrelated?
- Outputs: Score from 0.0 to 1.0

Score Interpretation (DeepEval Standard):
- 0.0-0.3   = Irrelevant (❌ FAIL) - Off-topic or wrong subject
- 0.3-0.5   = Marginally relevant (⚠️ PARTIAL) - Some related info
- 0.5-0.7   = Relevant (✅ PASS) - Mostly addresses query
- 0.7-1.0   = Highly relevant (✅ PASS) - Perfect answer to query

Threshold: 0.5 (50% - MINIMUM passing threshold)
- Score must be >= 0.5 to PASS
- Rationale: Response must meaningfully address user's question
- 0.5 threshold: Ensures at least reasonable relevance

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

Requires: Ollama running with deepseek-r1:8b model

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
