# RUNS WITH OPENAI - Requires OpenAI API key for GPT-4 evaluation

"""
DeepEval AnswerRelevancyMetric (LLM-Based with OpenAI GPT-4)
============================================================

What is AnswerRelevancyMetric?
- Measures what PROPORTION of the response directly addresses the input query
- Detects off-topic or tangential answers
- Uses OpenAI GPT-4 as judge
- Answers: "To what extent does the response address the query?"

How It Works:
- Takes: query, actual_output (LLM response)
- GPT-4 evaluates:
  - What proportion of the response directly addresses the question?
  - How much of the answer contains relevant information?
  - Is the output mostly on-topic or contains irrelevant sections?
- Outputs: Score from 0.0 to 1.0 (proportion of relevant content)

Score Interpretation (DeepEval Standard):
- Score Range: 0.0 to 1.0 (PROPORTION of response addressing query)
- 0.0         = Completely irrelevant - Off-topic or wrong subject entirely
- 0.0-0.3     = Mostly irrelevant (❌ FAIL) - ≤30% addresses query
- 0.3-0.5     = Partially relevant (⚠️ PARTIAL) - 30-50% addresses query
- 0.5-0.7     = Mostly relevant (✅ PASS) - 50-70% addresses query
- 0.7-1.0     = Highly relevant (✅ PASS) - ≥70% directly addresses query

Threshold: 0.5 (50% - MINIMUM passing threshold)
- Score must be ≥ 0.5 to PASS (at least half the response relevant)
- Higher scores are better: 1.0 = perfect relevance, 0.0 = irrelevant
- Interpretation: Score represents PROPORTION of response that addresses the query

Use Cases:
- Q&A system validation
- Chatbot accuracy assessment
- Search result relevance
- Customer support bot evaluation
- Semantic search quality
- RAG system answer quality

Requires: OPENAI_API_KEY environment variable

Reference: DeepEval Documentation
https://docs.depevalai.com/docs/metrics/answer-relevancy/
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, generate_ollama_response


def test_answer_relevancy(query):
    """
    Test AnswerRelevancyMetric - measures how relevant the LLM output is to the input query.
    
    Scoring:
    - Score ranges from 0 to 1
    - Score 1.0 = Highly relevant to query ✅ PASS
    - Score >= 0.5 = Reasonably relevant ✅ PASS
    - Score < 0.5 = Not relevant to query ❌ FAIL
    - Default threshold = 0.5
    
    AnswerRelevancyMetric evaluates:
    - Is the response directly addressing the query?
    - Does the answer contain information related to the question?
    - Is the output off-topic or completely unrelated?
    """
    
    # Generate response using local Ollama LLM
    ollama_response = generate_ollama_response(query)

    # Initialize AnswerRelevancyMetric
    answer_relevancy_metric = AnswerRelevancyMetric()
    
    # Create test case
    test_case = LLMTestCase(
        input=query,
        actual_output=ollama_response
    )

    print(f"Query: {query}")
    print(f"LLM Output: {ollama_response}")
    print("=" * 80)
    
    # Measure relevancy
    answer_relevancy_metric.measure(test_case)
    
    # Determine pass/fail based on relevancy score
    # AnswerRelevancyMetric: Score 1.0 = Fully relevant, Score 0.0 = Not relevant
    # threshold=0.5 is MINIMUM passing threshold (score >= 0.5 passes)
    if answer_relevancy_metric.score >= 0.5:
        print(f"✅ Test PASSED - Relevancy Score: {answer_relevancy_metric.score:.2f} (Output is relevant to query)")
    else:
        print(f"❌ Test FAILED - Relevancy Score: {answer_relevancy_metric.score:.2f} (Output is not relevant)")
        print(f"   Reason: {answer_relevancy_metric.reason}")


def test_answer_relevancy_custom(query, custom_answer):
    """
    Test with a custom answer for testing specific scenarios.
    Useful for testing failure cases or specific output patterns.
    """
    
    # Initialize AnswerRelevancyMetric
    answer_relevancy_metric = AnswerRelevancyMetric()
    
    # Create test case with custom answer
    test_case = LLMTestCase(
        input=query,
        actual_output=custom_answer,
    )

    print(f"Query: {query}")
    print(f"Custom Output: {custom_answer}")
    print("=" * 80)
    
    # Measure relevancy
    answer_relevancy_metric.measure(test_case)
    
    # Determine pass/fail based on relevancy score
    if answer_relevancy_metric.score >= 0.5:
        print(f"✅ Test PASSED - Relevancy Score: {answer_relevancy_metric.score:.2f}")
    else:
        print(f"❌ Test FAILED - Relevancy Score: {answer_relevancy_metric.score:.2f} (Output is not relevant)")
        print(f"   Reason: {answer_relevancy_metric.reason}")


if __name__ == "__main__":
    print("=" * 80)
    print("DEEPEVAL ANSWER RELEVANCY METRIC TEST - OpenAI GPT-4 Evaluator")
    print("=" * 80)
    print("\nAnswerRelevancyMetric Scoring:")
    print("  Score 1.0 = Highly relevant ✅ PASS")
    print("  Score >= 0.5 = Reasonably relevant ✅ PASS")
    print("  Score < 0.5 = Not relevant ❌ FAIL")
    print("  Default threshold = 0.5 (minimum score needed to pass)")
    print("\n" + "=" * 80)
    
    # Check if Ollama is running and start if needed
    setup_ollama()

    # Test 1: Direct factual question (Expected: ✅ PASS)
    print("\n📝 Test 1: Direct Factual Question")
    print("-" * 80)
    test_answer_relevancy("What is the capital of France?")
    
    # Test 2: Future event question (Expected: ✅ PASS - contextually relevant)
    print("\n📝 Test 2: Future Event Question")
    print("-" * 80)
    test_answer_relevancy("Who won the FIFA World Cup in 2099?")
    
    # Test 3: Completely off-topic answer (Expected: ❌ FAIL)
    print("\n📝 Test 3: Off-Topic Answer (Should FAIL)")
    print("-" * 80)
    test_answer_relevancy_custom(
        "What is the capital of France?",
        "Pizza is a delicious Italian dish made with dough, tomato sauce, and cheese. Popular toppings include pepperoni, mushrooms, and olives."
    )
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
