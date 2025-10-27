# RUNS LOCALLY - Uses Ollama for local LLM inference, no API keys required

"""
DeepEval AnswerRelevancyMetric - Batch Testing (Local Ollama)
=============================================================

What is Batch AnswerRelevancyMetric Testing?
- Evaluates MULTIPLE queries in one batch operation
- Measures how relevant LLM responses are to input queries
- Uses LOCAL Ollama model (deepseek-r1:8b) as judge
- Answers: "Do all responses directly address their queries?"

How It Works:
- Takes: List of queries
- For each query: Generate response and create test case
- Use batch evaluation instead of individual tests
- Generates scores for all queries at once

Score Interpretation (DeepEval Standard):
- 0.0-0.3   = Irrelevant (❌ FAIL) - Off-topic or wrong subject
- 0.3-0.5   = Marginally relevant (⚠️ PARTIAL) - Some related info
- 0.5-0.7   = Relevant (✅ PASS) - Mostly addresses query
- 0.7-1.0   = Highly relevant (✅ PASS) - Perfect answer

Threshold: 0.5 (50% - MINIMUM passing threshold)
- Score must be >= 0.5 to PASS
- Rationale: Response must meaningfully address user question
- Report: Overall pass rate across all queries

Batch vs Individual Evaluation:
- Batch: Evaluate all queries together, more efficient
- Individual: Evaluate one query at a time, more granular feedback
- Batch: Better for comprehensive testing scenarios
- Batch: Aggregates results into summary statistics

Use Cases:
- Comprehensive Q&A system testing
- Batch quality assurance of multiple queries
- Regression testing (verify performance doesn't degrade)
- Chatbot accuracy across various questions
- Testing search relevance on multiple queries

Local Evaluation:
- Uses local Ollama model (free, offline)
- No API costs or network dependency
- Runs completely in your environment
- Good for rapid iteration and large-scale testing

Aggregation Metrics:
- % PASS: Percentage of queries that pass (score >= 0.5)
- Avg Score: Average relevancy score across all queries
- Min/Max: Range of scores observed

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
from deepeval import evaluate


def test_answer_relevancy(queries, evaluationModel):
    """
    Test multiple queries in batch evaluation with local Ollama evaluator.
    
    Scoring:
    - Score ranges from 0 to 1
    - Score 1.0 = Highly relevant to query ✅ PASS
    - Score >= 0.5 = Reasonably relevant ✅ PASS
    - Score < 0.5 = Not relevant to query ❌ FAIL
    
    Batch Evaluation:
    - Evaluates multiple test cases together
    - More efficient than individual evaluations
    - Useful for comprehensive testing of multiple scenarios
    - Uses local Ollama model for evaluation
    
    Args:
        queries: List of query strings to test
        evaluationModel: The local evaluation model to use (e.g., deepseek-r1:8b)
    """
    
    print(f"\n📝 Testing {len(queries)} queries in batch...")
    print("-" * 80)
    
    # Initialize metric
    answer_relevancy_metric = AnswerRelevancyMetric(model=evaluationModel)
    test_cases = []
    
    # Generate responses and create test cases for each query
    for i, query in enumerate(queries, 1):
        ollama_response = generate_ollama_response(query)
        print(f"\n  {i}. Query: {query}")
        print(f"     Output: {ollama_response[:100]}...")  # Show first 100 chars
        
        test_cases.append(LLMTestCase(
            input=query,
            actual_output=ollama_response,
        ))
    
    print("\n" + "=" * 80)
    print(f"Running batch evaluation on {len(test_cases)} test cases...")
    print("=" * 80)
    
    # Batch evaluate all test cases
    result = evaluate(test_cases, metrics=[answer_relevancy_metric])
    
    print(f"\n✅ Batch Evaluation Complete!")
    print(f"   Overall Result: {result}")
    print(f"   Evaluator Model: {evaluationModel}")



if __name__ == "__main__":
    print("=" * 80)
    print("DEEPEVAL ANSWER RELEVANCY - BATCH EVALUATION WITH MULTIPLE TEST CASES")
    print("=" * 80)
    print("\nBatch Evaluation with Local Ollama Evaluator:")
    print("  - Tests multiple queries together for efficiency")
    print("  - Uses AnswerRelevancyMetric with local LLM evaluator")
    print("  - Score >= 0.5 = PASS ✅ | Score < 0.5 = FAIL ❌")
    print("\n" + "=" * 80)

    # Check if Ollama is running and start if needed
    setup_ollama()
    
    # Set local LLM as evaluation judge model
    evaluationModel = setup_custom_ollama_model_for_evaluation()

    # Series of test questions
    test_questions = [
        "What is the capital of France?",
        "Who won the FIFA World Cup in 2099?",
        "How does photosynthesis work?",
        "What are the benefits of renewable energy?",
        "Explain quantum computing in simple terms",
    ]
    
    # Test batch 1: First 3 questions
    print("\nBatch 1: Testing First 3 Questions")
    print("-" * 80)
    test_answer_relevancy(test_questions[:3], evaluationModel)
    
    # Test batch 2: Remaining questions
    print("\n\nBatch 2: Testing Remaining Questions")
    print("-" * 80)
    test_answer_relevancy(test_questions[3:], evaluationModel)
    
    print("\n" + "=" * 80)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 80)
