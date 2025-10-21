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
