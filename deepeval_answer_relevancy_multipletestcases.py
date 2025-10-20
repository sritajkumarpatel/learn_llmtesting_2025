from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import ollama
from local_llm_ollama_setup import setup_ollama, setup_custom_ollama_model_for_evaluation, generate_ollama_response
import os
from deepeval import evaluate

# TODO: This file needs to be tested to ensure all test cases work as expected
def test_answer_relevancy(queries, evaluationModel):
    """
    Test multiple queries (3 test cases) in batch evaluation
    
    Args:
        queries: List of 3 different questions
        evaluationModel: The evaluation model to use
    """
    
    answer_relevancy_metric = AnswerRelevancyMetric(model=evaluationModel)
    test_cases = []
    
    # Generate responses and create test cases for each query
    for query in queries:
        ollama_response = generate_ollama_response(query, 'llama3.2:3b')
        print(f"Query: {query}")
        print(f"Output: {ollama_response}\n")
        
        test_cases.append(LLMTestCase(
            input=query,
            actual_output=ollama_response,
        ))
    
    # Batch evaluate all 3 test cases
    result = evaluate(test_cases, metrics=[answer_relevancy_metric])
    print("Batch Evaluation Result:", result)



if __name__ == "__main__":

    # Check if Ollama is running and start if needed
    setup_ollama()
    # Set local LLM as evaluation judge model
    evaluationModel = setup_custom_ollama_model_for_evaluation()

    # Series of test questions (grouped in sets of 3)
    test_questions = [
        "What is the capital of France?",
        "Who won the FIFA World Cup in 2099?",
        "How does photosynthesis work?",
        "What are the benefits of renewable energy?",
        "Explain quantum computing in simple terms",
        "What is the meaning of life?"
    ]
    
    # Test batch 1: First 3 questions
    print("=" * 70)
    print("Batch 1: Testing 3 questions together")
    print("=" * 70)
    test_answer_relevancy(test_questions[:3], evaluationModel)
    
    print("\n" + "=" * 70)
    print("Batch 2: Testing remaining 2 questions")
    print("=" * 70)
    test_answer_relevancy(test_questions[3:], evaluationModel)
    
