from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import ollama
from local_llm_ollama_setup import setup_ollama, setup_custom_ollama_model_for_evaluation, generate_ollama_response
import os

def test_answer_relevancy(evaluationModel, query):

      # Generate response using local Ollama LLM
    ollama_response = generate_ollama_response(query)
    
    answer_relevancy_metric = AnswerRelevancyMetric(model=evaluationModel)
    test_case = LLMTestCase(
        input=query,
        actual_output=ollama_response,
    )

    print("Actual Output:", ollama_response)
    answer_relevancy_metric.measure(test_case)
    print("Answer Relevancy Score:", answer_relevancy_metric.score)


def test_answer_relevancy_custom(evaluationModel, query, custom_answer):
    """Test with a custom answer (useful for testing failure cases)"""
    answer_relevancy_metric = AnswerRelevancyMetric(model=evaluationModel)
    test_case = LLMTestCase(
        input=query,
        actual_output=custom_answer,
    )

    print("Actual Output:", custom_answer)
    answer_relevancy_metric.measure(test_case)
    print("Answer Relevancy Score:", answer_relevancy_metric.score)


if __name__ == "__main__":

    # Check if Ollama is running and start if needed
    setup_ollama()
    # Set local LLM as evaluation judge model
    evaluationModel = setup_custom_ollama_model_for_evaluation()

    # Test 1: "What is the capital of France?"
    # ✅ PASSES with Score: 1.0
    # Reason: The answer "The capital of France is Paris." is directly relevant and 
    # accurately answers the question. The LLM provides a concise, factually correct response 
    # that directly addresses the query without unnecessary information.
    print("Test 1 - Expected: ✅ PASS")
    test_answer_relevancy(evaluationModel, "What is the capital of France?")
    
    # Test 2: "Who won the FIFA World Cup in 2099?"
    # ✅ PASSES with Score: 1.0
    # Reason: Although the question asks about a future event (2099), the LLM's answer is 
    # highly relevant because it:
    #   1. Acknowledges the question directly
    #   2. Explains why it cannot answer (date hasn't occurred)
    #   3. Provides relevant alternative information (past/future tournaments)
    #   4. Offers to help with related queries
    # The answer is contextually appropriate and demonstrates understanding of the query's 
    # intent, making it fully relevant despite not being answerable.
    print("\nTest 2 - Expected: ✅ PASS")
    test_answer_relevancy(evaluationModel, "Who won the FIFA World Cup in 2099?")
    
    # Test 3: "What is the capital of France?"
    # ❌ FAILS with Score: 0.0 (or very low)
    # Reason: The answer is completely irrelevant to the query. When asked about the capital 
    # of France, the response is about pizza recipes instead. This demonstrates when 
    # AnswerRelevancyMetric fails - when the actual output has ZERO connection to the 
    # input question. The metric detects that the response doesn't address the user's query at all.
    print("\nTest 3 - Expected: ❌ FAIL")
    test_answer_relevancy_custom(
        evaluationModel,
        "What is the capital of France?",
        "Pizza is a delicious Italian dish made with dough, tomato sauce, and cheese. Popular toppings include pepperoni, mushrooms, and olives."
    )
