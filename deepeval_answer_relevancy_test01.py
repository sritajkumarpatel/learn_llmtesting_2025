from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import ollama
from local_llm_ollama_setup import setup_ollama

def test_answer_relevancy(query):
    setup_ollama()
    ollama_response = ollama.chat(model='llama3.2:3b', messages=[
        {
            'role': 'user',
            'content': query
        }
    ])
    answer_relevancy_metric = AnswerRelevancyMetric()
    test_case = LLMTestCase(
        input= query,
        actual_output=ollama_response['message']['content'],
    
    )

    print("Actual Output:", ollama_response['message']['content'])
    answer_relevancy_metric.measure(test_case)
    print("Answer Relevancy Score:", answer_relevancy_metric.score)


def test_answer_relevancy_custom(query, custom_answer):
    """Test with a custom answer (useful for testing failure cases)"""
    answer_relevancy_metric = AnswerRelevancyMetric()
    test_case = LLMTestCase(
        input=query,
        actual_output=custom_answer,
    )

    print("Actual Output:", custom_answer)
    answer_relevancy_metric.measure(test_case)
    print("Answer Relevancy Score:", answer_relevancy_metric.score)


if __name__ == "__main__":
    # Test 1: "What is the capital of France?"
    # ✅ PASSES with Score: 1.0
    # Reason: The answer "The capital of France is Paris." is directly relevant and 
    # accurately answers the question. The LLM provides a concise, factually correct response 
    # that directly addresses the query without unnecessary information.
    print("=" * 70)
    print("Test 1: Direct factual question with concise answer")
    print("Expected: ✅ PASS - Score 1.0 (highly relevant answer)")
    print("=" * 70)
    test_answer_relevancy("What is the capital of France?")
    
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
    print("\n" + "=" * 70)
    print("Test 2: Future event question with contextually relevant explanation")
    print("Expected: ✅ PASS - Score 1.0 (highly relevant explanation)")
    print("=" * 70)
    test_answer_relevancy("Who won the FIFA World Cup in 2099?")
    
    # Test 3: "What is the capital of France?"
    # ❌ FAILS with Score: 0.0 (or very low)
    # Reason: The answer is completely irrelevant to the query. When asked about the capital 
    # of France, the response is about pizza recipes instead. This demonstrates when 
    # AnswerRelevancyMetric fails - when the actual output has ZERO connection to the 
    # input question. The metric detects that the response doesn't address the user's query at all.
    print("\n" + "=" * 70)
    print("Test 3: Irrelevant answer (off-topic response)")
    print("Expected: ❌ FAIL - Score 0.0 (completely irrelevant answer)")
    print("=" * 70)
    test_answer_relevancy_custom(
        "What is the capital of France?",
        "Pizza is a delicious Italian dish made with dough, tomato sauce, and cheese. Popular toppings include pepperoni, mushrooms, and olives."
    )
