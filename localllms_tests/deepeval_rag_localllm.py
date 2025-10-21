from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, setup_custom_ollama_model_for_evaluation, generate_ollama_response
from utils.wikipedia_retriever import retrieve_context_from_wiki

def test_single_query_all_metrics(query, context, expected_output, evaluationModel, use_custom_response=False, custom_response=None):
    """Test query with all 3 RAG metrics: Precision, Recall, Relevancy."""
    
    if use_custom_response and custom_response:
        ollama_response = custom_response
    else:
        ollama_response = generate_ollama_response(query, model_name="llama3.1:8b")
    test_case = LLMTestCase(
        input=query,
        actual_output=ollama_response,
        expected_output = expected_output,
        retrieval_context=[context]
    )
    
    print(f"\nUser Query: {query}")
    print(f"Context derived from Wikipedia: {context[:150]}...")
    print(f"Output by Local LLM: {ollama_response[:150]}...")
   
    print("=" * 80)
    print("Testing Contextual Precision Metric")
    precision_metric = ContextualPrecisionMetric(model=evaluationModel)
    precision_metric.measure(test_case)
    precision_status = "✅" if precision_metric.score >= 0.5 else "❌"
    print(f"\n{precision_status} Precision: {precision_metric.score:.2f}")
     
    print("=" * 80)
    print("Testing Contextual Recall Metric")
    recall_metric = ContextualRecallMetric(model=evaluationModel)
    recall_metric.measure(test_case)
    recall_status = "✅" if recall_metric.score >= 0.5 else "❌"
    print(f"{recall_status} Recall: {recall_metric.score:.2f}")
    
    print("=" * 80)
    print("Testing Contextual Relevancy Metric")
    relevancy_metric = ContextualRelevancyMetric(model=evaluationModel)
    relevancy_metric.measure(test_case)
    relevancy_status = "✅" if relevancy_metric.score >= 0.5 else "❌"
    print(f"{relevancy_status} Relevancy: {relevancy_metric.score:.2f}")
    
    scores = [precision_metric.score, recall_metric.score, relevancy_metric.score]
    avg_score = sum(scores) / len(scores)
    passed = sum(1 for s in scores if s >= 0.5)
    
    print(f"\nAverage: {avg_score:.2f} | Passed: {passed}/3")
    print("=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("DEEPEVAL RAG LOCAL LLM - INDIVIDUAL EVALUATION WITH 3 METRICS")
    print("=" * 80)
    
    setup_ollama()
    evaluation_model = setup_custom_ollama_model_for_evaluation(model="llama3.2:3b")

    print("\n🧪 TEST 1: ARTIFICIAL INTELLIGENCE (Expected: PASS)")
    print("=" * 80)
    
    test_single_query_all_metrics(
        query="What are the main applications of artificial intelligence?",
        context=retrieve_context_from_wiki(topic="Artificial Intelligence"),
        expected_output="Artificial intelligence has applications in various fields including healthcare, finance, transportation, and customer service.",
        evaluationModel=evaluation_model
    )

    print("\n🧪 TEST 2: OFF-TOPIC RESPONSE (Expected: FAIL)")
    print("=" * 80)
    
    test_single_query_all_metrics(
        query="What are the main applications of artificial intelligence?",
        context=retrieve_context_from_wiki(topic="Artificial Intelligence"),
        expected_output="Pizza is a delicious Italian dish made with dough, tomato sauce, and cheese. Popular toppings include pepperoni, mushrooms, olives, and basil.",
        evaluationModel=evaluation_model,
        use_custom_response=True,
        custom_response="I really enjoy playing soccer on weekends with my friends. Soccer is a fun sport that keeps you active and healthy. Many people around the world love soccer."
    )
