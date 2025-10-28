# RUNS LOCALLY - Uses Ollama for local LLM inference, no API keys required

"""
DeepEval RAG Contextual Metrics (Local Ollama) - Precision, Recall, Relevancy
==============================================================================

What are RAG Contextual Metrics?
- Evaluate how effectively RETRIEVAL-AUGMENTED GENERATION (RAG) systems work
- Test what PROPORTION of retrieved context is used effectively
- Uses LOCAL Ollama model (config-defined evaluation model) as judge
- Answers: "What percentage of retrieved context is used effectively in response?"

The 3 RAG Metrics:

1. CONTEXTUAL PRECISION
   - Measures: PROPORTION of retrieved context ACTUALLY USED by response
   - Ideal: All retrieved documents are relevant and used
   - Formula: (# relevant docs used) / (total docs retrieved)
   - Score 0.0-1.0
   
2. CONTEXTUAL RECALL
   - Measures: PROPORTION of AVAILABLE INFO in context captured in response
   - Ideal: Response includes all important information
   - Formula: (# facts from context in response) / (total facts in context)
   - Score 0.0-1.0
   
3. CONTEXTUAL RELEVANCY
   - Measures: What PROPORTION of retrieved docs are RELEVANT to the query?
   - Ideal: All retrieved docs help answer the question
   - Formula: (# relevant docs) / (total docs retrieved)
   - Score 0.0-1.0

Score Interpretation (Each Metric):
- Score Range: 0.0 to 1.0 (PROPORTION metric - higher is better)
- 0.0         = None/no relevance - No context used/relevant
- 0.0-0.3     = Poor (❌ FAIL) - ≤30% precision/recall/relevancy
- 0.3-0.5     = Moderate (⚠️ PARTIAL) - 30-50% effectiveness
- 0.5-0.7     = Good (✅ PASS) - 50-70% effectiveness
- 0.7-1.0     = Excellent (✅ PASS) - ≥70% effectiveness, nearly complete

Threshold: 0.7 (70% - LLM-based metric standard)
- Score must be >= 0.7 to PASS (per test, per metric)
- Higher scores are better: 1.0 = perfect, 0.0 = useless
- Interpretation: Each metric represents PROPORTION of something (precision/recall/relevancy)

Typical RAG Evaluation:
- Query input to retrieve documents from knowledge base
- Retrieve top K relevant documents (vector similarity)
- Build prompt with context: "Context: [documents] \n Question: [query]"
- Generate LLM response using context
- Evaluate 3 metrics per test case
- All 3 metrics should pass (>= 0.7)

Use Cases:
- Document QA system validation
- FAQ chatbot evaluation
- Search result effectiveness
- Knowledge base coverage testing
- Customer support bot assessment
- Research document retrieval evaluation

Local vs OpenAI:
- Uses local Ollama for evaluation (free, offline)
- No API costs, runs in your environment
- Good for rapid iteration and testing
- May have lower accuracy than GPT-4

Data Source: Wikipedia documents via vector similarity search
Vector DB: ChromaDB with semantic embeddings

Requires: Ollama running with:
- Config-defined generation model for response generation
- Config-defined evaluation model for evaluation

Reference: DeepEval RAG Metrics
https://docs.depevalai.com/docs/metrics/contextual-precision/
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, setup_custom_ollama_model_for_evaluation, generate_ollama_response
from utils.create_vector_db import create_vector_db_from_wikipedia
from utils.wikipedia_retriever import retrieve_context_from_wiki


def build_prompt_with_context(user_query, retrieved_docs):
    """Build a prompt with retrieved Wikipedia documents as context."""
    prompt = f"Answer the following user_query based on the Wikipedia data provided:\n\n"
    
    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"[Document {i}]\n{doc.page_content}\n\n"
    
    prompt += f"Question: {user_query}\n\nProvide a clear answer based on the documents above:\n"
    return prompt


def test_single_query_all_metrics(user_query, retrieved_docs, expected_output, evaluationModel, use_custom_response=False, custom_response=None):
    """Test query with all 3 RAG metrics: Precision, Recall, Relevancy."""
    
    # Build context from retrieved documents
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Build prompt with context
    prompt_with_context = build_prompt_with_context(user_query, retrieved_docs)
    
    # Generate or use custom response
    if use_custom_response and custom_response:
        ollama_response = custom_response
    else:
        ollama_response = generate_ollama_response(prompt_with_context)
    
    # Create test case
    test_case = LLMTestCase(
        input=user_query,
        actual_output=ollama_response,
        expected_output=expected_output,
        retrieval_context=[context]
    )
    
    print(f"\nQuestion: {user_query}")
    print(f"Context (from {len(retrieved_docs)} docs): {context[:150]}...")
    print(f"Output: {ollama_response[:150]}...")
    print("=" * 80)
    
    # Test 1: Precision
    print("Testing Contextual Precision Metric")
    precision_metric = ContextualPrecisionMetric(model=evaluationModel)
    precision_metric.measure(test_case)
    precision_status = "✅" if precision_metric.score >= 0.5 else "❌"
    print(f"{precision_status} Precision: {precision_metric.score:.2f}")
    
    # Test 2: Recall
    print("\nTesting Contextual Recall Metric")
    recall_metric = ContextualRecallMetric(model=evaluationModel)
    recall_metric.measure(test_case)
    recall_status = "✅" if recall_metric.score >= 0.5 else "❌"
    print(f"{recall_status} Recall: {recall_metric.score:.2f}")
    
    # Test 3: Relevancy
    print("\nTesting Contextual Relevancy Metric")
    relevancy_metric = ContextualRelevancyMetric(model=evaluationModel)
    relevancy_metric.measure(test_case)
    relevancy_status = "✅" if relevancy_metric.score >= 0.5 else "❌"
    print(f"{relevancy_status} Relevancy: {relevancy_metric.score:.2f}")
    
    # Summary
    scores = [precision_metric.score, recall_metric.score, relevancy_metric.score]
    avg_score = sum(scores) / len(scores)
    passed = sum(1 for s in scores if s >= 0.5)
    
    print(f"\nAverage: {avg_score:.2f} | Passed: {passed}/3")
    print("=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("RAG LOCAL LLM - VECTOR DB + EVALUATION")
    print("=" * 80)
    
    setup_ollama()
    evaluation_model = setup_custom_ollama_model_for_evaluation()

    # Step 1: Create vector database from Wikipedia
    print("\n📚 STEP 1: Creating Vector Database from Wikipedia")
    print("=" * 80)
    vectordb = create_vector_db_from_wikipedia(
        query="KANTARA 2",
        num_docs=5,
        force_rebuild=False
    )

    # Step 2: Test 1 - Relevant question about Kantara 2
    print("\n\n🧪 TEST 1: RELEVANT QUESTION ABOUT KANTARA 2 (Expected: PASS)")
    print("=" * 80)
    
    user_query1 = "What is Kantara 2 about?"
    retrieved_docs1 = vectordb.similarity_search(user_query1, k=3)
    
    test_single_query_all_metrics(
        user_query=user_query1,
        retrieved_docs=retrieved_docs1,
        expected_output="Kantara 2 is an Indian film that continues the story with action and drama.",
        evaluationModel=evaluation_model
    )

    # Step 3: Test 2 - Off-topic response (failure case)
    print("\n\n🧪 TEST 2: OFF-TOPIC RESPONSE ABOUT SOCCER (Expected: FAIL)")
    print("=" * 80)
    
    user_query2 = "When was Kantara 2 released?"
    retrieved_docs2 = vectordb.similarity_search(user_query2, k=3)
    
    test_single_query_all_metrics(
        user_query=user_query2,
        retrieved_docs=retrieved_docs2,
        expected_output="Kantara 2 was released in 2024.",
        evaluationModel=evaluation_model,
        use_custom_response=True,
        custom_response="I really enjoy playing soccer on weekends with my friends. Soccer is a fun sport that keeps you active and healthy. Many people around the world love soccer."
    )

    print("\n\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)

