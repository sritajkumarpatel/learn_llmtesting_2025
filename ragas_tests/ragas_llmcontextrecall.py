"""
RAGAS LLM Context Recall Metric
================================

What is LLMContextRecall?
- Measures if the LLM response contains information relevant to the retrieved context
- Uses an LLM judge to evaluate how well the response recalls/uses the given context
- Answers: "Does the response cover information from the provided context?"

How It Works:
- Takes: query, response, and retrieved_contexts
- LLM evaluates: "What portion of the retrieved context is recalled in the response?"
- Outputs: Score from 0.0 to 1.0

Score Interpretation (RAGAS Standard):
- 0.0-0.3   = Poor recall (❌ FAIL) - Response ignores context
- 0.3-0.5   = Fair recall (⚠️ PARTIAL) - Response uses only some context
- 0.5-0.7   = Good recall (✅ PASS) - Response covers most context
- 0.7-1.0   = Excellent recall (✅ PASS) - Response fully recalls context

Threshold: 0.7 (70%)
- Minimum acceptable: Response should recall 70% of relevant context
- Follows RAGAS LLM-based metric standards
- Stricter than non-LLM metrics (0.5 threshold)

Use Cases:
- RAG system evaluation
- Fact-checking accuracy
- Context utilization in responses
- Hallucination detection

Reference: RAGAS Documentation
https://docs.ragas.io/en/latest/concepts/metrics/
"""

import sys
from pathlib import Path
from ragas import SingleTurnSample
from ragas.metrics import LLMContextRecall
from ragas.llms.base import LangchainLLMWrapper
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, setup_custom_ollama_model_for_evaluation, generate_ollama_response
from utils.wikipedia_retriever import retrieve_context_from_wiki


def test_single_query_llm_context_recall(user_query, expected_output, context, llm, use_custom_response=False, custom_response=None):
    """Test query with LLMContextRecall metric."""

    # Generate response from LLM under test
    ollama_response = generate_ollama_response(user_query, model_name="llama3.1:8b")
    
    test_data = {
        "user_input": user_query,
        "response": ollama_response,
        "reference": expected_output,
        "retrieved_contexts": [context]
    }

    print(f"\n📝 Query: {user_query}")
    print(f"💬 Response: {ollama_response[:150]}...")
    print(f"💬 Reference: {expected_output[:150]}...")
    print(f"📚 Context: {context[:150]}...")

    ollama_chat = ChatOllama(model="deepseek-r1:8b", base_url="http://localhost:11434")
    evaluator_model = LangchainLLMWrapper(ollama_chat)
    context_recall = LLMContextRecall(llm=evaluator_model)

    test_data = SingleTurnSample(**test_data)
    finalscore = context_recall.single_turn_score(test_data)


    # Determine pass/fail based on LLMContextRecall score threshold
    # LLM-based metrics typically use 0.7 as standard threshold (70% quality)
    # Range: 0.0 (no recall) to 1.0 (perfect recall)
    threshold = 0.7
    if finalscore >= threshold:
        status = "✅ PASS"
    else:
        status = "❌ FAIL"
    
    print(f"LLM Context Recall Score: {finalscore:.4f} | Threshold: {threshold} | {status}")

if __name__ == "__main__":
    print("RAGAS LLM Context Recall Score Evaluation")
    print("=" * 50)
    print("Threshold: 0.7 (70% - LLM-based metric standard)\n")
    
    setup_ollama()

    llm = setup_custom_ollama_model_for_evaluation(
        model="deepseek-r1:8b",
        temperature=1
    )

    # Example test case
    user_query = "Who created the Mona Lisa?"
    expected_output = generate_ollama_response(user_query, model_name="deepseek-r1:8b")

    context = retrieve_context_from_wiki(u"Mona Lisa")

    test_single_query_llm_context_recall(user_query, expected_output, context, llm)