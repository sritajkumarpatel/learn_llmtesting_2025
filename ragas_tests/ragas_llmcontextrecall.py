"""
RAGAS LLM Context Recall Metric
================================

What is LLMContextRecall?
- Measures what proportion of retrieved context is used/recalled in the response
- Uses an LLM judge to evaluate how well the response utilizes given context
- Answers: "What percentage of available context information appears in the response?"

How It Works:
- Takes: query, response, and retrieved_contexts
- LLM evaluates: "What portion of the retrieved context is recalled in the response?"
- Outputs: Score from 0.0 to 1.0 (proportion of context recalled)

Score Interpretation (RAGAS Standard):
- Score Range: 0.0 to 1.0 (PROPORTION of context recalled)
- 0.0         = No context recalled - Response ignores provided context
- 0.0-0.3     = Poor recall (❌ FAIL) - ≤30% of context used
- 0.3-0.5     = Fair recall (⚠️ PARTIAL) - 30-50% of context used
- 0.5-0.7     = Good recall (✅ PASS) - 50-70% of context used
- 0.7-1.0     = Excellent recall (✅ PASS) - ≥70% of context used, nearly complete

Threshold: 0.7 (70%)
- Minimum acceptable: Response should recall ≥70% of context (0.7 = excellent)
- Higher scores are better: 1.0 = all context recalled, 0.0 = none recalled
- Interpretation: Score represents PROPORTION of context information that appears in response

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
from utils import config, setup_ollama, generate_ollama_response
from utils.config import get_ollama_model, get_ollama_url, ModelType
from utils.wikipedia_retriever import retrieve_context_from_wiki


def test_single_query_llm_context_recall(user_query, expected_output, context):
    """Test query with LLMContextRecall metric."""

    # Generate response from LLM under test
    actual_output = generate_ollama_response(user_query, model_name=get_ollama_model(ModelType.ACTUAL_OUTPUT))
    
    test_data = {
        "user_input": user_query,
        "response": actual_output,
        "reference": expected_output,
        "retrieved_contexts": [context]
    }

    print(f"\n📝 Query: {user_query}")
    print(f"💬 Response: {actual_output[:150]}...")
    print(f"💬 Reference: {expected_output[:150]}...")
    print(f"📚 Context: {context[:150]}...")

    ollama_chat = ChatOllama(model=get_ollama_model(ModelType.EVALUATION), base_url=get_ollama_url())
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

    # Example test case
    user_query = "Who created the Mona Lisa?"
    expected_output = generate_ollama_response(user_query, model_name=get_ollama_model(ModelType.EXPECTED_OUTPUT))

    context = retrieve_context_from_wiki(u"Mona Lisa")

    test_single_query_llm_context_recall(user_query, expected_output, context)