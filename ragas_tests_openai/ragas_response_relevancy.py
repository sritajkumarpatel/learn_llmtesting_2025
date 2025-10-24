"""
RAGAS Response Relevancy Metric
================================

What is ResponseRelevancy?
- Measures what proportion of the LLM response is relevant to the user query
- Uses embeddings and LLM evaluation to assess semantic relevance
- Answers: "How relevant is the response to the question asked?"

How It Works:
- Takes: user query and response text
- Generates question embeddings and compares with response sentence embeddings
- Uses LLM to evaluate relevance of response components
- Outputs: Score from 0.0 to 1.0 (proportion of relevant content)

Score Interpretation (RAGAS Standard):
- Score Range: 0.0 to 1.0 (PROPORTION of response relevant to query)
- 0.0-0.3 = Irrelevant (❌ FAIL) - ≤30% of response addresses query
- 0.3-0.5 = Partially relevant (⚠️ PARTIAL) - 30-50% relevant
- 0.5-0.7 = Moderately relevant (⚠️ PARTIAL) - 50-70% relevant
- 0.7-1.0 = Highly relevant (✅ PASS) - ≥70% directly addresses query

Threshold: 0.7 (70%)
- Minimum acceptable: Response should be ≥70% relevant to query
- Higher scores indicate better relevance to the user's question

Use Cases:
- Q&A system evaluation
- Chatbot response quality assessment
- Search result relevance checking
- Content filtering and moderation

Reference: RAGAS Documentation
https://docs.ragas.io/en/latest/concepts/metrics/
"""

import os
import sys
from pathlib import Path
from ragas import SingleTurnSample
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import ResponseRelevancy
from huggingface_hub import login

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import config, setup_ollama, generate_ollama_response
from utils.config import get_ollama_model, get_ollama_url, ModelType

def test_responserelevancy(user_query: str) -> None:
    """
    Test Response Relevancy metric from RAGAS framework.

    Measures what proportion of the response is relevant to the user query.
    Returns a score from 0.0 to 1.0, where higher scores indicate better relevance.

    Parameters:
    user_query (str): The query to evaluate response relevance against

    Returns:
    None: Prints evaluation results including score and pass/fail status
    """

    actual_output = generate_ollama_response(
        "What is the capital of France?",
        model_name=get_ollama_model(ModelType.ACTUAL_OUTPUT),
    )

    test_data = {
        "user_input": user_query,
        "response": actual_output
    }

    print(f"\n📝 Query: {user_query}")
    print(f"💬 Response (model-under-test): {actual_output}")

    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    login(token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
    embeddings_model = HuggingFaceEmbeddings(model_name ="google/embeddinggemma-300m")

    response_relevancy = ResponseRelevancy(
        embeddings=embeddings_model,
        llm=evaluator_llm,
    )
    
    test_data = SingleTurnSample(**test_data)
    finalscore = response_relevancy.single_turn_score(test_data)

    # Determine pass/fail based on ResponseRelevancy score threshold
    # ResponseRelevancy returns scores from 0.0 to 1.0 (continuous)
    # Higher scores indicate better relevance to the query
    # Score of 0.7+ indicates highly relevant response
    threshold = 0.7
    if finalscore >= threshold:
        status = "✅ PASS"
    else:
        status = "❌ FAIL"

    print(f"Response Relevancy Score: {finalscore:.4f} | Threshold: {threshold} | {status}")

if __name__ == "__main__":
    print("RAGAS ResponseRelevancy Score Evaluation")
    print("=" * 50)

    setup_ollama()

    # Example test case
    user_query = "What is the capital of France?"
    test_responserelevancy(user_query)

    print("=" * 50)

    user_query2 = "What is the formula 1 event held in Monaco called?"
    test_responserelevancy(user_query2)