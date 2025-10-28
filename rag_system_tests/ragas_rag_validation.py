# RUNS LOCALLY - Uses Ollama for generation, OpenAI for evaluation (requires API key)

"""
RAGAS EvaluationDataset and evaluate() Method Demonstration - Jagannatha Temple, Odisha
=====================================================================================

Demonstrates how to use RAGAS evaluate() method with EvaluationDataset for batch evaluation.

What this script does:
- Creates an EvaluationDataset from SingleTurnSample objects
- Uses the evaluate() method to run multiple metrics in batch
- Demonstrates proper RAG context augmentation for response generation
- Includes debugging output for each test case

Key Components:

1. EVALUATIONDATASET CREATION
   - Creates SingleTurnSample objects with user_input, response, reference, retrieved_contexts
   - Builds EvaluationDataset from list of samples
   - Supports both single-turn and multi-turn evaluation patterns

2. BATCH EVALUATION WITH evaluate()
   - evaluate(dataset, metrics) runs all metrics on all samples asynchronously
   - Returns Result object with scores for each metric across all samples
   - More efficient than individual single_turn_score() calls

3. RAG CONTEXT AUGMENTATION
   - Retrieves relevant context from vector database
   - Augments prompts with retrieved context before generation
   - Ensures responses use retrieved information (critical for context recall)

4. DEBUGGING OUTPUT
   - Prints query, generated response, expected response, and retrieved context
   - Helps identify why metrics like context_recall may be low

Current Setup:
- Generation: Local Ollama models (no API keys required)
- Evaluation: OpenAI GPT-4o-mini (requires API key)
- Vector DB: ChromaDB with Wikipedia data about Jagannatha Temple

The RAGAS Metrics Evaluated:

1. LLM CONTEXT RECALL
   - Measures: PROPORTION of retrieved context that is recalled/used in response
   - Ideal: Response uses all relevant context information
   - Formula: (# context facts in response) / (total facts in context)
   - Score 0.0-1.0 (higher is better)

2. NOISE SENSITIVITY
   - Measures: How robust response is to irrelevant/noisy context injection
   - Ideal: Response unaffected by irrelevant information
   - Formula: Proportion of claims that become incorrect with noise
   - Score 0.0-1.0 (lower is better)

3. RESPONSE RELEVANCY
   - Measures: PROPORTION of response that directly addresses the query
   - Ideal: All response content is relevant to the question
   - Formula: (# relevant response sentences) / (total response sentences)
   - Score 0.0-1.0 (higher is better)

4. FAITHFULNESS
   - Measures: Factual consistency between response and retrieved context
   - Ideal: Response contains no hallucinations or contradictions
   - Formula: (# claims supported by context) / (total claims in response)
   - Score 0.0-1.0 (higher is better)

5. CULTURAL SENSITIVITY (AspectCritic)
   - Custom metric evaluating respectful treatment of Hindu traditions
   - Checks for appropriate tone and cultural understanding
   - Score 0.0-1.0 (higher is better)

6. HISTORICAL ACCURACY (AspectCritic)
   - Custom metric verifying factual historical information
   - Validates dates, figures, and historical significance
   - Score 0.0-1.0 (higher is better)

Score Interpretation (Each Metric):
- Score Range: 0.0 to 1.0 (PROPORTION metric)
- For Recall/Relevancy/Faithfulness/AspectCritic: Higher is better (1.0 = perfect, 0.0 = poor)
- For Noise Sensitivity: Lower is better (0.0 = robust, 1.0 = sensitive)
- 0.0-0.3     = Poor performance
- 0.3-0.5     = Moderate performance
- 0.5-0.7     = Good performance
- 0.7-1.0     = Excellent performance

Thresholds (RAGAS Standards):
- Context Recall: ≥ 0.7 (70% of context should be recalled)
- Noise Sensitivity: ≤ 0.5 (≤50% claims affected by noise)
- Response Relevancy: ≥ 0.7 (70% of response should be relevant)
- Faithfulness: ≥ 0.7 (70% of claims should be supported by context)
- Cultural Sensitivity: ≥ 0.8 (80% appropriate cultural handling)
- Historical Accuracy: ≥ 0.8 (80% factual accuracy)

Golden Test Cases for Jagannatha Temple:
- Uses structured test cases with predefined queries and expected outputs
- Tests factual accuracy about temple history, architecture, festivals
- Tests cultural significance, rituals, and location information
- Validates RAG system performance against predefined standards

Use Cases:
- Demonstrating RAGAS EvaluationDataset usage
- Batch evaluation of multiple test cases
- Debugging RAG system performance issues
- Comparing different RAG implementations

Data Source: Wikipedia documents about Jagannatha Temple, Odisha
Vector DB: ChromaDB with semantic embeddings

Requires:
- Ollama running locally for response generation
- OpenAI API key for evaluation metrics
- ChromaDB vector database with Jagannatha Temple data

Reference: RAGAS Documentation
https://docs.ragas.io/en/latest/concepts/metrics/
"""

import sys
from pathlib import Path
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, NoiseSensitivity, ResponseRelevancy, Faithfulness, AspectCritic
from ragas.llms.base import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, generate_ollama_response
from utils.config import get_ollama_model, get_ollama_url, ModelType
from utils.create_vector_db import add_documents_to_vector_db, create_vector_db_from_wikipedia, get_vector_db
from utils.wikipedia_retriever import retrieve_context_from_wiki


def create_ragas_evaluators():
    """Create RAGAS evaluators using config-defined models."""
    openai_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    evaluator_model = LangchainLLMWrapper(openai_chat)

    evaluators = {
        'context_recall': LLMContextRecall(llm=evaluator_model),
        'noise_sensitivity': NoiseSensitivity(mode="irrelevant", llm=evaluator_model),
        'response_relevancy': ResponseRelevancy(llm=evaluator_model),
        'faithfulness': Faithfulness(llm=evaluator_model),
        'cultural_sensitivity': AspectCritic(
            name="cultural_sensitivity",
            llm=evaluator_model,
            definition="Does the response show appropriate respect and understanding of Hindu religious traditions, cultural context, and sacred significance of Jagannatha Temple? Check for respectful language, accurate representation of religious practices, and appropriate tone for discussing sacred sites."
        ),
        'historical_accuracy': AspectCritic(
            name="historical_accuracy",
            llm=evaluator_model,
            definition="Does the response provide factually correct historical information about Jagannatha Temple, including dates, rulers, architectural history, and cultural significance? Verify historical dates, figures, and cultural importance claims."
        )
    }

    return evaluators

def create_golden_test_cases():
    """Create comprehensive test cases for Jagannatha Temple evaluation."""

    test_cases = [
        {
            'query': 'What is the historical significance of Jagannatha Temple?',
            'expected_output': 'Jagannatha Temple is one of the most sacred Hindu pilgrimage sites, known as the "White Pagoda" and part of the Char Dham pilgrimage circuit.',
            'topic': 'Jagannatha Temple'
        },
        {
            'query': 'Describe the architecture and design of Jagannatha Temple.',
            'expected_output': 'The temple features a unique pyramidal structure with a curved tower (shikhara), distinctive horse-shaped chariot festival designs, and traditional Kalinga architecture.',
            'topic': 'Jagannatha Temple'
        },
        {
            'query': 'What are the major festivals celebrated at Jagannatha Temple?',
            'expected_output': 'Major festivals include Rath Yatra (Chariot Festival), Snana Yatra (Bathing Festival), and various Hindu festivals throughout the year.',
            'topic': 'Jagannatha Temple'
        },
        {
            'query': 'Where is Jagannatha Temple located and how can visitors reach it?',
            'expected_output': 'Jagannatha Temple is located in Puri, Odisha, India. Visitors can reach it by train to Puri station, flight to Bhubaneswar airport (60km away), or road transport.',
            'topic': 'Jagannatha Temple'
        },
        {
            'query': 'What are the visiting hours and entry requirements for Jagannatha Temple?',
            'expected_output': 'The temple is open from early morning to evening. Non-Hindus can visit the outer complex but only Hindus may enter the inner sanctum. Photography may be restricted in certain areas.',
            'topic': 'Jagannatha Temple'
        }
    ]

    return test_cases


def run_evaluation_with_dataset():
    """Demonstrate how the evaluate() method works with EvaluationDataset."""

     # Setup
    setup_ollama()
    user_query = "Jagannatha Temple"

    print("=" * 80)
    print(f"RAGAS COMPREHENSIVE RAG EVALUATION USING EVALUATION DATASET, AND EVALUATE- {user_query}")
    print("=" * 80)

   
    # Update vector database creation step
    print("\n📚 STEP 1: Updating Vector Database from Wikipedia")
    print("=" * 50)
    vectordb = add_documents_to_vector_db(
        query=user_query,
        num_docs=5
    )

    # vectordb = get_vector_db()
    # Get test cases
    test_cases = create_golden_test_cases()

    # Create samples for the dataset
    samples = []

    print("=" * 80)
    print(f"CREATING SAMPLES FOR EVALUATION DATASET")
    for test_case in test_cases:
        # Retrieve context
        retrieved_docs = vectordb.similarity_search(test_case['query'], k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Build prompt with context (add this function if needed)
        prompt_with_context = f"Context:\n{context}\n\nQuestion: {test_case['query']}\nAnswer:"

        # Generate response using context
        actual_output = generate_ollama_response(prompt_with_context)

        # Create SingleTurnSample
        sample = SingleTurnSample(
            user_input=test_case['query'],
            response=actual_output,
            reference=test_case['expected_output'],
            retrieved_contexts=[context]
        )
        samples.append(sample)

    print("=" * 80)

    # Create EvaluationDataset
    dataset = EvaluationDataset(samples=samples)

    evaluators = create_ragas_evaluators()

    # Define metrics list (including all evaluators)
    metrics = [
        evaluators['context_recall'],
        evaluators['noise_sensitivity'],
        evaluators['response_relevancy'],
        evaluators['faithfulness'],
        evaluators['cultural_sensitivity'],
        evaluators['historical_accuracy']
    ]

    print(f"📊 Dataset created with {len(samples)} samples")
    print(f"📏 Metrics to evaluate: {[m.name for m in metrics]}")

    # Run evaluation using evaluate() method
    print("\n⚡ Running batch evaluation with evaluate()...")
    result = evaluate(dataset=dataset, metrics=metrics)

    # Print results
    print("\n📈 EVALUATION RESULTS:")
    print(result)
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation_with_dataset()