# RUNS LOCALLY - Uses Ollama for generation, OpenAI for evaluation (requires API key)

"""
RAGAS EvaluationDataset and evaluate() Method
==============================================

What is EvaluationDataset and evaluate()?
- EvaluationDataset: Container for batch evaluation of multiple test samples
- evaluate(): Batch evaluation method that runs multiple metrics on all samples asynchronously
- Answers: "How do multiple RAG responses perform across various quality metrics?"

How It Works:
- Takes: EvaluationDataset (containing SingleTurnSample/MultiTurnSample objects) and list of metrics
- Creates samples with user_input, response, reference, retrieved_contexts for each test case
- Runs all metrics on all samples in parallel using async processing
- Outputs: Result object with scores for each metric across all samples

Score Interpretation (RAGAS Standard):
- Score Range: 0.0 to 1.0 (PROPORTION/BINARY depending on metric)
- Multiple metrics evaluated: Context Recall, Noise Sensitivity, Response Relevancy, Faithfulness, AspectCritic
- Individual metric scores: 0.0 (poor) to 1.0 (excellent) - see metric-specific thresholds
- Batch results: Aggregated scores across all test samples for comprehensive evaluation

Threshold: Varies by metric (see individual metric documentation)
- Context Recall: ≥0.7 (70% context utilization)
- Noise Sensitivity: ≤0.5 (≤50% affected by noise)
- Response Relevancy: ≥0.7 (70% response relevance)
- Faithfulness: ≥0.7 (70% factual consistency)
- AspectCritic: 1.0 (100% - binary pass/fail)

Use Cases:
- Batch evaluation of RAG systems
- Comparative analysis of different RAG implementations
- Comprehensive quality assessment across multiple dimensions
- Automated testing pipelines for RAG applications

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
    """
    Create RAGAS evaluator metrics using OpenAI for evaluation.

    Initializes multiple RAGAS metrics including standard metrics (Context Recall,
    Noise Sensitivity, Response Relevancy, Faithfulness) and custom AspectCritic
    metrics for cultural sensitivity and historical accuracy.

    Returns:
    dict: Dictionary containing initialized RAGAS metric objects with keys:
        - 'context_recall': LLMContextRecall metric
        - 'noise_sensitivity': NoiseSensitivity metric  
        - 'response_relevancy': ResponseRelevancy metric
        - 'faithfulness': Faithfulness metric
        - 'cultural_sensitivity': AspectCritic for cultural sensitivity
        - 'historical_accuracy': AspectCritic for historical accuracy
    """
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
    """
    Create comprehensive test cases for Jagannatha Temple evaluation.

    Defines structured test cases with queries, expected outputs, and topics
    covering various aspects of Jagannatha Temple including history, architecture,
    festivals, location, and visiting information.

    Returns:
    list: List of dictionaries, each containing:
        - 'query' (str): The test question
        - 'expected_output' (str): Expected response content
        - 'topic' (str): Topic category for the test case
    """

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
    """
    Demonstrate RAGAS EvaluationDataset and evaluate() method for batch evaluation.

    This function showcases the complete RAG evaluation pipeline:
    1. Sets up vector database with Wikipedia data
    2. Creates test samples with context-augmented responses
    3. Builds EvaluationDataset from SingleTurnSample objects
    4. Runs batch evaluation using evaluate() method
    5. Displays results for all metrics across all test cases

    The function demonstrates proper RAG context augmentation by retrieving
    relevant context and including it in the prompt before generation.

    Returns:
    None: Prints evaluation results and debugging information
    """

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