# RUNS HYBRID - Uses Ollama for generation + OpenAI GPT-4o-mini for evaluation

"""
DeepEval RAG Contextual Metrics - Jagannatha Temple, Odisha (Hybrid Setup)
===========================================================================

Comprehensive RAG evaluation framework using DeepEval's Golden test cases with JSON output for HTML reporting.

What are RAG Contextual Metrics?
- Evaluate how effectively RETRIEVAL-AUGMENTED GENERATION (RAG) systems work
- Test what PROPORTION of retrieved context is used effectively
- Uses OpenAI GPT-4o-mini as judge for evaluation metrics
- Uses LOCAL Ollama model (llama3.1:8b) for response generation
- Focus: Jagannatha Temple, Odisha - Famous Hindu temple and cultural site

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

Custom GEval Metrics for Cultural Evaluation:
- Cultural Sensitivity: Respectful treatment of Hindu traditions
- Historical Accuracy: Factual correctness of historical information
- Tourism Relevance: Practical visitor information provided
- Educational Value: Clarity and learning effectiveness
- Completeness: Comprehensive coverage of query aspects

Golden Test Cases for Jagannatha Temple:
- Uses DeepEval's Golden framework for structured test cases
- Each Golden contains: input query, expected output, and context hints
- Tests factual accuracy about temple history, architecture, festivals
- Tests cultural significance, rituals, and location information
- Validates RAG system performance against predefined standards

Output: JSON file with detailed results for HTML report generation
- Automatically saved as: deepeval_rag_evaluation_with_YYYYMMDD_HHMMSS.json
- Compatible with generate_html_report.py for visual reports
- Includes all RAG and GEval metric scores per test case

Use Cases:
- Religious site information system validation
- Cultural heritage chatbot evaluation
- Tourism information accuracy testing
- Historical fact-checking systems

Data Source: Wikipedia documents about Jagannatha Temple, Odisha
Vector DB: ChromaDB with semantic embeddings

Requires: 
- Ollama running with llama3.1:8b (or similar) for response generation
- OpenAI API key for GPT-4o-mini evaluation metrics

Reference: DeepEval RAG Metrics
https://docs.depevalai.com/docs/metrics/contextual-precision/
"""

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, GEval
from deepeval.dataset import EvaluationDataset, Golden
import sys
from pathlib import Path
import json
import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_ollama, setup_custom_ollama_model_for_evaluation, generate_ollama_response
from utils.create_vector_db import add_documents_to_vector_db, create_vector_db_from_wikipedia
from utils.config import get_ollama_model, get_ollama_url, ModelType
from utils.wikipedia_retriever import retrieve_context_from_wiki


# GEval Metrics for Cultural and Domain-Specific Evaluation
cultural_sensitivity_metric = GEval(
    name="Cultural Sensitivity",
    criteria="""Assess if the response shows appropriate respect and understanding of Hindu religious traditions, cultural context, and sacred significance of Jagannatha Temple.
    Check for: respectful language, accurate representation of religious practices, avoidance of cultural appropriation, and appropriate tone for discussing sacred sites.""",
    evaluation_steps=[
        "Examine language used - is it respectful and appropriate for religious context?",
        "Check accuracy of cultural/religious information presented",
        "Assess whether response shows understanding of Hindu traditions",
        "Verify appropriate tone for discussing sacred pilgrimage site"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini"
)

historical_accuracy_metric = GEval(
    name="Historical Accuracy",
    criteria="""Evaluate the factual correctness of historical information about Jagannatha Temple, including dates, rulers, architectural history, and cultural significance.
    Verify: construction dates, historical figures mentioned, architectural evolution, and cultural importance.""",
    evaluation_steps=[
        "Verify historical dates and timelines mentioned",
        "Check accuracy of historical figures and rulers",
        "Validate architectural and construction history",
        "Confirm cultural and religious significance claims"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    model="gpt-4o-mini"
)

tourism_relevance_metric = GEval(
    name="Tourism Relevance",
    criteria="""Assess if the response provides practical, useful information for visitors to Jagannatha Temple, including accessibility, timings, cultural etiquette, and visitor experience.""",
    evaluation_steps=[
        "Check if practical visitor information is included",
        "Verify accessibility and entry information",
        "Assess cultural etiquette guidance for visitors",
        "Evaluate overall usefulness for tourists/pilgrims"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini"
)

educational_value_metric = GEval(
    name="Educational Value",
    criteria="""Evaluate how well the response explains Jagannatha Temple concepts for educational purposes, including clarity of explanations, appropriate detail level, and learning effectiveness.""",
    evaluation_steps=[
        "Assess clarity and comprehensibility of explanations",
        "Check appropriate level of detail for education",
        "Evaluate logical flow and structure of information",
        "Verify accuracy and reliability of educational content"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    model="gpt-4o-mini"
)

completeness_metric = GEval(
    name="Information Completeness",
    criteria="""Determine if the response provides comprehensive coverage of the query about Jagannatha Temple, addressing all key aspects without omitting important information.""",
    evaluation_steps=[
        "Identify all key aspects that should be covered for the query",
        "Check if response addresses all important elements",
        "Assess depth vs breadth balance",
        "Verify no critical information is missing"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    model="gpt-4o-mini"
)


def build_prompt_with_context(user_query, retrieved_docs):
    """Build a prompt with retrieved Wikipedia documents as context."""
    prompt = f"Answer the following question based on the Wikipedia data about Jagannatha Temple provided:\n\n"

    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"[Document {i}]\n{doc.page_content}\n\n"

    prompt += f"Question: {user_query}\n\nProvide a clear, accurate answer based on the documents above:\n"
    return prompt


def test_single_query_all_metrics(golden, retrieved_docs, evaluationModel, use_custom_response=False, custom_response=None, test_description=""):
    """Test query with RAG contextual metrics AND GEval custom metrics using DeepEval Golden."""

    # Build context from retrieved documents
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Build prompt with context
    prompt_with_context = build_prompt_with_context(golden.input, retrieved_docs)

    # Generate or use custom response
    if use_custom_response and custom_response:
        ollama_response = custom_response
    else:
        ollama_response = generate_ollama_response(prompt_with_context, model_name=get_ollama_model(ModelType.EXPECTED_OUTPUT))

    # Create test case using Golden
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=ollama_response,
        expected_output=golden.expected_output,
        retrieval_context=[context]
    )

    # Run DeeEval metrics
    precision_metric = ContextualPrecisionMetric(model="gpt-4o-mini")
    precision_metric.measure(test_case)
    print(f"Precision Metric Score for Query - {golden.input}: {precision_metric.score} and reason : {precision_metric.reason}")

    recall_metric = ContextualRecallMetric(model="gpt-4o-mini")
    recall_metric.measure(test_case)
    print(f"Recall Metric Score for Query - {golden.input}: {recall_metric.score} and reason : {recall_metric.reason}")

    relevancy_metric = ContextualRelevancyMetric(model="gpt-4o-mini")
    relevancy_metric.measure(test_case)
    print(f"Relevancy Metric Score for Query - {golden.input}: {relevancy_metric.score} and reason : {relevancy_metric.reason}")

    # GEval Metrics
    cultural_sensitivity_metric.measure(test_case)
    print(f"Cultural Sensitivity Metric Score for Query - {golden.input}: {cultural_sensitivity_metric.score}")
    historical_accuracy_metric.measure(test_case)
    print(f"Historical Accuracy Metric Score for Query - {golden.input}: {historical_accuracy_metric.score}")
    tourism_relevance_metric.measure(test_case)
    print(f"Tourism Relevance Metric Score for Query - {golden.input}: {tourism_relevance_metric.score}")
    educational_value_metric.measure(test_case)
    print(f"Educational Value Metric Score for Query - {golden.input}: {educational_value_metric.score}")
    completeness_metric.measure(test_case)
    print(f"Completeness Metric Score for Query - {golden.input}: {completeness_metric.score}")


    return {
        'query': golden.input,
        'actual_output': ollama_response,
        'expected_output': golden.expected_output,
        'evaluation_model': "gpt-4o-mini",
        'rag': {
            'precision': precision_metric.score,
            'recall': recall_metric.score,
            'relevancy': relevancy_metric.score,
        },
        'geval': {
            'cultural_sensitivity': cultural_sensitivity_metric.score,
            'historical_accuracy': historical_accuracy_metric.score,
            'tourism_relevance': tourism_relevance_metric.score,
            'educational_value': educational_value_metric.score,
            'completeness': completeness_metric.score,
        }
    }


# DeepEval Golden Test Cases for Jagannatha Temple
GOLDEN_TESTS = [
    Golden(
        input="What is the Jagannatha Temple?",
        expected_output="The Jagannatha Temple is a famous Hindu temple located in Puri, Odisha, India, dedicated to Lord Jagannatha (an incarnation of Lord Vishnu). It is one of the Char Dham pilgrimage sites and known for its unique wooden idols and annual Rath Yatra festival.",
        context=["The Jagannatha Temple in Puri, Odisha is one of the most sacred Hindu temples in India. Dedicated to Lord Jagannatha, it features distinctive wooden idols and hosts the famous Rath Yatra festival annually."]
    ),
    Golden(
        input="Describe the architecture of Jagannatha Temple",
        expected_output="The Jagannatha Temple features Kalinga architecture with a curved tower (shikhara) reaching about 58 meters high. The temple complex includes multiple structures like the main sanctum (garbhagriha), audience hall (natamandira), and kitchen (bhogamandapa). The idols are made of neem wood and replaced periodically.",
        context=["The temple showcases traditional Kalinga architecture with a 58-meter high shikhara. The complex includes garbhagriha, natamandira, and bhogamandapa. The deities are crafted from neem wood and renewed every 12-19 years."]
    ),
    Golden(
        input="What is the Rath Yatra festival at Jagannatha Temple?",
        expected_output="Rath Yatra is the annual chariot festival where the deities Jagannatha, Balabhadra, and Subhadra are pulled on massive wooden chariots through the streets of Puri. It occurs in the month of Ashadha (June-July) and attracts millions of devotees. The chariots are about 45 feet high and take several days to build.",
        context=["Rath Yatra is the famous chariot festival of Jagannatha Temple held annually in Ashadha. Millions of devotees pull the massive 45-foot chariots carrying the deities through Puri's streets. The festival symbolizes the journey from heaven to earth."]
    ),
    Golden(
        input="Where is Jagannatha Temple located?",
        expected_output="Jagannatha Temple is located in Puri, Odisha, on the eastern coast of India. Puri is about 60 km from Bhubaneswar, the state capital, and is well-connected by road, rail, and air. The temple is situated near the Bay of Bengal and is part of the Golden Triangle of Odisha tourism.",
        context=["Jagannatha Temple is situated in Puri, Odisha on India's eastern coast. Located 60 km from Bhubaneswar, Puri is accessible by road, rail, and air. The temple overlooks the Bay of Bengal and forms part of Odisha's Golden Triangle tourism circuit."]
    ),
    Golden(
        input="What are the famous foods offered at Jagannatha Temple?",
        expected_output="The temple kitchen prepares traditional Odia cuisine including khichdi, dalma, and various sweets. Prasad (sanctified food) is distributed to devotees and includes items like chakuli pitha and enduri pitha.",
        context=["The temple's kitchen serves traditional Odia cuisine as prasad. Famous offerings include khichdi, dalma, chakuli pitha, and enduri pitha. The kitchen feeds thousands of devotees daily."]
    ),
    Golden(
        input="What is the historical significance of Jagannatha Temple?",
        expected_output="Jagannatha Temple has immense historical and religious significance. It was originally built in the 12th century by King Anantavarman Chodaganga. The temple represents the unity of all religions and has been a center of spiritual learning for centuries.",
        context=["Built in the 12th century by King Anantavarman Chodaganga, the temple holds great historical significance. It serves as a center for spiritual learning and represents religious unity across different faiths."]
    )
]


if __name__ == "__main__":
    query = "Jagannatha Temple Puri Odisha"
    print("=" * 80)
    print("RAG EVALUATION")
    print("=" * 80)

    setup_ollama()
    evaluation_model = setup_custom_ollama_model_for_evaluation(model=get_ollama_model(ModelType.EVALUATION))
    # Step 1: Create vector database from Wikipedia
    print("\n📚 STEP 1: Updating Vector Database from Wikipedia")
    print(f"Topic: {{ {query} }}")
    print("=" * 80)
    vectordb = add_documents_to_vector_db(
        query=query,
        num_docs=8,
    )

    # Step 2: Create Evaluation Dataset and Run all golden tests
    print("\n\n🧪 CREATING DEEPEVAL EVALUATION DATASET")
    print("=" * 80)

    # Create EvaluationDataset from GOLDEN_TESTS
    dataset = EvaluationDataset(goldens=GOLDEN_TESTS)
    print(f"✅ Created EvaluationDataset with {len(dataset.goldens)} golden test cases")

    results = []
    test_descriptions = [
        "🧪 TEST 1: BASIC FACTS ABOUT JAGANNATHA TEMPLE",
        "🧪 TEST 2: ARCHITECTURE AND DESIGN",
        "🧪 TEST 3: RATH YATRA FESTIVAL",
        "🧪 TEST 4: LOCATION AND ACCESSIBILITY",
        "🧪 TEST 5: TEMPLE CUISINE AND PRASAD",
        "🧪 TEST 6: HISTORICAL SIGNIFICANCE"
    ]

    for i, (golden, description) in enumerate(zip(dataset.goldens, test_descriptions), 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {description.split(':')[1].strip()}")
        print(f"Question: {golden.input}")
        print("-" * 60)

        # Retrieve relevant documents using similarity search
        retrieved_docs = vectordb.similarity_search(golden.input, k=4)

        # Build prompt and generate response for display
        prompt_with_context = build_prompt_with_context(golden.input, retrieved_docs)
        ollama_response = generate_ollama_response(prompt_with_context, model_name=get_ollama_model(ModelType.EXPECTED_OUTPUT))
        print(f"Output: {ollama_response}")
        print("-" * 60)

        # Run the test with Golden
        result = test_single_query_all_metrics(
            golden=golden,
            retrieved_docs=retrieved_docs,
            evaluationModel=evaluation_model,
            test_description=description,
            custom_response=ollama_response,
            use_custom_response=True
        )

        results.append(result)


    # Save results to JSON file
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"deepeval_rag_evaluation_with_{timestamp}.json"

    json_data = {
        'detailed_results': results
    }

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Results saved to: {json_filename}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)