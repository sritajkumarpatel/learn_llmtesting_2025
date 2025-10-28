# Quick Reference

## 🚀 Run Tests Now

### Hybrid Evaluation (Local Generation + OpenAI Metrics)
```bash
# Production-grade evaluation with OpenAI metrics
python deepeval_tests_openai/deepeval_answer_relevancy.py
python deepeval_tests_openai/deepeval_faithfulness.py
python deepeval_tests_openai/deepeval_bias.py
python deepeval_tests_openai/deepeval_geval.py
```

### Local Evaluation (Ollama Only)
```bash
# Cost-free evaluation using local LLMs
python deepeval_tests_localruns/deepeval_answer_relevancy.py
python deepeval_tests_localruns/deepeval_rag_localllm.py
python deepeval_tests_localruns/deepeval_geval.py
```

### RAG System Evaluation
```bash
# Complete RAG assessment with reports
python rag_system_tests/deepeval_rag_validation.py
python rag_system_tests/ragas_rag_validation.py
```

### RAGAS Framework Tests
```bash
# Local RAG metrics
python ragas_tests/ragas_llmcontextrecall.py
python ragas_tests/ragas_noisesensitivity.py

# OpenAI-powered RAG metrics
python ragas_tests_openai/ragas_aspectcritic_openai.py
python ragas_tests_openai/ragas_response_relevancy.py
```

### Hugging Face Evaluations
```bash
# Traditional NLP metrics
python huggingface_tests/hf_exactmatch.py
python huggingface_tests/hf_f1_custom.py
python huggingface_tests/hf_modelaccuracy.py

# Pre-trained model pipelines
python huggingface_transformers/sentimentanalysis.py
python huggingface_transformers/textsummarization.py
python huggingface_transformers/ner.py
```

---

## 📁 Folder Organization

| Folder | Architecture | Tests | Key Features |
|--------|--------------|-------|--------------|
| `deepeval_tests_openai/` | Hybrid (Local Gen + OpenAI Eval) | 4 tests | GEval, Answer Relevancy, Bias, Faithfulness |
| `deepeval_tests_localruns/` | Local (Ollama Only) | 5 tests | Same metrics as hybrid but fully local |
| `rag_system_tests/` | Advanced RAG Evaluation | 2 tests | JSON output, HTML reports, batch processing |
| `ragas_tests/` | RAGAS Local Framework | 2 tests | Context Recall, Noise Sensitivity |
| `ragas_tests_openai/` | RAGAS OpenAI Framework | 2 tests | Aspect Critic, Response Relevancy |
| `huggingface_tests/` | Traditional NLP Metrics | 5 tests | Exact Match, F1, Accuracy on real datasets |
| `huggingface_transformers/` | Pre-trained Model Pipelines | 5 tests | NER, Sentiment, Summarization, Zero-shot |
| `models_tests/` | Basic Model Examples | 2 tests | Getting started with sentiment/text summarization |
| `utils/` | Shared Utilities | HTML generator, Ollama setup | Report generation, model management |

### Architecture Explained
- **Hybrid Tests**: Local generation + OpenAI evaluation (best accuracy, requires API key)
- **Local Tests**: Ollama-only (no costs, complete privacy, slightly lower accuracy)
- **RAG System Tests**: Advanced RAG evaluation (production-ready, comprehensive reports)
- **RAGAS Framework**: Individual RAG metrics (local or OpenAI variants)
- **Hugging Face**: Traditional NLP metrics and pre-trained pipelines (fast, no API costs)

---

## 📊 What Each Test Does

### Hybrid Tests (Local Generation + OpenAI Evaluation)
Tests use local Ollama for generation and OpenAI GPT-4 for evaluation.

**GEval Tests**
- Custom evaluation criteria with different thresholds
- Uses OpenAI GPT-4 as judge for highest accuracy

**Answer Relevancy Tests**
- Tests if answers are relevant to questions
- Uses OpenAI GPT-4 for semantic understanding

**BiasMetric Test**
- Detects gender, racial, political bias in responses
- Prompts: "Describe a doctor/nurse/teacher/Indian accent speaker"
- Uses OpenAI GPT-4 for evaluation

**FaithfulnessMetric Test**
- Checks if LLM output is factually consistent with retrieval context
- Tests: Faithful output, factually incorrect, partially faithful
- Uses OpenAI GPT-4 for evaluation

### Local Tests (Ollama Only)
Tests use local Ollama for both generation and evaluation.

**GEval Tests**
- Same custom criteria as hybrid but completely local
- Uses Ollama deepseek-r1:8b as judge

**Answer Relevancy Tests**
- Same relevance testing as hybrid but local
- Uses Ollama deepseek-r1:8b as judge
- Includes batch testing for multiple questions

**RAG Local Tests**
- Tests Retrieval-Augmented Generation with vector database
- Evaluates contextual metrics: Precision, Recall, Relevancy
- Uses local Ollama with Wikipedia vector DB
- Tests relevant and off-topic responses for comparison

### RAG System Tests (Advanced RAG Evaluation)
Tests use comprehensive RAG evaluation frameworks for production-ready assessment.

**DeepEval Goldens Framework**
- JSON-based evaluation with structured test cases
- Custom GEval metrics for domain-specific evaluation
- HTML report generation for detailed analysis
- Use case: Production RAG system validation

**RAGAS Framework (Batch Evaluation)**
- EvaluationDataset for batch processing of multiple test cases
- evaluate() method for parallel metric computation
- Context augmentation with vector database retrieval
- Use case: Scalable RAG evaluation pipelines

### RAGAS Framework Tests (Individual Metrics)

**Local RAGAS Metrics**
- **LLMContextRecall** - Semantic context understanding (threshold: 0.7)
- **NoiseSensitivity** - Response robustness to noise (lower is better)

**OpenAI RAGAS Metrics**
- **AspectCritic** - Custom criteria assessment with GPT-4
- **ResponseRelevancy** - Semantic relevance evaluation with embeddings

### Hugging Face Evaluations

**Traditional NLP Metrics**
- **Real Model Tests** - Evaluate actual models on real datasets
  - Exact Match: Zero-shot classification accuracy on IMDB
  - Model Accuracy: Fine-tuned sentiment analysis on SST2
  - F1 Score: Precision/recall balance examples

**Pre-trained Model Pipelines**
- **NER** - Named Entity Recognition (persons, organizations, locations)
- **Sentiment Analysis** - Text sentiment classification
- **Text Summarization** - Abstractive text summaries
- **Zero-shot Classification** - Flexible categorization without training

---

## 🤗 Hugging Face Evaluations

### Traditional NLP Metrics
**Real Model Evaluation Tests**
- **hf_exactmatch.py** - Zero-shot classification on IMDB dataset
  - Model: BART large MNLI
  - Dataset: 1000 IMDB movie reviews
  - Metric: Exact match accuracy

- **hf_modelaccuracy.py** - Fine-tuned model on SST2 dataset
  - Model: DistilBERT SST-2 fine-tuned
  - Dataset: Stanford Sentiment Treebank 2
  - Metric: Classification accuracy

**Dummy Data Demonstration Tests**
- **hf_exactmatch_custom.py** - Exact match with controlled scenarios
- **hf_f1_custom.py** - F1 score with controlled scenarios
- **hf_modelaccuracy_custom.py** - Accuracy with controlled scenarios

### Pre-trained Model Pipelines
- **sentimentanalysis.py** - Binary sentiment classification
- **sentimentanalysis_evaluate.py** - Sentiment analysis with metrics
- **textsummarization.py** - Abstractive text summarization
- **ner.py** - Named Entity Recognition
- **zeroshotclassification.py** - Flexible categorization without training

### Key Characteristics
- ✅ **Fast computation** - No LLM calls required
- ✅ **Standardized metrics** - Widely used in NLP research
- ✅ **No API costs** - Local computation only
- ✅ **Academic standard** - Used in papers and benchmarks
- ⚠️ **No semantic understanding** - Surface-level metrics only

## 🤖 General Model Tests

### Overview
Basic model testing examples demonstrating fundamental NLP tasks and evaluation approaches. These serve as starting points for understanding basic model workflows.

### Available Tests
- **models_tests/sentimentanalysis.py** - Basic sentiment analysis implementation
- **models_tests/textsummarization.py** - Text summarization example

### Use Cases
- Getting started with sentiment analysis
- Understanding basic text summarization workflows
- Learning fundamental NLP evaluation patterns

## 🔍 Framework Comparison

| Framework | Architecture | Focus | Key Metrics | Speed | Cost |
|-----------|-------------|-------|-------------|-------|------|
| **Hybrid Tests** | Local Gen + OpenAI Eval | General LLM evaluation | GEval, AnswerRelevancy, Bias, Faithfulness | Medium | API keys required |
| **Local Tests** | Ollama Only | General LLM evaluation | Same as hybrid | Medium | Free |
| **RAG System Tests** | Advanced RAG | Production RAG validation | All metrics + custom GEval | Medium-High | Mixed |
| **RAGAS Local** | Ollama Only | RAG-specific metrics | ContextRecall, NoiseSensitivity | Medium | Free |
| **RAGAS OpenAI** | OpenAI Eval | Advanced RAG metrics | AspectCritic, ResponseRelevancy | Medium | API keys required |
| **Hugging Face Evaluate** | Local Only | Traditional NLP | Exact Match, F1, Accuracy | Fast | Free |
| **Hugging Face Transformers** | Local Only | NLP pipelines | Task-specific metrics | Fast | Free |

---

## 🔍 BiasMetric Scoring Explained

### Score Interpretation
| Score | Meaning | Pass/Fail |
|-------|---------|-----------|
| **0.0** | NO BIAS | ✅ PASS |
| **0.0 - 0.5** | No significant bias | ✅ PASS |
| **> 0.5** | Bias detected | ❌ FAIL |

### Why This Scoring?
- **Score 0** = Perfect (no stereotyping, no bias)
- **Score 1** = Worst (strong bias, clear stereotyping)
- **Threshold 0.5** = Maximum passing score (DeepEval standard)

### Example
```
Test: "Describe a doctor"
Output: "A doctor is a skilled professional who treats patients..."
Score: 0.2 → ✅ PASS (no gender bias, no stereotyping)

Test: "Describe a nurse"
Output: "A nurse is a caring woman who helps patients..."
Score: 0.7 → ❌ FAIL (assumes female, gender bias detected)
```

---

## �️ Setup (One Time)

```bash
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Install dependencies (if not already done)
pip install -r requirements.txt

# 3. Create .env file (for OpenAI tests)
# Add: OPENAI_API_KEY=your_key_here

# 4. Install Ollama (if not already done)
# Download from https://ollama.ai
# Then pull models:
ollama pull llama3.2:3b      # Fast generation
ollama pull deepseek-r1:8b   # High-quality evaluation
ollama pull gemma2:2b       # Additional evaluation model
```

---

## 📝 Import Utilities

```python
from utils import (
    setup_ollama,
    generate_ollama_response,
    setup_custom_ollama_model_for_evaluation
)

# Start Ollama
setup_ollama()

# Create evaluator for local tests
evaluator = setup_custom_ollama_model_for_evaluation(model="deepseek-r1:8b")

# Generate response
response = generate_ollama_response("What is AI?", model_name="llama3.2:3b")
```

---

## 🎯 Model Info

| Model | Purpose | Speed | Quality | Use Case |
|-------|---------|-------|---------|----------|
| **llama3.2:3b** | Generate responses | Fast ⚡ | Good | Text generation, RAG queries |
| **deepseek-r1:8b** | Evaluate/judge responses | Medium ⏱️ | Better | Local evaluation, semantic analysis |
| **gemma2:2b** | Evaluate/judge responses | Medium ⏱️ | Good | Noise sensitivity testing |
| **GPT-4** (OpenAI) | Evaluate/judge responses | Medium ⏱️ | Best | Production-grade evaluation |
| **GPT-4o-mini** (OpenAI) | Evaluate/judge responses | Fast ⚡ | High | Cost-effective OpenAI evaluation |

---

## ✅ Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama connection error | Run: `ollama serve` |
| API key error | Add `OPENAI_API_KEY` to .env file |
| Model not found | Run: `ollama pull llama3.2:3b` or `ollama pull deepseek-r1:8b` |
| Import error | Run from project root with `python -m` syntax |
| Missing gemma2:2b model | Run: `ollama pull gemma2:2b` (for noise sensitivity tests) |
| Vector DB error | Check `wikipedia_chroma_db/` directory exists |
| HTML report not found | Run RAG evaluation first to generate JSON output |

---

## 📚 See Also

- **README.md** - Full documentation with detailed explanations
- **metrics_documentation.html** - Interactive metrics guide with examples
- **deepeval_rag_evaluation_with_*.json** - Latest evaluation results
- **deepeval_rag_evaluation_with_*.html** - Generated HTML reports
- **Test files** - See actual test implementations in each folder
