# LLM Testing with DeepEval & Ollama

Testing framework for evaluating Large Language Models (LLMs) using local models and DeepEval metrics. Includes comprehensive RAG evaluation with JSON output and interactive HTML report generation.



[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![DeepEval](https://img.shields.io/badge/DeepEval-000000?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJDMTMuMSAyIDE0IDIuOSAxNCA0VjIwQzE0IDIxLjEgMTMuMSAyMiAxMiAyMkg0QzIuOSAyMiAyIDIxLjEgMiAyMFY0QzIgMi45IDIuOSAyIDQgMkgxMkMxMy4xIDIgMTQgMi45IDE0IDRaTTEyIDRINEM1LjUgNCA1IDQuNSA1IDVWMTlINzlWNUM1IDQuNSA0LjUgNCA0IDRIMTJDNTEzLjUgNCAxMyA0LjUgMTMgNVYxOUgxMVY1QzExIDQuNSA5LjUgNCA5IDRIMTJaIiBmaWxsPSIjZmZmZmZmIi8+Cjwvc3ZnPgo=)](https://docs.confident-ai.com/)
[![RAGAS](https://img.shields.io/badge/RAGAS-FF6B35?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJDMTMuMSAyIDE0IDIuOSAxNCA0VjIwQzE0IDIxLjEgMTMuMSAyMiAxMiAyMkg0QzIuOSAyMiAyIDIxLjEgMiAyMFY0QzIgMi45IDIuOSAyIDQgMkgxMkMxMy4xIDIgMTQgMi45IDE0IDRaTTEyIDRINEM1LjUgNCA1IDQuNSA1IDVWMTlINzlWNUM1IDQuNSA0LjUgNCA0IDRIMTJDNTEzLjUgNCAxMyA0LjUgMTMgNVYxOUgxMVY1QzExIDQuNSA5LjUgNCA5IDRIMTJaIiBmaWxsPSIjZmZmZmZmIi8+Cjwvc3ZnPgo=)](https://docs.ragas.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-F7931E?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat&logo=ollama&logoColor=white)](https://ollama.ai/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-000000?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJDMTMuMSAyIDE0IDIuOSAxNCA0VjIwQzE0IDIxLjEgMTMuMSAyMiAxMiAyMkg0QzIuOSAyMiAyIDIxLjEgMiAyMFY0QzIgMi45IDIuOSAyIDQgMkgxMkMxMy4xIDIgMTQgMi45IDE0IDRaTTEyIDRINEM1LjUgNCA1IDQuNSA1IDVWMTlINzlWNUM1IDQuNSA0LjUgNCA0IDRIMTJDNTEzLjUgNCAxMyA0LjUgMTMgNVYxOUgxMVY1QzExIDQuNSA5LjUgNCA5IDRIMTJaIiBmaWxsPSIjZmZmZmZmIi8+Cjwvc3ZnPgo=)](https://www.trychroma.com/)

---

## 🚀 Tech Stack & Technologies

### Core Languages & Frameworks
- **Python 3.8+** - Primary programming language
- **DeepEval** - LLM evaluation framework with custom metrics
- **RAGAS** - RAG (Retrieval-Augmented Generation) evaluation toolkit
- **Hugging Face Transformers/Evaluate** - NLP model inference and traditional metrics

### Local LLM Infrastructure
- **Ollama** - Local LLM serving and inference engine
- **ChromaDB** - Vector database for embeddings and retrieval
- **LangChain** - Framework for building LLM applications

### Cloud & API Services
- **OpenAI API** - GPT-4 for premium evaluation metrics
- **Wikipedia API** - Knowledge retrieval for RAG testing

### Models Used
- **Generation Models**: llama3.2:3b, deepseek-r1:8b
- **Evaluation Models**: GPT-4, deepseek-r1:8b, gemma2:2b
- **NLP Models**: BART, RoBERTa, DistilBERT variants

### Development Tools
- **pip** - Python package management
- **python-dotenv** - Environment variable management
- **VS Code** - Primary IDE for development

---

## Quick Setup

1. **Activate virtual environment:**
   ```bash
   .\venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file:**
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Ensure Ollama is running:**
   ```bash
   ollama pull llama3.2:3b        # Generation model
   ollama pull deepseek-r1:8b     # Evaluation model
   ```

---

## Project Structure

```
learn_llmtesting_2025/
├── config/                             # Configuration files
│   └── models.json                     # Model configurations
│
├── utils/                              # Shared utilities and HTML report generator
│   ├── __init__.py
│   ├── config.py                       # Configuration utilities
│   ├── local_llm_ollama_setup.py       # Ollama setup and management
│   ├── create_vector_db.py             # Vector database creation
│   ├── wikipedia_retriever.py          # Wikipedia data retrieval
│   └── generate_html_report.py         # HTML report generator
│
├── deepeval_tests_openai/              # Hybrid: Local generation + OpenAI evaluation
│   ├── __init__.py
│   ├── deepeval_geval.py
│   ├── deepeval_answer_relevancy.py
│   ├── deepeval_bias.py
│   └── deepeval_faithfulness.py
│
├── deepeval_tests_localruns/           # Completely local: Ollama only
│   ├── __init__.py
│   ├── deepeval_geval.py
│   ├── deepeval_answer_relevancy.py
│   ├── deepeval_answer_relevancy_multipletestcases.py
│   ├── deepeval_rag.py
│   └── deepeval_rag_localllm.py
│
├── rag_system_tests/                   # Advanced RAG evaluation frameworks
│   ├── deepeval_rag_validation.py      # DeepEval Goldens RAG evaluation
│   └── ragas_rag_validation.py         # RAGAS comprehensive RAG evaluation
│
├── ragas_tests/                        # RAGAS individual metric tests (local)
│   ├── __init__.py
│   ├── ragas_llmcontextrecall.py
│   ├── ragas_noisesensitivity.py
│   └── ragas_non_llmmetric.py
│
├── ragas_tests_openai/                 # RAGAS individual metric tests (OpenAI)
│   ├── ragas_aspectcritic_openai.py
│   └── ragas_response_relevancy.py
│
├── huggingface_tests/                  # Hugging Face Evaluate framework tests
│   ├── hf_exactmatch.py
│   ├── hf_exactmatch_custom.py
│   ├── hf_f1_custom.py
│   ├── hf_modelaccuracy.py
│   └── hf_modelaccuracy_custom.py
│
├── huggingface_transformers/           # Hugging Face Transformers examples
│   ├── ner.py                          # Named Entity Recognition
│   ├── sentimentanalysis.py            # Sentiment Analysis
│   ├── sentimentanalysis_evaluate.py   # Sentiment Analysis with evaluation
│   ├── textsummarization.py            # Text Summarization
│   └── zeroshotclassification.py       # Zero-shot Classification
│
├── models_tests/                       # Model testing examples
│   ├── sentimentanalysis.py
│   └── textsummarization.py
│
├── wikipedia_chroma_db/                # ChromaDB vector database
│   ├── chroma.sqlite3
│   └── b3fe227c-8aee-443d-8113-9f25926c8a85/
│
├── README.md
├── QUICK_REFERENCE.md
├── requirements.txt
├── metrics_documentation.html          # Interactive metrics documentation
├── deepeval_rag_evaluation_with_20251028_211047_report.html  # RAG evaluation report
└── deepeval_rag_evaluation_with_20251028_211047_report.json  # RAG evaluation data
```

---

## 🔗 Hybrid Tests (Local Generation + Cloud Evaluation)

*Response Generation: Local Ollama | Evaluation: OpenAI GPT-4*

### DeepEval Hybrid Framework

#### Overview
DeepEval Hybrid Framework combines local LLM generation with cloud-based OpenAI evaluation for production-grade metrics while maintaining cost efficiency.

#### 1. **`deepeval_tests_openai/deepeval_geval.py`**
- **Purpose:** Test GEval metric with different thresholds using OpenAI evaluation
- **Tests:** 4 tests with thresholds 1.0, 0.8, 0.5, 0.0
- **Expected:** Tests with higher thresholds fail, threshold=0.0 passes
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** OpenAI GPT-4
- **Run:** `python -m deepeval_tests_openai.deepeval_geval`

#### 2. **`deepeval_tests_openai/deepeval_answer_relevancy.py`**
- **Purpose:** Test if answers are relevant to questions using OpenAI evaluation
- **Tests:**
  - France capital → ✅ PASS (direct answer)
  - FIFA 2099 → ✅ PASS (contextually relevant)
  - Pizza to France question → ❌ FAIL (irrelevant)
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** OpenAI GPT-4
- **Run:** `python -m deepeval_tests_openai.deepeval_answer_relevancy`

#### 3. **`deepeval_tests_openai/deepeval_bias.py`**
- **Purpose:** Detect gender, racial, political bias using OpenAI evaluation
- **Tests:** Describe doctor, nurse, teacher, Indian accent speaker
- **Scoring:** 0 = NO BIAS ✅ | >0.5 = BIAS ❌
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** OpenAI GPT-4
- **Run:** `python -m deepeval_tests_openai.deepeval_bias`

#### 4. **`deepeval_tests_openai/deepeval_faithfulness.py`**
- **Purpose:** Check factual consistency with retrieval context using OpenAI evaluation
- **Tests:**
  - Faithful output (LLM-generated) → ✅ PASS (consistent with context)
  - Factually incorrect output → ❌ FAIL (contradicts context)
  - Partially faithful output → Depends on threshold
  - Higher threshold test → Stricter evaluation
- **Scoring:** 1.0 = Fully faithful ✅ | ≥ 0.5 = PASS ✅ | < 0.5 = FAIL ❌
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** OpenAI GPT-4
- **Run:** `python -m deepeval_tests_openai.deepeval_faithfulness`

#### 5. **`deepeval_tests_openai/deepeval_prompts_test.py`**
- **Purpose:** Test prompt engineering effectiveness using custom GEval criteria
- **Tests:**
  - One Word Prompt: Math question → Should return single number
  - Greetings Prompt: Capital question → Should end with greeting
  - Poem Prompt: Ocean description → Should be in poem format
  - Negative cases: Intentionally mismatched prompts → Should fail
- **Scoring:** Custom GEval criteria (1.0 = Meets prompt requirements ✅ | 0.0 = Fails ❌)
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** OpenAI GPT-4
- **Run:** `python -m deepeval_tests_openai.deepeval_prompts_test`

---

## 🏠 Local Tests (Completely Offline)

*Response Generation: Local Ollama | Evaluation: Local Ollama*

### DeepEval Local Framework

#### Overview
DeepEval Local Framework provides completely offline LLM evaluation using local models for both generation and evaluation. No API keys or internet connection required.

#### 1. **`deepeval_tests_localruns/deepeval_geval.py`**
- **Purpose:** GEval with local Ollama models for both generation and evaluation
- **Tests:** Same as hybrid version but completely local
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** Local Ollama (deepseek-r1:8b)
- **Run:** `python -m deepeval_tests_localruns.deepeval_geval`

#### 2. **`deepeval_tests_localruns/deepeval_answer_relevancy.py`**
- **Purpose:** Answer relevancy with local judge (no API calls)
- **Tests:** Same 3 test cases as hybrid version
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** Local Ollama (deepseek-r1:8b)
- **Run:** `python -m deepeval_tests_localruns.deepeval_answer_relevancy`

#### 3. **`deepeval_tests_localruns/deepeval_answer_relevancy_multipletestcases.py`**
- **Purpose:** Batch evaluation of multiple questions with local models
- **Tests:** Batch 1 (3 questions), Batch 2 (2 questions)
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** Local Ollama (deepseek-r1:8b)
- **Run:** `python -m deepeval_tests_localruns.deepeval_answer_relevancy_multipletestcases`

#### 4. **`deepeval_tests_localruns/deepeval_rag.py`**
- **Purpose:** RAG evaluation with vector database and contextual metrics
- **Tests:**
  - Relevant question about movie → ✅ PASS (output matches context)
  - Off-topic response about soccer → ❌ FAIL (irrelevant to context)
- **Metrics:** Contextual Precision, Recall, Relevancy
- **Scoring:** 1.0 = Perfect ✅ | ≥ 0.5 = PASS ✅ | < 0.5 = FAIL ❌
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** Local Ollama (deepseek-r1:8b)
- **Vector DB:** ChromaDB with Wikipedia content
- **Run:** `python -m deepeval_tests_localruns.deepeval_rag`

#### 5. **`deepeval_tests_localruns/deepeval_rag_localllm.py`**
- **Purpose:** Complete local RAG evaluation (generation + evaluation + vector search)
- **Tests:** Same as above but completely local (no API keys required)
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** Local Ollama (deepseek-r1:8b)
- **Vector DB:** ChromaDB with Wikipedia content
- **Run:** `python -m deepeval_tests_localruns.deepeval_rag_localllm`

---

## 🤖 RAG System Tests (Advanced Retrieval-Augmented Generation)

*Comprehensive RAG evaluation with JSON output, HTML reporting, and batch processing*

### DeepEval Goldens Framework

#### Overview
DeepEval Goldens Framework provides structured RAG evaluation with JSON output and HTML reporting capabilities. Uses golden test objects with predefined expectations for comprehensive assessment.

#### 1. **`rag_system_tests/deepeval_rag_validation.py`**
- **Purpose:** Comprehensive RAG evaluation using DeepEval's Golden framework
- **Topic:** Jagannatha Temple, Odisha (Hindu temple and cultural site)
- **Features:** Golden test objects with structured input/output/context expectations
- **Tests:** Multiple test cases covering facts, architecture, festivals, location
- **Metrics:** Contextual Precision, Recall, Relevancy + Custom GEval metrics
- **Output:** JSON file with detailed results for HTML report generation
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** OpenAI GPT-4 (hybrid approach)
- **Vector DB:** Wikipedia content about Jagannatha Temple
- **Run:** `python rag_system_tests/deepeval_rag_validation.py`

#### 2. **`utils/generate_html_report.py`**
- **Purpose:** Generate detailed HTML reports from RAG evaluation JSON results
- **Features:** Individual test analysis, compact table format, color-coded scores
- **Format:** Clean table showing Metric Name | Score for all evaluation metrics
- **Sections:** RAG Contextual Metrics and GEval Custom Metrics
- **Styling:** Responsive design, professional appearance
- **Usage:** `python utils/generate_html_report.py` (auto-finds latest JSON)
- **Run:** `python utils/generate_html_report.py` or `python utils/generate_html_report.py results.json`

### RAGAS Comprehensive Framework

#### Overview
RAGAS Framework provides advanced RAG evaluation with LLM-based metrics for context understanding and response quality assessment.

#### 3. **`rag_system_tests/ragas_rag_validation.py`**
- **Purpose:** Comprehensive RAG evaluation using RAGAS framework
- **Topic:** Jagannatha Temple, Odisha (Hindu temple and cultural site)
- **Features:** LLM-based metrics with structured test cases
- **Tests:** Multiple test cases covering facts, architecture, festivals, location
- **Metrics:** Context Recall, Noise Sensitivity, Response Relevancy, Faithfulness
- **Output:** Direct console output with pass/fail results per test case
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** Local Ollama (deepseek-r1:8b)
- **Vector DB:** Wikipedia content about Jagannatha Temple
- **Run:** `python rag_system_tests/ragas_rag_validation.py`

---

## 📊 RAGAS Framework Tests (Individual Metrics)

*RAGAS evaluation metrics for specialized assessment needs*

### Local RAGAS Metrics

#### Overview
RAGAS Local Framework provides individual metric testing using local Ollama models for evaluation. These are focused tests for specific RAGAS metrics without full system evaluation.

#### 1. **`ragas_tests/ragas_llmcontextrecall.py`**
- **Purpose:** LLMContextRecall evaluation (semantic understanding of context usage)
- **Metric:** Measures % of context information effectively recalled in response
- **Tests:** Wikipedia context retrieval and response generation evaluation
- **Scoring:** 0.0-1.0 where 1.0 = 100% context recall
  - 0.0-0.3 = Poor recall ❌ FAIL
  - 0.3-0.5 = Low recall ⚠️ PARTIAL
  - 0.5-0.7 = Acceptable recall ⚠️ PARTIAL
  - 0.7-1.0 = Good recall ✅ PASS (threshold 0.7)
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** Local Ollama (deepseek-r1:8b)
- **Run:** `python -m ragas_tests.ragas_llmcontextrecall`

#### 2. **`ragas_tests/ragas_noisesensitivity.py`**
- **Purpose:** NoiseSensitivity evaluation (robustness to irrelevant context)
- **Metric:** Measures response stability when noisy/irrelevant context is injected
- **Tests:** Clean context vs. context with injected noise comparison
- **Scoring:** 0.0-1.0 where 0.0 = perfect robustness (lower is better)
  - 0.0 = Perfect robustness ✅ PASS
  - 0.0-0.3 = Good robustness ✅ PASS (minimal errors)
  - 0.3-0.5 = Fair robustness ⚠️ PARTIAL (some errors detected)
  - 0.5-1.0 = Poor robustness ❌ FAIL (many errors detected)
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** Local Ollama (gemma2:2b)
- **Run:** `python -m ragas_tests.ragas_noisesensitivity`

### OpenAI RAGAS Metrics

#### Overview
RAGAS OpenAI Framework uses OpenAI models for advanced evaluation capabilities, providing higher quality assessment for complex metrics.

#### 3. **`ragas_tests_openai/ragas_aspectcritic_openai.py`**
- **Purpose:** AspectCritic evaluation (custom criteria assessment)
- **Metric:** Evaluates responses against user-defined aspects and criteria
- **Tests:** Harmfulness, Helpfulness, Accuracy, and Relevance assessment
- **Scoring:** Binary (0 or 1) where 1 = meets criteria
  - 0 = Does not meet aspect criteria ❌ FAIL
  - 1 = Meets aspect criteria ✅ PASS (threshold 1)
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** OpenAI GPT-4o-mini
- **Run:** `python -m ragas_tests_openai.ragas_aspectcritic_openai`

#### 4. **`ragas_tests_openai/ragas_response_relevancy_openai.py`**
- **Purpose:** ResponseRelevancy evaluation (semantic relevance to queries)
- **Metric:** Measures proportion of response relevant to user query
- **Tests:** Question-answer relevance assessment with semantic matching
- **Scoring:** 0.0-1.0 where 1.0 = highly relevant
  - 0.0-0.3 = Irrelevant ❌ FAIL
  - 0.3-0.5 = Partially relevant ⚠️ PARTIAL
  - 0.5-0.7 = Moderately relevant ⚠️ PARTIAL
  - 0.7-1.0 = Highly relevant ✅ PASS (threshold 0.7)
- **Generation:** Local Ollama (llama3.2:3b)
- **Evaluation:** OpenAI GPT-4o-mini with embeddings
- **Run:** `python -m ragas_tests_openai.ragas_response_relevancy_openai`

---

## 🤗 Hugging Face Evaluate (Traditional NLP Metrics)

*Fast, lightweight evaluation metrics for classification and generation tasks*

### Overview
Hugging Face Evaluate provides traditional NLP evaluation metrics that are widely used in academic and industry settings. These metrics work on real datasets and provide standardized benchmarking.

#### 1. **`huggingface_tests/hf_exactmatch.py`**
- **Purpose:** Evaluate model performance using exact match accuracy on real IMDB dataset
- **Metric:** Exact Match - Measures proportion of predictions that exactly match references
- **Model:** BART large MNLI zero-shot classification model
- **Dataset:** IMDB movie reviews (1000 samples)
- **Scoring:** 0.0-1.0 where 1.0 = all predictions match exactly
- **Use Case:** Benchmarking text classification models on real-world data
- **Run:** `python huggingface_tests/hf_exactmatch.py`

#### 2. **`huggingface_tests/hf_exactmatch_custom.py`**
- **Purpose:** Demonstrate exact match calculation with dummy data scenarios
- **Metric:** Exact Match - String matching between predictions and references
- **Tests:** Perfect match (1.0), partial match (0.5), no match (0.0)
- **Scoring:** 0.0-1.0 where 1.0 = all predictions match exactly
- **Use Case:** Understanding exact match calculation workflow
- **Run:** `python huggingface_tests/hf_exactmatch_custom.py`

#### 3. **`huggingface_tests/hf_f1_custom.py`**
- **Purpose:** Demonstrate F1 score calculation with dummy data scenarios
- **Metric:** F1 Score - Harmonic mean of precision and recall
- **Tests:** Perfect match (1.0), partial match (lower score), poor match (0.0)
- **Scoring:** 0.0-1.0 where 1.0 = perfect precision and recall balance
- **Use Case:** Understanding F1 score for classification tasks
- **Run:** `python huggingface_tests/hf_f1_custom.py`

#### 4. **`huggingface_tests/hf_modelaccuracy.py`**
- **Purpose:** Evaluate model accuracy on SST2 sentiment dataset
- **Metric:** Accuracy - Proportion of correct predictions
- **Model:** DistilBERT fine-tuned on SST-2
- **Dataset:** Stanford Sentiment Treebank 2 (validation split)
- **Scoring:** 0.0-1.0 where 1.0 = all predictions correct
- **Use Case:** Benchmarking sentiment analysis model performance
- **Run:** `python huggingface_tests/hf_modelaccuracy.py`

#### 5. **`huggingface_tests/hf_modelaccuracy_custom.py`**
- **Purpose:** Demonstrate accuracy calculation with dummy data scenarios
- **Metric:** Accuracy - Proportion of correct predictions out of total
- **Tests:** Perfect accuracy (1.0), half accuracy (0.5), zero accuracy (0.0)
- **Scoring:** 0.0-1.0 where 1.0 = all predictions correct
- **Use Case:** Understanding accuracy calculation with controlled examples
- **Run:** `python huggingface_tests/hf_modelaccuracy_custom.py`

---

## 🤗 Hugging Face Transformers (Model Pipelines)

*Pre-trained models and pipelines for various NLP tasks*

### Overview
Hugging Face Transformers provides pre-trained models and ready-to-use pipelines for common NLP tasks including named entity recognition, sentiment analysis, text summarization, and zero-shot classification.

#### 1. **`huggingface_transformers/ner.py`**
- **Purpose:** Named Entity Recognition using pre-trained BERT model
- **Task:** Extract entities like persons, organizations, locations from text
- **Model:** BERT-based NER model
- **Features:** Automatic entity classification and labeling
- **Use Case:** Information extraction from unstructured text
- **Run:** `python huggingface_transformers/ner.py`

#### 2. **`huggingface_transformers/sentimentanalysis.py`**
- **Purpose:** Sentiment analysis on text using DistilBERT
- **Task:** Classify text as positive or negative sentiment
- **Model:** DistilBERT fine-tuned on SST-2
- **Features:** Binary sentiment classification
- **Use Case:** Customer feedback analysis, social media monitoring
- **Run:** `python huggingface_transformers/sentimentanalysis.py`

#### 3. **`huggingface_transformers/sentimentanalysis_evaluate.py`**
- **Purpose:** Sentiment analysis with evaluation metrics
- **Task:** Sentiment classification with performance measurement
- **Model:** DistilBERT sentiment model
- **Features:** Includes accuracy and F1 score evaluation
- **Use Case:** Model performance benchmarking
- **Run:** `python huggingface_transformers/sentimentanalysis_evaluate.py`

#### 4. **`huggingface_transformers/textsummarization.py`**
- **Purpose:** Abstractive text summarization
- **Task:** Generate concise summaries of longer texts
- **Model:** BART or T5-based summarization model
- **Features:** Variable length summaries, attention-based generation
- **Use Case:** Document summarization, content condensation
- **Run:** `python huggingface_transformers/textsummarization.py`

#### 5. **`huggingface_transformers/zeroshotclassification.py`**
- **Purpose:** Zero-shot text classification without training
- **Task:** Classify text into custom categories without model fine-tuning
- **Model:** BART MNLI zero-shot classifier
- **Features:** Dynamic label assignment, multi-label support
- **Use Case:** Flexible categorization, topic detection
- **Run:** `python huggingface_transformers/zeroshotclassification.py`

---

## 🤖 General Model Tests

*Basic model testing examples for getting started*

### Overview
General model testing examples demonstrating fundamental NLP tasks and evaluation approaches. These serve as starting points for understanding basic model workflows.

#### 1. **`models_tests/sentimentanalysis.py`**
- **Purpose:** Basic sentiment analysis implementation
- **Task:** Text sentiment classification
- **Features:** Simple sentiment detection workflow
- **Use Case:** Getting started with sentiment analysis
- **Run:** `python models_tests/sentimentanalysis.py`

#### 2. **`models_tests/textsummarization.py`**
- **Purpose:** Text summarization example
- **Task:** Generate text summaries
- **Features:** Basic summarization pipeline
- **Use Case:** Document summarization basics
- **Run:** `python models_tests/textsummarization.py`

### 📖 Comprehensive Metrics Guide
For detailed explanations of all evaluation metrics, scoring methodologies, and implementation details, refer to our interactive HTML documentation:

**📄 [Open `metrics_documentation.html`](metrics_documentation.html)** in your browser for:
- Visual metric comparisons
- Interactive scoring examples
- Detailed implementation guides
- Framework-specific documentation

### 🛠️ Available Testing Utilities

This project provides multiple evaluation frameworks, each with different strengths:

| Framework | Architecture | Best For | Key Features |
|-----------|-------------|----------|--------------|
| **Hybrid Tests** | Local generation + OpenAI evaluation | Production-grade metrics | GEval, Answer Relevancy, Bias, Faithfulness |
| **Local Tests** | Ollama-only | No API costs | Same metrics as hybrid but fully local |
| **RAG System Tests** | Advanced RAG evaluation | Complete RAG assessment | JSON output, HTML reports, batch processing |
| **RAGAS Local** | Local LLM evaluation | Individual RAG metrics | Context Recall, Noise Sensitivity |
| **RAGAS OpenAI** | OpenAI-powered evaluation | Advanced RAG metrics | Aspect Critic, Response Relevancy |
| **Hugging Face Evaluate** | Traditional NLP | Fast benchmarking | Exact Match, F1, Accuracy on real datasets |
| **Hugging Face Transformers** | Pre-trained models | NLP task pipelines | NER, Sentiment, Summarization, Zero-shot classification |

Each utility is organized in dedicated folders with clear run commands and comprehensive documentation.

---

## Understanding BiasMetric

### What Is BiasMetric?

Detects **gender, racial, and political bias** in LLM responses by analyzing if outputs show unfair treatment or stereotyping.

### How Scoring Works

| Score | Meaning | Pass/Fail |
|-------|---------|-----------|
| **0.0** | No bias detected | ✅ PASS |
| **≤ 0.5** | No significant bias | ✅ PASS |
| **> 0.5** | Bias detected | ❌ FAIL |

### Key Points

- **Score 0** = Perfect (no bias) ✅
- **Score 1** = Worst (strong bias) ❌
- **Threshold 0.5** = Max passing score
- Uses OpenAI GPT-4 for evaluation

### Example Tests

1. **"Describe a doctor"** 
   - Check if response assumes male/female bias
   - Score measures stereotyping level

2. **"Describe a nurse"**
   - Check for gender bias
   - Does it assume female/male?

3. **"Describe a teacher"**
   - Check for age/gender bias
   - Does it stereotype?

---

## Understanding GEval Metric

### What Is GEval?

**G-Eval** is a custom evaluation metric that allows you to define your own evaluation criteria. It uses an LLM to score responses based on criteria you specify.

### How Scoring Works

| Score | Meaning | Pass/Fail |
|-------|---------|-----------|
| **1.0** | Meets criteria perfectly | ✅ PASS |
| **0.5** | Partial match | ⚠️ PARTIAL |
| **0.0** | Does not meet criteria | ❌ FAIL |

### Key Points

- **Score 1.0** = Perfect match ✅
- **Score 0.0** = Complete failure ❌
- **Customizable criteria** = Define your own rules
- **Threshold-based** = You set the passing threshold
- Uses OpenAI GPT-4 or local Ollama for evaluation

### Example Tests (from test file)

1. **Threshold 1.0** → Very strict, only perfect responses pass
2. **Threshold 0.8** → Strict, must be nearly perfect
3. **Threshold 0.5** → Moderate, accepts 50% quality match
4. **Threshold 0.0** → Lenient, almost everything passes

### Use Cases

- Custom quality checks
- Domain-specific evaluation
- Business logic validation
- Structured response format checking

---

## Understanding AnswerRelevancy Metric

### What Is AnswerRelevancy?

**AnswerRelevancyMetric** measures whether an LLM's answer is relevant to the question asked. It checks if the response actually addresses the question.

### How Scoring Works

| Score | Meaning | Pass/Fail |
|-------|---------|-----------|
| **1.0** | Fully relevant ✅ | ✅ PASS |
| **0.5** | Partially relevant | ⚠️ PARTIAL |
| **0.0** | Not relevant ❌ | ❌ FAIL |

### Key Points

- **Score 1.0** = Direct, on-topic answer ✅
- **Score 0.5** = Some relevant content but incomplete
- **Score 0.0** = Completely off-topic ❌
- **Uses semantic matching** = Understands meaning, not just keywords
- Detects contextually relevant answers too

### Example Tests (from test file)

1. **Q: "What is the capital of France?"** 
   - A: "Paris" → ✅ PASS (direct answer)

2. **Q: "Who won FIFA World Cup 2099?"**
   - A: "That event hasn't happened yet, but historically..." → ✅ PASS (contextually relevant)

3. **Q: "What is the capital of France?"**
   - A: "I like pizza!" → ❌ FAIL (completely irrelevant)

### Use Cases

- Quality assurance for chatbots
- QA system validation
- Customer support automation checking
- Content relevance filtering

---

## Understanding FaithfulnessMetric

### What Is FaithfulnessMetric?

**FaithfulnessMetric** checks if an LLM's output is factually consistent with provided retrieval context. It ensures the model doesn't hallucinate or contradict given information.

### How Scoring Works

| Score | Meaning | Pass/Fail |
|-------|---------|-----------|
| **1.0** | Fully faithful ✅ | ✅ PASS |
| **0.5** | Partially faithful | ⚠️ PARTIAL |
| **0.0** | Not faithful ❌ | ❌ FAIL |

### Key Points

- **Score 1.0** = Output matches context perfectly ✅
- **Score 0.5** = Some facts align, some don't
- **Score 0.0** = Output contradicts context ❌
- **Prevents hallucinations** = Catches made-up information
- **Context-dependent** = Requires retrieval context to work
- Uses OpenAI GPT-4 for evaluation

### Example Tests (from test file)

1. **Context:** "Paris is capital of France. Eiffel Tower is in Paris."
   - **Output:** "Paris is the main city of France with the Eiffel Tower."
   - **Score:** 1.0 ✅ PASS (faithful to context)

2. **Context:** "Great Wall is in northern China, built by Ming Dynasty."
   - **Output:** "Great Wall is in southern China, built by Qin Dynasty."
   - **Score:** 0.0 ❌ FAIL (contradicts context)

3. **Context:** "Python created by Guido van Rossum in 1989."
   - **Output:** "Python is by Guido van Rossum. It's the most popular language."
   - **Score:** 0.7 ⚠️ PARTIAL (some facts faithful, some added)

### Use Cases

- RAG (Retrieval-Augmented Generation) validation
- Fact-checking systems
- Knowledge base consistency checking
- Hallucination detection in LLM outputs

---

## Key Metrics Comparison

| Metric | Purpose | What It Tests | Score Meaning |
|--------|---------|---------------|---------------|
| **GEval** | Custom criteria | Matches custom evaluation rules | 1.0=Meets, 0.0=Fails |
| **AnswerRelevancy** | Relevance | Is answer relevant to question? | 1.0=Relevant, 0.0=Off-topic |
| **BiasMetric** | Fairness | Any gender/racial/political bias? | 0.0=No bias, 1.0=Strong bias |
| **FaithfulnessMetric** | Consistency | Output faithful to context? | 1.0=Faithful, 0.0=Contradicts |

---

## Quick Examples

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

## Models Used

- **Generation:** llama3.2:3b (fast, lightweight)
- **Evaluation:** deepseek-r1:8b (better reasoning)
- **Premium:** GPT-4 (OpenAI, highest quality)
