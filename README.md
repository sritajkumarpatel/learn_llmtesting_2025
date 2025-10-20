# LLM Testing with DeepEval & Ollama

Testing framework for evaluating Large Language Models (LLMs) using local models and DeepEval metrics.

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
├── utils/                              # Shared utilities
│   └── local_llm_ollama_setup.py
│
├── openapi_tests/                      # OpenAI API tests
│   ├── geval_basictest_openai.py
│   ├── deepeval_answer_relevancy_openai.py
│   └── deepeval_bias_openai.py
│
├── localllms_tests/                    # Local LLM tests
│   ├── geval_basictest_localllm.py
│   ├── deepeval_answer_relevancy_localllm.py
│   └── deepeval_answer_relevancy_multipletestcases.py
│
├── README.md
├── QUICK_REFERENCE.md
└── requirements.txt
```

---

## Test Files Overview

### OpenAI Tests (Requires API Key)

#### 1. **`geval_basictest_openai.py`**
- **Purpose:** Test GEval metric with different thresholds
- **Tests:** 4 tests with thresholds 1.0, 0.8, 0.5, 0.0
- **Expected:** Tests with higher thresholds fail, threshold=0.0 passes
- **Run:** `python -m openapi_tests.geval_basictest_openai`

#### 2. **`deepeval_answer_relevancy_openai.py`**
- **Purpose:** Test if answers are relevant to questions
- **Tests:**
  - France capital → ✅ PASS (direct answer)
  - FIFA 2099 → ✅ PASS (contextually relevant)
  - Pizza to France question → ❌ FAIL (irrelevant)
- **Run:** `python -m openapi_tests.deepeval_answer_relevancy_openai`

#### 3. **`deepeval_bias_openai.py`**
- **Purpose:** Detect gender, racial, political bias in responses
- **Tests:** Describe doctor, nurse, teacher, Indian accent speaker
- **Scoring:** 0 = NO BIAS ✅ | >0.5 = BIAS ❌
- **Run:** `python -m openapi_tests.deepeval_bias_openai`

---

### Local LLM Tests (Free & Offline)

#### 1. **`geval_basictest_localllm.py`**
- **Purpose:** GEval with local Ollama as judge
- **Tests:** Same as OpenAI version but uses local model
- **Run:** `python -m localllms_tests.geval_basictest_localllm`

#### 2. **`deepeval_answer_relevancy_localllm.py`**
- **Purpose:** Answer relevancy with local judge (no API calls)
- **Tests:** Same 3 test cases as OpenAI version
- **Run:** `python -m localllms_tests.deepeval_answer_relevancy_localllm`

#### 3. **`deepeval_answer_relevancy_multipletestcases.py`**
- **Purpose:** Batch evaluation of multiple questions
- **Tests:** Batch 1 (3 questions), Batch 2 (2 questions)
- **Run:** `python -m localllms_tests.deepeval_answer_relevancy_multipletestcases`

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

## Key Metrics Comparison

| Metric | Purpose | What It Tests |
|--------|---------|---------------|
| **GEval** | Custom criteria | Matches custom evaluation rules |
| **AnswerRelevancy** | Relevance | Is answer relevant to question? |
| **BiasMetric** | Fairness | Any gender/racial/political bias? |

---

## Score Reference

### AnswerRelevancyMetric
- **1.0** = Fully relevant ✅
- **0.5** = Partially relevant
- **0.0** = Not relevant ❌

### BiasMetric
- **0.0** = NO BIAS ✅
- **0.5** = Minimal bias
- **1.0** = Strong bias ❌

### GEval
- **1.0** = Meets criteria ✅
- **0.5** = Partial match
- **0.0** = Fails criteria ❌

---

## Quick Examples

### Run OpenAI tests:
```bash
python -m openapi_tests.geval_basictest_openai
python -m openapi_tests.deepeval_bias_openai
```

### Run local tests:
```bash
python -m localllms_tests.geval_basictest_localllm
python -m localllms_tests.deepeval_answer_relevancy_localllm
```

### Import utilities in your code:
```python
from utils import setup_ollama, generate_ollama_response

setup_ollama()
response = generate_ollama_response("What is AI?", model_name="llama3.2:3b")
```

---

## Models Used

- **Generation:** llama3.2:3b (fast, lightweight)
- **Evaluation:** deepseek-r1:8b (better reasoning)
- **Premium:** GPT-4 (OpenAI, highest quality)
