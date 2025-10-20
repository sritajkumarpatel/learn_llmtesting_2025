# LLM Testing with DeepEval & Ollama

Testing framework for evaluating Large Language Models (LLMs) using local models and DeepEval metrics.

## Setup

1. **Create and activate Python virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   
   # Activate on Windows (Command Prompt)
   .\venv\Scripts\activate.bat
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file** in project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Ensure Ollama is running** (for local LLM inference)

## Project Structure

### Core Files

- **`requirements.txt`** - Python dependencies (deepeval, ollama, openai, etc.)
- **`.env`** - Environment variables (API keys) - **Keep secret, don't commit**
- **`.gitignore`** - Git ignore rules (excludes `.env`, `venv/`, etc.)

### Test Files

- **`geval_basictest_test01.py`** - Tests GEval metric with different thresholds
  - Shows how threshold affects pass/fail
  - Scores actual output against expected output

- **`deepeval_answer_relevancy_test01.py`** - Tests AnswerRelevancyMetric
  - Tests 1-2: Relevant answers (Score: 1.0) ✅
  - Test 3: Irrelevant answer (Score: 0.0) ❌

## Running Tests

```bash
# Test GEval correctness metric
python geval_basictest_test01.py

# Test Answer Relevancy metric
python deepeval_answer_relevancy_test01.py
```

## Key Metrics

| Metric | Purpose | Score Range |
|--------|---------|-------------|
| **GEval** | Custom evaluation criteria using LLM | 0.0 - 1.0 |
| **AnswerRelevancyMetric** | Checks if answer addresses query | 0.0 - 1.0 |

## Quick Reference

- **Score 1.0** = Perfect match/fully relevant
- **Score 0.0** = No match/not relevant
- **Threshold** = Minimum score required to pass test
