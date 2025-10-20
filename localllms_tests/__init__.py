"""
Local LLM-based tests for DeepEval metrics.

These tests use local Ollama LLM instances for both response generation and evaluation.
No API keys required - everything runs locally.

Test Files:
- geval_basictest_localllm.py: GEval metric with local LLM as judge
- deepeval_answer_relevancy_localllm.py: AnswerRelevancyMetric with local LLM evaluator
- deepeval_answer_relevancy_multipletestcases.py: Batch evaluation of multiple test cases

Models Used:
- Generation Model: llama3.2:3b (lightweight, fast)
- Evaluation Model: deepseek-r1:8b (more capable, better reasoning)
"""
