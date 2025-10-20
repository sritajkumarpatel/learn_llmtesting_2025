"""
OpenAI API-based tests for DeepEval metrics.

These tests use OpenAI's GPT models for evaluation.
Requires OPENAI_API_KEY to be set in .env file.

Test Files:
- geval_basictest_openai.py: GEval metric with different thresholds
- deepeval_answer_relevancy_openai.py: AnswerRelevancyMetric with OpenAI evaluator
- deepeval_bias_openai.py: BiasMetric to detect gender/racial/political bias
"""
