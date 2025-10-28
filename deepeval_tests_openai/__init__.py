"""
OpenAI Hybrid Tests for DeepEval Metrics
=========================================

Architecture: Local LLM Generation + OpenAI Evaluation
- Response Generation: Local Ollama (config-defined generation model)
- Evaluation/Judging: OpenAI GPT-4
- Best For: Production-quality evaluation with local efficiency

This is a hybrid approach combining:
✅ Local response generation (fast, no API costs for generation)
✅ OpenAI evaluation (high-quality metrics, best accuracy)

Requirements:
- OPENAI_API_KEY environment variable in .env file
- Ollama running with generation models
- Internet connection for OpenAI API calls

Test Files:
- deepeval_geval.py: Custom GEval criteria with GPT-4 judge
- deepeval_answer_relevancy.py: AnswerRelevancyMetric with GPT-4 evaluator
- deepeval_bias.py: BiasMetric for detecting gender/racial/political bias
- deepeval_faithfulness.py: FaithfulnessMetric for fact-checking consistency

Models Used:
- Generation: Config-defined local model (default: lightweight for generation)
- Evaluation: GPT-4 (OpenAI, high quality)
"""
