"""
Local-Only Tests for DeepEval Metrics
======================================

Architecture: Completely Local LLM (Ollama)
- Response Generation: Local Ollama LLM
- Evaluation/Judging: Local Ollama LLM
- Best For: Offline testing, cost-free evaluation, privacy-critical work

This is a fully local approach:
✅ No API keys required
✅ No internet dependency
✅ No API costs
✅ Complete data privacy
✅ Runs entirely on your machine

Requirements:
- Ollama running locally
- Models: llama3.2:3b (generation) and deepseek-r1:8b (evaluation)
- No internet connection needed
- No external API keys needed

Test Files:
- deepeval_geval.py: Custom GEval criteria with local LLM judge
- deepeval_answer_relevancy.py: AnswerRelevancyMetric with local evaluator
- deepeval_answer_relevancy_multipletestcases.py: Batch evaluation of multiple questions
- deepeval_rag.py: RAG evaluation with vector database and contextual metrics
- deepeval_rag_localllm.py: RAG with alternative retrieval approach

Models Used:
- Generation Model: llama3.2:3b (lightweight, fast)
- Evaluation Model: deepseek-r1:8b (better reasoning for judging)
- Both run locally via Ollama
"""
