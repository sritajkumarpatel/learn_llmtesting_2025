from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import ollama
import os
from dotenv import load_dotenv
from local_llm_ollama_setup import setup_ollama, generate_ollama_response, setup_custom_ollama_model_for_evaluation


# TODO: This file needs to be tested to ensure all test cases work as expected
def test_correctness(evaluationModel,threshold=1.0):

    # Generate response using local Ollama LLM
    response = generate_ollama_response('What is the capital of India?')

    correctness_metric = GEval(
        name="test_correctness",
        criteria="Determine if the actual output matches the expected output",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=threshold,
        model=evaluationModel
    )
    test_case = LLMTestCase(
        input="What is the capital of India?",
        actual_output=response,
        expected_output="The capital of India is New York.",
    )
    print("Actual Output:", response)
    
    try:
        assert_test(test_case, [correctness_metric])
        print(f"✅ Test Passed! (Score: {correctness_metric.score} >= Threshold: {correctness_metric.threshold})")
    except AssertionError as e:
        print(f"❌ Test Failed! (Score: {correctness_metric.score} < Threshold: {correctness_metric.threshold})")
        print(f"   Reason: {correctness_metric.reason}")

if __name__ == "__main__":

    # Check if Ollama is running and start if needed
    setup_ollama()
    # Set local LLM as evaluation judge model
    evaluationModel = setup_custom_ollama_model_for_evaluation()


    # Test 1: Threshold 1.0 - Will FAIL
    print("Test 1 (threshold=1.0) - Expected: ❌ FAIL")
    test_correctness(evaluationModel)
    
    # # Test 2: Threshold 0.8 - Will FAIL
    # print("\nTest 2 (threshold=0.8) - Expected: ❌ FAIL")
    # test_correctness(evaluationModel='llama3.2:3b', threshold=0.8)
    
    # # Test 3: Threshold 0.5 - Will FAIL
    # print("\nTest 3 (threshold=0.5) - Expected: ❌ FAIL")
    # test_correctness(evaluationModel='llama3.2:3b', threshold=0.5)