# RUNS WITH OPENAI - Requires OpenAI API key for GPT-4 evaluation

"""
DeepEval Prompt Engineering Effectiveness Testing
=================================================

What is Prompt Engineering Effectiveness Testing?
- Evaluates how well different prompt templates control LLM output format and behavior
- Tests prompt constraints using custom GEval criteria for format adherence
- Demonstrates prompt engineering validation through structured evaluation
- Answers: "How effectively do prompts control LLM response characteristics?"

How It Works:
- Takes: user input, prompt template, and evaluation criteria
- Generates responses using LangChain ChatOpenAI with DeepEval prompt templates
- Evaluates responses against custom GEval criteria for format compliance
- Outputs: Response content and evaluation scores for each prompt type

Score Interpretation (DeepEval GEval Standard):
- Score Range: 0.0 to 1.0 (PROPORTION of criteria met)
- 0.0-0.3 = Poor compliance (❌ FAIL) - ≤30% of criteria met
- 0.3-0.5 = Fair compliance (⚠️ PARTIAL) - 30-50% of criteria met
- 0.5-0.7 = Good compliance (✅ PASS) - 50-70% of criteria met
- 0.7-1.0 = Excellent compliance (✅ PASS) - ≥70% of criteria met, nearly perfect

Threshold: 0.5 (50%)
- Minimum acceptable: Response should meet ≥50% of evaluation criteria (0.5 = good)
- Higher scores are better: 1.0 = perfect criteria compliance, 0.0 = no compliance

Use Cases:
- Prompt engineering validation and testing
- Format control effectiveness measurement
- LLM behavior constraint evaluation
- Custom criteria assessment for specialized tasks
- Prompt template optimization and comparison

Reference: DeepEval Documentation
https://docs.deepevalai.com/docs/metrics/GEval/
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from deepeval.prompt import Prompt, PromptMessage
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

# Set the API key in environment (required for LangChain)
os.environ["OPENAI_API_KEY"] = openai_api_key

# Create DeepEval prompt template
prompt = Prompt(
    alias="One Word Assistant Prompt",
    messages_template=[
        PromptMessage(role="system", content="You are a helpful assistant. Answer questions with exactly one word")
    ]
)

prompt2 = Prompt(
    alias="Greetings Assistant Prompt",
    messages_template=[
        PromptMessage(role="system", content="You are a helpful assistant. Answer questions with adding greetings at the end")
    ]
)

prompt3 = Prompt(
    alias="Poem Assistant Prompt",
    messages_template=[
        PromptMessage(role="system", content="You are a helpful assistant. Answer questions in a poem form")
    ]
)

def generate_response_with_prompt_and_evaluate(user_input: str, prompt: Prompt):
    """
    Generate a response using LangChain ChatOpenAI with DeepEval prompt template and evaluate it.

    Uses the specified prompt template to generate a response, then evaluates the response
    against custom GEval criteria based on the prompt type (one word, greetings, poem).

    Parameters:
    user_input (str): The user's question or input text to be processed
    prompt (Prompt): The DeepEval prompt template to use for response generation

    Returns:
    tuple: A tuple containing:
        - response_content (str): The generated response text
        - evaluation_result: GEval evaluation result object with score and reasoning
    """
    # Initialize LangChain ChatOpenAI
    openai_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # Convert DeepEval prompt messages to LangChain format
    messages = []
    for msg in prompt.messages_template:
        if msg.role == "system":
            messages.append(SystemMessage(content=msg.content))
        elif msg.role == "user":
            messages.append(HumanMessage(content=msg.content))

    # Add the user input
    messages.append(HumanMessage(content=user_input))

    # Generate response
    response = openai_model.invoke(messages)

    # Create test case for evaluation
    test_case = LLMTestCase(
        input=user_input,
        actual_output=response.content
    )

    print("=" * 80)
    print("User Input:", user_input)
    print("Generated Response:", response.content)

    # Evaluate based on prompt type - each prompt has different format requirements
    if prompt.alias == "One Word Assistant Prompt":
        # Test: Response should be exactly one word and relevant to query
        one_word_metric = GEval(
            name="One Word Response",
            criteria="Assess if the response is a single word and relevant to the query.",
            evaluation_steps=[
                "Check if the response contains only one word.",
                "Evaluate if that word appropriately answers the user's question."
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
            model="gpt-4o-mini"
        )
        evaluation_result = one_word_metric.measure(test_case)
        print(f"One Word Score: {evaluation_result}")
    
    elif prompt.alias == "Greetings Assistant Prompt":
        # Test: Response should answer question and end with a greeting
        greetings_metric = GEval(
            name="Greetings Response",
            criteria="Assess if the response includes a greeting at the end and answers the question.",
            evaluation_steps=[
                "Check if the response ends with a greeting phrase.",
                "Evaluate if the response appropriately answers the user's question."
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
            model="gpt-4o-mini"
        )
        evaluation_result = greetings_metric.measure(test_case)
        print(f"Greetings Score: {evaluation_result}")
     
    elif prompt.alias == "Poem Assistant Prompt":
        # Test: Response should be structured as a poem and answer the question
        poem_metric = GEval(
            name="Poem Response",
            criteria="Assess if the response is in poem form and answers the question.",
            evaluation_steps=[
                "Check if the response is structured as a poem.",
                "Evaluate if the poem appropriately answers the user's question."
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
            model="gpt-4o-mini"
        )
        evaluation_result = poem_metric.measure(test_case)
        print(f"Poem Score: {evaluation_result}")

def evaluate_prompt_effectiveness(user_input: str, prompt_type: str):
    """
    Evaluate prompt effectiveness for a specific prompt type.

    Parameters:
    user_input (str): The user's question or input text
    prompt_type (str): Type of prompt to use ("one_word", "greetings", "poem")

    Returns:
    tuple: (response_content, evaluation_result)
    """
    # Select the appropriate prompt based on type
    if prompt_type == "one_word":
        selected_prompt = prompt
    elif prompt_type == "greetings":
        selected_prompt = prompt2
    elif prompt_type == "poem":
        selected_prompt = prompt3
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    # Generate response and evaluate
    return generate_response_with_prompt_and_evaluate(user_input, selected_prompt)


if __name__ == "__main__":
    """
    Main execution block for testing prompt engineering effectiveness.

    Runs comprehensive tests of three different prompt templates:
    - One Word Prompt: Tests format constraint to single-word responses
    - Greetings Prompt: Tests addition of greeting phrases
    - Poem Prompt: Tests creative poem format responses

    Includes both positive test cases (designed to pass) and negative test cases
    (designed to fail) to validate the evaluation criteria.

    Prints detailed results for each test case including response content and evaluation scores.
    """
    # Test cases designed to demonstrate prompt effectiveness
    # Positive cases: Should pass evaluation criteria
    input_query_for_one_word = "Tom & Jerry used to fight in a game of baseball, Tom won 10 rounds and Jerry won 7 rounds. How many rounds were played in total?"
    input_query_for_greetings = "What is the capital of France?"
    input_query_for_poem = "Can you describe the ocean?"

    # Negative cases: Designed to fail evaluation criteria
    input_query_for_one_word_failure = "Tom & Jerry used to fight in a game of baseball, Tom won many rounds as big as universe and jerry won as low as petals of flowers. How many rounds were played in total?"
    input_query_for_greetings_failure = "Answer capital of france with only one word?"
    input_query_for_poem_failure = "Can you describe the ocean in one word?"

    print("=" * 80)
    print("DEEPEVAL PROMPT TESTING WITH LANGCHAIN OPENAI")
    print("=" * 80)

    # Evaluate positive test cases - these should pass the criteria
    print("\n=== Testing Positive Cases (Should Pass) ===")
    evaluate_prompt_effectiveness(input_query_for_one_word, "one_word")
    evaluate_prompt_effectiveness(input_query_for_greetings, "greetings")
    evaluate_prompt_effectiveness(input_query_for_poem, "poem")

    # Evaluate negative test cases - these are designed to fail the criteria
    print("\n=== Testing Negative Cases (Should Fail) ===")
    evaluate_prompt_effectiveness(input_query_for_one_word_failure, "one_word")
    evaluate_prompt_effectiveness(input_query_for_greetings_failure, "greetings")
    evaluate_prompt_effectiveness(input_query_for_poem_failure, "poem")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)