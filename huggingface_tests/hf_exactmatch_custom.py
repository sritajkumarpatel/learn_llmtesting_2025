"""
Exact Match Calculation Demonstration with Dummy Data (Hugging Face)
===================================================================

What is Exact Match Calculation Demonstration?
- Demonstrates how to calculate exact match using the Hugging Face evaluate library
- Uses dummy predictions and references instead of actual model outputs
- Shows the process of exact match computation with hardcoded data for educational purposes
- Designed to illustrate different exact match scenarios: perfect, partial, and no matches

How It Works:
- Uses hardcoded dummy predictions and references to demonstrate exact match calculation
- Computes exact match for multiple scenarios: perfect match (1.0), partial match (0.5), no match (0.0)
- Prints the calculated exact match scores for each scenario

Score Interpretation:
- Exact Match Range: 0.0 to 1.0 (higher is better)
- 0.0 = No predictions match exactly
- 1.0 = All predictions match exactly

Threshold: N/A (raw score, no predefined pass/fail criteria)

Use Cases:
- Testing exact match calculation workflow
- Demonstrating evaluation metrics with dummy data
- Validating metric implementation for different scenarios

Reference: Hugging Face Evaluate
https://huggingface.co/docs/evaluate/index
"""

from evaluate import load

def test_exactmatch_dummy():
    # Load exact match metric
    exact_match_metric = load("exact_match")
    
    # Common references
    references = ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"]
    
    # Perfect match scenario
    predictions_perfect = ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"]
    
    # Partial match scenario (50% correct)
    predictions_partial = ["POSITIVE", "POSITIVE", "POSITIVE", "NEGATIVE"]
    
    # No match scenario
    predictions_none = ["NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]
    
    # Compute exact match for each scenario
    perfect_result = exact_match_metric.compute(predictions=predictions_perfect, references=references)
    partial_result = exact_match_metric.compute(predictions=predictions_partial, references=references)
    none_result = exact_match_metric.compute(predictions=predictions_none, references=references)
    
    print(f"Perfect Match Score: {perfect_result['exact_match']:.4f}")
    print(f"Partial Match Score: {partial_result['exact_match']:.4f}")
    print(f"No Match Score: {none_result['exact_match']:.4f}")
    
    return perfect_result, partial_result, none_result

if __name__ == "__main__":
    test_exactmatch_dummy()