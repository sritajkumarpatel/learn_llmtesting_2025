# RUNS WITH HUGGING FACE - Free models, no API keys required

"""
F1 Score Calculation Demonstration with Dummy Data (Hugging Face)
================================================================

What is F1 Score Calculation Demonstration?
- Demonstrates how to calculate F1 score using the Hugging Face evaluate library
- F1 score is the harmonic mean of precision and recall
- Uses dummy predictions and references instead of actual model outputs
- Shows the process of F1 computation with hardcoded data for educational purposes
- Designed to illustrate different F1 score scenarios: perfect, partial, and poor performance

How It Works:
- Uses hardcoded dummy predictions and references to demonstrate F1 score calculation
- Computes F1 score for multiple scenarios: perfect match (1.0), partial match (lower score), poor match (0.0)
- Uses macro averaging for multiclass F1 calculation
- Prints the calculated F1 scores for each scenario

Score Interpretation:
- F1 Score Range: 0.0 to 1.0 (higher is better)
- 0.0 = Worst performance (no correct predictions)
- 1.0 = Perfect performance (all predictions correct)

Threshold: N/A (raw score, no predefined pass/fail criteria)

Use Cases:
- Testing F1 score calculation workflow
- Demonstrating evaluation metrics with dummy data
- Understanding F1 score for classification tasks
- Validating metric implementation

Reference: Hugging Face Evaluate
https://huggingface.co/docs/evaluate/index
"""

from evaluate import load

def test_f1_dummy():
    # Load F1 metric
    f1_metric = load("f1")
    
    # Common references (numeric labels: 0=negative, 1=positive)
    references = [0, 1, 0, 1]
    
    # Perfect match scenario
    predictions_perfect = [0, 1, 0, 1]
    
    # Partial match scenario
    predictions_partial = [0, 1, 1, 0]
    
    # Poor match scenario
    predictions_poor = [1, 1, 1, 1]
    
    # Compute F1 for each scenario
    perfect_result = f1_metric.compute(predictions=predictions_perfect, references=references, average="macro")
    partial_result = f1_metric.compute(predictions=predictions_partial, references=references, average="macro")
    poor_result = f1_metric.compute(predictions=predictions_poor, references=references, average="macro")
    
    print(f"Perfect Match F1 Score: {perfect_result['f1']:.4f}")
    print(f"Partial Match F1 Score: {partial_result['f1']:.4f}")
    print(f"Poor Match F1 Score: {poor_result['f1']:.4f}")
    
    return perfect_result, partial_result, poor_result

if __name__ == "__main__":
    test_f1_dummy()