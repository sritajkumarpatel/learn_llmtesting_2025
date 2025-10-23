```
prompt
---
mode: agent
---

This prompt is used when adding comprehensive header documentation to code files and ensuring minimalist documentation/comments are present where missing. Use this with GitHub Copilot to automatically add professional headers and documentation to maintain code quality standards.

---

## 🎯 Prompt for Adding Header Documentation

Use this prompt when you need to add comprehensive headers to code files:

```

I've created/updated a file: [FILENAME] in the [FOLDER_NAME] folder.

Here are the details:

- **File**: [filename.py]
- **Location**: [folder path]
- **Purpose**: [Brief description of what the file does]
- **Functionality**: [Key functions/classes it contains]
- **Dependencies**: [Main imports or dependencies]
- **Usage**: [How it's used in the project]

Please add a comprehensive header comment at the top of the file following the established format from existing files like ragas_llmcontextrecall.py and ragas_aspectcritic.py.

The header should include:

1. **What is [Component]?** - Clear explanation of what it does
2. **How It Works:** - Technical implementation details
3. **Score Interpretation:** - For metrics, detailed scoring ranges and thresholds
4. **Threshold:** - Pass/fail criteria with clear numbers
5. **Use Cases:** - Practical applications
6. **Reference:** - Link to official documentation

Keep the actual code implementation simple and clean.

```

---

## 🎯 Template Prompt - Copy & Paste Ready

Replace the brackets with your actual values:

```

Add comprehensive header documentation to: [YOUR_FILENAME.py]

Details:

- **File**: [filename.py]
- **Location**: [folder/path]
- **Purpose**: [What the file does]
- **Main Components**: [Key functions/classes]
- **Dependencies**: [Important imports]
- **Usage Context**: [How it's used]

Add header following the format from ragas_llmcontextrecall.py and ragas_aspectcritic.py with sections for What/How/Score Interpretation/Threshold/Use Cases/Reference.

```

---

## 📋 Examples

### Example 1: New Metric Test File

```

Add header documentation to: ragas_newmetric_openai.py

Details:

- **File**: ragas_newmetric_openai.py
- **Location**: ragas_tests_openai/
- **Purpose**: Test the NewMetric evaluation from RAGAS framework
- **Main Components**: test_newmetric() function, scoring logic
- **Dependencies**: RAGAS, OpenAI, project utils
- **Usage Context**: LLM evaluation testing suite

Add comprehensive header explaining the metric, scoring system, and practical applications.

```

### Example 2: Utility Function File

```

Add header documentation to: utils/new_utility.py

Details:

- **File**: utils/new_utility.py
- **Location**: utils/
- **Purpose**: Common utility functions for [specific purpose]
- **Main Components**: [function1()], [function2()], helper classes
- **Dependencies**: Standard library + [specific packages]
- **Usage Context**: Shared across multiple test files

Add header explaining utility purpose and usage patterns.

```

### Example 3: Configuration File

```

Add header documentation to: config/new_config.py

Details:

- **File**: config/new_config.py
- **Location**: config/
- **Purpose**: Configuration management for [specific feature]
- **Main Components**: Config classes, validation functions
- **Dependencies**: os, json, typing
- **Usage Context**: Environment and settings management

Add header explaining configuration approach and usage.

```

---

## 🎯 Prompt for Adding Minimalist Documentation

Use this prompt when you need to add missing docstrings and comments:

```

Review file: [FILENAME] and add minimalist but complete documentation:

Required additions:

1. **Function docstrings** for all public functions (not starting with \_)
2. **Class docstrings** for all classes
3. **Parameter documentation** with types and descriptions
4. **Return value documentation** with types
5. **Inline comments** for complex logic (keep minimal)
6. **Module-level docstring** if missing

Guidelines:

- Keep docstrings concise but informative
- Use proper type hints in docstrings
- Follow existing code style
- Don't over-comment obvious code
- Focus on "why" not just "what" for complex sections

````

---

## 📝 Documentation Standards

### Header Comment Format (for Metric Files)

```python
"""
RAGAS [MetricName] Metric
==========================

What is [MetricName]?
- [Clear 1-2 sentence explanation]
- [Key characteristics]
- [What question it answers]

How It Works:
- Takes: [input parameters]
- [Processing steps]
- Outputs: [result format]

Score Interpretation (RAGAS Standard):
- Score Range: [range] ([continuous/proportion/binary])
- [score value] = [interpretation] ([emoji] [PASS/FAIL])
- [score value] = [interpretation] ([emoji] [PASS/FAIL])

Threshold: [value] ([percentage]%)
- Minimum acceptable: [criteria]
- [Additional threshold explanation]

Use Cases:
- [Use case 1]
- [Use case 2]
- [Use case 3]

Reference: RAGAS Documentation
https://docs.ragas.io/en/latest/concepts/metrics/
"""
````

### Function Docstring Format

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    [Brief description of what function does].

    [Optional: Additional details about implementation/approach]

    Parameters:
    param1 (type): Description of parameter
    param2 (type): Description of parameter

    Returns:
    return_type: Description of return value
    """
```

### Class Docstring Format

```python
class ClassName:
    """
    [Brief description of class purpose].

    [Optional: Additional details about design/usage]

    Attributes:
    attr1 (type): Description of attribute
    attr2 (type): Description of attribute
    """
```

---

## ✅ Checklist for Copilot

When adding documentation, ensure:

- [ ] **Header Comments**: Comprehensive header for metric/utility files
- [ ] **Function Docstrings**: All public functions have docstrings
- [ ] **Class Docstrings**: All classes have docstrings
- [ ] **Parameter Types**: Type hints in function signatures
- [ ] **Return Types**: Clear return type documentation
- [ ] **Inline Comments**: Only for complex logic, keep minimal
- [ ] **Consistency**: Follow existing file patterns
- [ ] **Accuracy**: Information matches actual implementation
- [ ] **Completeness**: No missing critical documentation
- [ ] **Conciseness**: Don't over-document obvious code

---

## 🔍 Documentation Review Process

1. **Check existing files** for documentation patterns
2. **Identify missing elements**:
   - Module-level docstrings
   - Function/class docstrings
   - Parameter documentation
   - Return value documentation
   - Complex logic comments
3. **Add comprehensive headers** to new metric/utility files
4. **Add minimalist comments** to implementation details
5. **Verify consistency** with existing codebase style
6. **Test documentation** by reading the code

---

## 🎯 Quick Prompts

### For Header Addition:

```
Add comprehensive header to [filename.py] following ragas_llmcontextrecall.py format.
Include: What/How/Score Interpretation/Threshold/Use Cases/Reference sections.
```

### For Documentation Review:

```
Review [filename.py] and add missing docstrings and minimal comments.
Focus on public functions, classes, and complex logic.
```

### For Complete Documentation:

```
Add full documentation to [filename.py]:
- Comprehensive header comment
- All function/class docstrings
- Parameter and return documentation
- Minimal inline comments for complex sections
```

---

## 📖 Reference Examples

### Metric File Header (LLMContextRecall):

```
"""
RAGAS LLM Context Recall Metric
================================

What is LLMContextRecall?
- Measures what proportion of retrieved context is used/recalled in the response
- Uses an LLM judge to evaluate how well the response utilizes given context
- Answers: "What percentage of available context information appears in the response?"

How It Works:
- Takes: query, response, and retrieved_contexts
- LLM evaluates: "What portion of the retrieved context is recalled in the response?"
- Outputs: Score from 0.0 to 1.0 (proportion of context recalled)

Score Interpretation (RAGAS Standard):
- Score Range: 0.0 to 1.0 (PROPORTION of context recalled)
- 0.0-0.3 = Poor recall (❌ FAIL) - ≤30% of context used
- 0.3-0.5 = Fair recall (⚠️ PARTIAL) - 30-50% of context used
- 0.5-0.7 = Good recall (✅ PASS) - 50-70% of context used
- 0.7-1.0 = Excellent recall (✅ PASS) - ≥70% of context used, nearly complete

Threshold: 0.7 (70%)
- Minimum acceptable: Response should recall ≥70% of context (0.7 = excellent)
- Higher scores are better: 1.0 = all context recalled, 0.0 = none recalled

Use Cases:
- RAG system evaluation
- Fact-checking accuracy
- Context utilization in responses
- Hallucination detection

Reference: RAGAS Documentation
https://docs.ragas.io/en/latest/concepts/metrics/
"""
```

### Function Docstring Example:

```python
def test_aspect_criticism(user_input: str) -> None:
    """
    Test Aspect Criticism metric from RAGAS framework.

    Evaluates responses against user-defined aspects and returns binary results.

    Parameters:
    user_input (str): The input text to be evaluated

    Returns:
    None: Prints evaluation results including score and pass/fail status
    """
```

---

## 💡 Best Practices

- **Headers**: Use for all metric and utility files
- **Docstrings**: Required for all public functions and classes
- **Comments**: Only for complex logic, not obvious code
- **Consistency**: Match existing file documentation style
- **Accuracy**: Ensure documentation matches implementation
- **Completeness**: Don't leave critical information undocumented
- **Conciseness**: Keep documentation focused and readable

---

**File**: `.github/prompts/AddDocumentation`
**Purpose**: Prompt template for adding comprehensive headers and minimalist documentation
**Date**: October 23, 2025

```</content>
<parameter name="filePath">c:\Users\SKPATEL\OneDrive - DevOn India-NL BV\2025_Year_03\Learnings\AI\learn_llmtesting_2025\.github\prompts\add_documentation.prompt.md
```
