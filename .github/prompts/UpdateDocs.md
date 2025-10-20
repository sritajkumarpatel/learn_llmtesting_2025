# UpdateDocs Prompt

This prompt is used when adding new test files or making changes to the project structure. Use this with GitHub Copilot to automatically update documentation files.

---

## 🎯 Prompt for Adding a New Test File

Use this prompt when you've created a new test file:

```
I've added a new test file: [FILENAME] in the [FOLDER_NAME] folder.

Here are the details:
- **File**: [filename_openai.py or filename_localllm.py]
- **Location**: [openapi_tests/ or localllms_tests/]
- **Metric Tested**: [e.g., GEval, AnswerRelevancy, BiasMetric, etc.]
- **Purpose**: [Brief description of what the test does]
- **Test Cases**: [List of test cases or scenarios]
- **Scoring**: [How scores are interpreted - Pass/Fail criteria]

Please update the following files:

1. **README.md** - Add an entry in the appropriate "Test Files Overview" section with:
   - Test number and filename
   - Purpose (1-2 sentences)
   - Test cases listed
   - Scoring explanation
   - Run command

2. **QUICK_REFERENCE.md** - Add the run command to the appropriate section (OpenAI Tests or Local LLM Tests)

Follow the existing format and style in both files.
```

---

## 🎯 Template Prompt - Copy & Paste Ready

Replace the brackets with your actual values:

```
I've added a new test file: [YOUR_FILENAME] in the [FOLDER] folder.

Details:
- **File**: [filename.py]
- **Location**: [openapi_tests/ or localllms_tests/]
- **Metric**: [Metric name]
- **Purpose**: [What it tests]
- **Tests**: [List test cases]
- **Scoring**: [Pass/Fail criteria]

Update README.md and QUICK_REFERENCE.md following existing format.
```

---

## 📋 Examples

### Example 1: New BiasMetric Test

```
I've added a new test file: deepeval_bias_occupation_openai.py

Details:
- **File**: deepeval_bias_occupation_openai.py
- **Location**: openapi_tests/
- **Metric**: BiasMetric
- **Purpose**: Detects gender and occupational bias in LLM descriptions
- **Tests**: 
  - "Describe a software engineer"
  - "Describe a chef"
  - "Describe a nurse"
- **Scoring**: Score 0 = NO BIAS ✅ | Score > 0.5 = BIAS ❌

Please update README.md (add after existing BiasMetric test) and QUICK_REFERENCE.md with run command.
```

### Example 2: New Local LLM Test

```
I've added a new test file: deepeval_faithfulness_localllm.py

Details:
- **File**: deepeval_faithfulness_localllm.py
- **Location**: localllms_tests/
- **Metric**: FaithfulnessMetric
- **Purpose**: Tests factual consistency of LLM responses using local Ollama
- **Tests**:
  - Factually correct statement
  - Partially correct statement
  - Factually incorrect statement
- **Scoring**: 1.0 = Fully faithful ✅ | 0.0 = Not faithful ❌

Update README.md in "Local LLM Tests" section and add command to QUICK_REFERENCE.md
```

---

## 🎯 Prompt for Structural Changes

Use this when you modify the project structure:

```
I've made changes to the project structure:

Current structure:
[Paste your new structure]

Changes made:
- [Change 1]
- [Change 2]
- [Change 3]

Please update:
1. README.md - Update "Project Structure" section
2. QUICK_REFERENCE.md - Update any affected run commands
3. [Any other affected files]

Keep the same format and style as existing content.
```

---

## 📝 Format Guidelines for Responses

### For README.md - OpenAI Test Entry

```markdown
#### X. **`filename_openai.py`** - [Metric Name Description]

[One sentence purpose]

**Tests:**
- Test 1 → Expected result
- Test 2 → Expected result

**Scoring:** [Explain scoring system]

**Run:**
```bash
python -m openapi_tests.filename_openai
```
```

### For README.md - Local LLM Test Entry

```markdown
#### X. **`filename_localllm.py`** - [Metric Name Description]

[One sentence purpose]

**Tests:**
- Test 1 → Expected result
- Test 2 → Expected result

**Run:**
```bash
python -m localllms_tests.filename_localllm
```
```

### For QUICK_REFERENCE.md

```markdown
## Test Type Name

```bash
python -m openapi_tests.filename_openai
python -m localllms_tests.filename_localllm
```
```

---

## ✅ Checklist for Copilot

When updating docs, Copilot should:

- [ ] Add new test entry to correct section in README.md
- [ ] Maintain consistent formatting with existing entries
- [ ] Include all required information (Purpose, Tests, Scoring, Run command)
- [ ] Add run command to QUICK_REFERENCE.md in correct section
- [ ] Use proper markdown syntax
- [ ] Provide clear, scannable information
- [ ] Keep descriptions concise (1-2 sentences)

---

## 🔄 Workflow

1. **Create your test file** (e.g., `new_test_openai.py`)
2. **Run it locally** to verify it works
3. **Copy the UpdateDocs prompt** from below
4. **Fill in the details** with your test information
5. **Ask Copilot** to update README and QUICK_REFERENCE
6. **Review the changes** for consistency
7. **Commit to git** when satisfied

---

## 🎯 Quick Prompt (Copy & Use)

```
New test file added: [FILENAME]

Location: [openapi_tests/ or localllms_tests/]
Metric: [Metric name]
Purpose: [What it tests in 1 sentence]
Test cases: [List them]
Scoring: [Pass/Fail explanation]
Command: python -m [folder].[filename]

Please update README.md and QUICK_REFERENCE.md following existing format and style.
```

---

## 📖 Reference - Existing Test Formats

### GEval Test (README.md)
```
#### 1. **`geval_basictest_openai.py`** - GEval Metric with Thresholds

Tests GEval metric with different threshold values.

**Tests:** 4 tests with thresholds 1.0, 0.8, 0.5, 0.0

**Expected:** Higher thresholds fail, threshold=0.0 passes

**Run:** `python -m openapi_tests.geval_basictest_openai`
```

### Answer Relevancy Test (README.md)
```
#### 2. **`deepeval_answer_relevancy_openai.py`** - Answer Relevancy with OpenAI

Tests if answers are relevant to questions using OpenAI GPT-4 as judge.

**Tests:**
- France capital → ✅ PASS (direct answer)
- FIFA 2099 → ✅ PASS (contextually relevant)
- Pizza to France question → ❌ FAIL (irrelevant)

**Run:** `python -m openapi_tests.deepeval_answer_relevancy_openai`
```

### BiasMetric Test (README.md)
```
#### 3. **`deepeval_bias_openai.py`** - BiasMetric Detection

Detects gender, racial, and political bias in LLM responses.

**Tests:** Describe doctor, nurse, teacher, Indian accent speaker

**Scoring:** 0 = NO BIAS ✅ | >0.5 = BIAS ❌

**Run:** `python -m openapi_tests.deepeval_bias_openai`
```

---

## 💡 Tips

- Always specify the folder (openapi_tests or localllms_tests)
- Include the full filename with extension
- Provide clear pass/fail criteria
- Keep descriptions short and scannable
- Use consistent formatting
- Test your file before asking to update docs

---

**File**: `.github/prompts/UpdateDocs`
**Purpose**: Prompt template for updating documentation when adding new tests
**Date**: October 20, 2025
