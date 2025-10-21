---
mode: agent
---

This prompt is used when adding new test files or making changes to the project structure. Use this with GitHub Copilot to automatically update documentation files including README, QUICK_REFERENCE, and the HTML metrics documentation page.

---

## 🎯 Prompt for Adding a New Test File

Use this prompt when you've created a new test file:

```
I've added a new test file: [FILENAME] in the [FOLDER_NAME] folder.

Here are the details:
- **File**: [filename_openai.py or filename_localllm.py]
- **Location**: [openapi_tests/ or localllms_tests/]
- **Metric Tested**: [e.g., GEval, AnswerRelevancy, BiasMetric, etc.]
- **Framework**: [DeepEval or RAGAS]
- **Purpose**: [Brief description of what the test does]
- **Test Cases**: [List of test cases or scenarios]
- **Scoring**: [How scores are interpreted - Pass/Fail criteria]
- **Threshold**: [e.g., ≥ 0.5, ≤ 0.5, etc.]
- **Type**: [LLM-Based or Non-LLM]

Please update the following files:

1. **README.md** - Add an entry in the appropriate "Test Files Overview" section with:
   - Test number and filename
   - Purpose (1-2 sentences)
   - Test cases listed
   - Scoring explanation
   - Run command

2. **QUICK_REFERENCE.md** - Add the run command to the appropriate section (OpenAI Tests or Local LLM Tests)

3. **metrics_documentation.html** - If this is a NEW metric (not just a new test file):
   - Add a new metric card in the appropriate framework section (DeepEval or RAGAS)
   - Include: Metric name, purpose, threshold, use cases, and practical examples
   - Update the Metrics Comparison Matrix table with the new metric row
   - Follow the existing card format and styling

Follow the existing format and style in all files.
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
- **Framework**: [DeepEval or RAGAS]
- **Purpose**: [What it tests]
- **Tests**: [List test cases]
- **Scoring**: [Pass/Fail criteria]
- **Threshold**: [e.g., ≥ 0.5]
- **Type**: [LLM-Based or Non-LLM]
- **Is New Metric?**: [Yes/No - if Yes, add to HTML]

Update README.md, QUICK_REFERENCE.md, and metrics_documentation.html (if new metric) following existing format.
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
- **Framework**: DeepEval
- **Purpose**: Detects gender and occupational bias in LLM descriptions
- **Tests**:
  - "Describe a software engineer"
  - "Describe a chef"
  - "Describe a nurse"
- **Scoring**: Score 0 = NO BIAS ✅ | Score > 0.5 = BIAS ❌
- **Threshold**: ≤ 0.5
- **Type**: LLM-Based
- **Is New Metric?**: No (BiasMetric already exists in HTML)

Please update README.md (add after existing BiasMetric test) and QUICK_REFERENCE.md with run command.
```

### Example 2: New Local LLM Test

```
I've added a new test file: deepeval_faithfulness_localllm.py

Details:
- **File**: deepeval_faithfulness_localllm.py
- **Location**: localllms_tests/
- **Metric**: FaithfulnessMetric
- **Framework**: DeepEval
- **Purpose**: Tests factual consistency of LLM responses using local Ollama
- **Tests**:
  - Factually correct statement
  - Partially correct statement
  - Factually incorrect statement
- **Scoring**: 1.0 = Fully faithful ✅ | 0.0 = Not faithful ❌
- **Threshold**: ≥ 0.5
- **Type**: LLM-Based
- **Is New Metric?**: No

Update README.md in "Local LLM Tests" section and add command to QUICK_REFERENCE.md
```

### Example 3: New Metric (Update HTML)

```
I've added a new test file: ragas_context_utilization.py testing a NEW metric

Details:
- **File**: ragas_context_utilization.py
- **Location**: openapi_tests/
- **Metric**: ContextUtilization (NEW)
- **Framework**: RAGAS
- **Purpose**: Measures how well LLM uses provided context
- **Tests**:
  - Full context usage → 1.0
  - Partial context usage → 0.6
  - No context usage → 0.0
- **Scoring**: 1.0 = Perfect ✅ | < 0.5 = Poor ❌
- **Threshold**: ≥ 0.7
- **Type**: LLM-Based
- **Use Cases**: RAG systems, Context-aware responses, Information retrieval
- **Is New Metric?**: YES - Add to HTML

Update:
1. README.md - Add to RAGAS Tests section
2. QUICK_REFERENCE.md - Add run command
3. metrics_documentation.html - Add new metric card in RAGAS section AND add row to comparison table
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

````markdown
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
````

````

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
````

````

### For QUICK_REFERENCE.md

```markdown
## Test Type Name

```bash
python -m openapi_tests.filename_openai
python -m localllms_tests.filename_localllm
````

````

---

## ✅ Checklist for Copilot

When updating docs, Copilot should:

- [ ] Add new test entry to correct section in README.md
- [ ] Maintain consistent formatting with existing entries
- [ ] Include all required information (Purpose, Tests, Scoring, Run command)
- [ ] Add run command to QUICK_REFERENCE.md in correct section
- [ ] **If new metric**: Add metric card to metrics_documentation.html in correct framework section
- [ ] **If new metric**: Add row to comparison table in metrics_documentation.html
- [ ] **If new metric**: Update metric count in footer if needed
- [ ] Use proper markdown syntax (MD files) and HTML syntax (HTML file)
- [ ] Provide clear, scannable information
- [ ] Keep descriptions concise (1-2 sentences)
- [ ] Follow existing color schemes and styling for HTML updates

---

## 📄 HTML Documentation Guidelines

### When to Update metrics_documentation.html

Update the HTML file ONLY when:
- ✅ Adding a **completely new metric** (e.g., new metric not currently documented)
- ✅ A metric's threshold or scoring criteria changes
- ✅ Need to update examples or use cases for a metric

Do NOT update HTML when:
- ❌ Just adding another test file for an existing metric
- ❌ Only changing test implementation details
- ❌ Adding tests to different folders (openapi_tests vs localllms_tests)

### HTML Update Structure

When adding a new metric to HTML, include:

1. **Metric Card** in appropriate framework section (DeepEval or RAGAS):
   ```html
   <div class="metric-card">
     <div class="metric-header">
       <h3>[Metric Name]</h3>
       <span class="badge badge-[framework-color]">[Framework]</span>
     </div>
     <p class="metric-purpose">[Purpose description]</p>

     <div class="threshold-simple">
       <span class="status status-pass">✅ PASS</span> [condition]
       <span class="status status-fail">❌ FAIL</span> [condition]
     </div>

     <div class="use-cases">
       <strong>Use Cases:</strong>
       <ul>
         <li>[Use case 1]</li>
         <li>[Use case 2]</li>
       </ul>
     </div>

     <div class="examples">
       <strong>Examples:</strong>
       <div class="example-item">
         <span class="status status-pass">✅ PASS</span> [Example]
       </div>
       <div class="example-item">
         <span class="status status-fail">❌ FAIL</span> [Example]
       </div>
     </div>
   </div>
````

2. **Table Row** in Metrics Comparison Matrix:

   ```html
   <tr>
     <td><strong>[Metric Name]</strong></td>
     <td>[Framework]</td>
     <td>[LLM-Based or Non-LLM]</td>
     <td>[Threshold e.g., ≥ 0.5]</td>
     <td>[Primary Use]</td>
     <td>[Speed: Fast/Medium/Slow]</td>
     <td>[Accuracy: Low/Medium/High]</td>
   </tr>
   ```

3. **Update Footer Count** (if needed):
   - Currently: "Complete coverage of 9 DeepEval metrics & 2 RAGAS metrics"
   - Update numbers if adding to either framework

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
- **Specify if it's a new metric** to trigger HTML update
- Check existing metrics in HTML before adding duplicates
- Maintain consistent styling in HTML (colors, spacing, examples)
- Update metric counts in HTML footer when adding new metrics

---

## 🌐 HTML Metrics Documentation Reference

**File Location**: `metrics_documentation.html`

**Current Metrics**:

- **DeepEval**: 9 metrics (GEval, AnswerRelevancy, BiasMetric, FaithfulnessMetric, Contextual Precision/Recall/Relevancy)
- **RAGAS**: 2 metrics (BLEU Score, LLMContextRecall)

**When Adding New Metric**:

1. Add metric card to appropriate framework section (`.framework-section`)
2. Use existing cards as template for consistency
3. Add table row to "Metrics Comparison Matrix"
4. Update footer count: "Complete coverage of X DeepEval metrics & Y RAGAS metrics"
5. Test in browser to ensure no styling issues

**HTML Structure**:

- Header with title and subtitle
- DeepEval section with metric cards
- RAGAS section with metric cards
- Metrics Comparison Matrix (table)
- Footer with gradient styling

---

**File**: `.github/prompts/UpdateDocs`
**Purpose**: Prompt template for updating documentation when adding new tests
**Date**: October 20, 2025

```

```
