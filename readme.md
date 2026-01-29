# A²RBench: Automated Paradigm for Formally Verifiable Abstract Reasoning

**Anonymous Repository for Review**

This repository contains the official implementation of the pipeline described in the paper: **"A²RBench: An Automatic Paradigm for Formally Verifiable Abstract Reasoning Benchmark Generation"**.

A²RBench is an automated framework that leverages LLMs to generate, verify, expand, and evaluate abstract reasoning tasks. It ensures logical soundness through **Cycle Consistency Check ($g(f(x)) = x$)** and code-based verification.

## 📂 Repository Structure

```text
.
├── llm_client.py           # Unified API wrapper for various LLM providers (OpenAI, Gemini, etc.)
├── question.py             # Stage 1: Seed Generation (Rules & Code) via Cycle Consistency
├── expand_questions.py     # Stage 2: Task Expansion (Variations V0-V9)
├── answer.py               # Stage 3: Solver Evaluation (P0 & P1 perturbations)
├── analysis.py             # Stage 4: Global Performance & Symbolic Dependency Analysis
├── analyze_complexity.py   # Analysis: Code Complexity (AST) & Author Fingerprinting
├── classify_failures.py    # Analysis: Cognitive Failure Classification (CoT Diagnosis)
└── analysis_expand.py      # Analysis: Augmentation Paradox & Entropy Analysis
```

## 🛠️ Installation

1. **Clone the repository:**
2. **Install dependencies:**
   
   *(Note: Core dependencies include `openai`, `pandas`, `numpy`, `scipy`, `rapidfuzz`, `matplotlib`, `seaborn`, `tqdm`)*

## ⚙️ Configuration

Before running the pipeline, you must configure your LLM endpoints in `llm_client.py`.

Open `llm_client.py` and update the `API_CONFIG` dictionary with your actual provider details.

> **Note:** The code supports multiple providers. Ensure your API keys are set up correctly.

```python
# llm_client.py
API_CONFIG = {
    "all": {
        "base_url": "YOUR_BASE_URL_HERE", 
        "keys": [
            "YOUR_API_KEY_HERE"
        ]
    },
}
```

## 🚀 Usage Pipeline

The pipeline consists of four sequential stages.

### Stage 1: Seed Generation

Generates valid Python-based abstract reasoning rules (Forward $f$ and Inverse $g$) and verifies them via Cycle Consistency. Supports 1D, 2D, and 3D tasks.

```bash
python question.py
```

*Outputs: `questions_arc_text_v9/validated_arc_text_questions.jsonl`*

### Stage 2: Task Expansion

Augments the seed tasks into 9 distinct variations (Standard, Edge Case, Adversarial) to test robustness.

```bash
python expand_questions.py
```

*Outputs: `questions_arc_text_v9/expanded_arc_text_questions.jsonl`*

### Stage 3: Solver Evaluation

Evaluates various "Solver" LLMs on the generated tasks. This script also handles **Symbolic Dependency Testing** by creating symbol-remapped versions (P1) of the tasks.

```bash
python answer.py
```

*Outputs: `results_arc_text_v9/results_raw.jsonl`*

### Stage 4: Analysis & Metrics

We provide multiple scripts to analyze the results from different perspectives:

1. **Global Leaderboard & Symbolic Dependency:**
    Generates the main accuracy tables and symbolic dependency gaps.
    
    ```bash
    python analysis.py
    ```
2. **Code Complexity Analysis:**
    Analyzes the AST (Abstract Syntax Tree) of generated rules to measure loop depth, conditional complexity, etc.
    
    ```bash
    python analyze_complexity.py
    ```
3. **Cognitive Failure Classification:**
    Uses an Analyst LLM to diagnose *why* a model failed (e.g., Abstraction Failure vs. Reasoning Failure).
    
    ```bash
    python classify_failures.py
    ```
4. **Augmentation & Entropy Analysis:**
    Investigates the relationship between input complexity (compression ratio) and model performance.
    
    ```bash
    python analysis_expand.py
    ```

## 📊 Data Format

### Task Format (JSON)

Each generated task includes the executable Python code, the natural language rule description, and the puzzle data.

```json
{
  "question_id": "SYM_O4_D1_001_V0",
  "task_type": "SymbolicRule",
  "dimensionality": 1,
  "rule_description": "Reverse string...",
  "python_code": "def transform(x): ...",
  "puzzle_data": {
    "examples": [{"input": "ABC", "output": "CBA"}],
    "question_plaintext": "XYZ",
    "answer_ciphertext": "ZYX"
  }
}
```

## ⚖️ License

This project is licensed under the MIT License.


