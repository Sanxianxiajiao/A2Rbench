\# A²RBench: Automated Paradigm for Formally Verifiable Abstract Reasoning



\*\*Anonymous Repository for Review\*\*



This repository contains the official implementation of the pipeline described in the paper: \*\*"A²RBench: An Automatic Paradigm for Formally Verifiable Abstract Reasoning Benchmark Generation"\*\*.



A²RBench is an automated framework that leverages LLMs to generate, verify, expand, and evaluate abstract reasoning tasks. It ensures logical soundness through \*\*Cycle Consistency Check ($g(f(x)) = x$)\*\* and code-based verification.



\## 📂 Repository Structure



```text

.

├── llm\_client.py           # Unified API wrapper for various LLM providers (OpenAI, Gemini, etc.)

├── question.py             # Stage 1: Seed Generation (Rules \& Code) via Cycle Consistency

├── expand\_questions.py     # Stage 2: Task Expansion (Variations V0-V9)

├── answer.py               # Stage 3: Solver Evaluation (P0 \& P1 perturbations)

├── analysis.py             # Stage 4: Global Performance \& Symbolic Dependency Analysis

├── analyze\_complexity.py   # Analysis: Code Complexity (AST) \& Author Fingerprinting

├── classify\_failures.py    # Analysis: Cognitive Failure Classification (CoT Diagnosis)

└── analysis\_expand.py      # Analysis: Augmentation Paradox \& Entropy Analysis

```



\## 🛠️ Installation



1\. \*\*Clone the repository:\*\*

2\. \*\*Install dependencies:\*\*

&nbsp;  

&nbsp;  \*(Note: Core dependencies include `openai`, `pandas`, `numpy`, `scipy`, `rapidfuzz`, `matplotlib`, `seaborn`, `tqdm`)\*



\## ⚙️ Configuration



Before running the pipeline, you must configure your LLM endpoints in `llm\_client.py`.



Open `llm\_client.py` and update the `API\_CONFIG` dictionary with your actual provider details.



> \*\*Note:\*\* The code supports multiple providers. Ensure your API keys are set up correctly.



```python

\# llm\_client.py

API\_CONFIG = {

&nbsp;   "all": {

&nbsp;       "base\_url": "YOUR\_BASE\_URL\_HERE", 

&nbsp;       "keys": \[

&nbsp;           "YOUR\_API\_KEY\_HERE"

&nbsp;       ]

&nbsp;   },

}

```



\## 🚀 Usage Pipeline



The pipeline consists of four sequential stages.



\### Stage 1: Seed Generation



Generates valid Python-based abstract reasoning rules (Forward $f$ and Inverse $g$) and verifies them via Cycle Consistency. Supports 1D, 2D, and 3D tasks.



```bash

python question.py

```



\*Outputs: `questions\_arc\_text\_v9/validated\_arc\_text\_questions.jsonl`\*



\### Stage 2: Task Expansion



Augments the seed tasks into 9 distinct variations (Standard, Edge Case, Adversarial) to test robustness.



```bash

python expand\_questions.py

```



\*Outputs: `questions\_arc\_text\_v9/expanded\_arc\_text\_questions.jsonl`\*



\### Stage 3: Solver Evaluation



Evaluates various "Solver" LLMs on the generated tasks. This script also handles \*\*Symbolic Dependency Testing\*\* by creating symbol-remapped versions (P1) of the tasks.



```bash

python answer.py

```



\*Outputs: `results\_arc\_text\_v9/results\_raw.jsonl`\*



\### Stage 4: Analysis \& Metrics



We provide multiple scripts to analyze the results from different perspectives:



1\. \*\*Global Leaderboard \& Symbolic Dependency:\*\*

&nbsp;   Generates the main accuracy tables and symbolic dependency gaps.

&nbsp;   

&nbsp;   ```bash

&nbsp;   python analysis.py

&nbsp;   ```

2\. \*\*Code Complexity Analysis:\*\*

&nbsp;   Analyzes the AST (Abstract Syntax Tree) of generated rules to measure loop depth, conditional complexity, etc.

&nbsp;   

&nbsp;   ```bash

&nbsp;   python analyze\_complexity.py

&nbsp;   ```

3\. \*\*Cognitive Failure Classification:\*\*

&nbsp;   Uses an Analyst LLM to diagnose \*why\* a model failed (e.g., Abstraction Failure vs. Reasoning Failure).

&nbsp;   

&nbsp;   ```bash

&nbsp;   python classify\_failures.py

&nbsp;   ```

4\. \*\*Augmentation \& Entropy Analysis:\*\*

&nbsp;   Investigates the relationship between input complexity (compression ratio) and model performance.

&nbsp;   

&nbsp;   ```bash

&nbsp;   python analysis\_expand.py

&nbsp;   ```



\## 📊 Data Format



\### Task Format (JSON)



Each generated task includes the executable Python code, the natural language rule description, and the puzzle data.



```json

{

&nbsp; "question\_id": "SYM\_O4\_D1\_001\_V0",

&nbsp; "task\_type": "SymbolicRule",

&nbsp; "dimensionality": 1,

&nbsp; "rule\_description": "Reverse string...",

&nbsp; "python\_code": "def transform(x): ...",

&nbsp; "puzzle\_data": {

&nbsp;   "examples": \[{"input": "ABC", "output": "CBA"}],

&nbsp;   "question\_plaintext": "XYZ",

&nbsp;   "answer\_ciphertext": "ZYX"

&nbsp; }

}

```



\## ⚖️ License



This project is licensed under the MIT License.







