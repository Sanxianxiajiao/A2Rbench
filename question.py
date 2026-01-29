# question.py (Version 9.3 - 3D-Hard Mode & Bi-Directional Logic)
import os
import json
import itertools
import time
import random
import math
import subprocess
import tempfile
from llm_client import LLMClient
from collections import defaultdict

# --- CONFIGURATION V9.3 ---
START_FRESH = False
TARGET_PER_COMBO = 1
ARC_RULE_FILE = "arc-rule/rule.json"
NUM_SAMPLED_RULES = 4
MAX_ATTEMPTS = 3 # Maximum number of retries for a failed stage

AUTHOR_MODELS = {
    "Author_O4_mini": {"provider": "all", "model": "o4-mini"},
    "Author_GPT5_Mini": {"provider": "all", "model": "gpt-5-mini"},
    "Author_Gemini_Pro": {"provider": "all", "model": "gemini-2.5-pro"},
    "Author_Gemini_Flash": {"provider": "all", "model": "gemini-2.5-flash"},
}

JUDGE_MODEL = {"provider": "all", "model": "gpt-5-mini"}

TASK_TYPES = ["SymbolicRule", "SemanticRule"]
# Added "3D-Hard" to explicitly test LLM limits on true volumetric reasoning
DIMENSIONALITIES = [1, 2, 3]

# --- FILE PATHS V9 ---
OUTPUT_DIR = "questions_arc_text_v10" #$1046.74115 ->  $1051.34567 -> 1052.396182
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "validated_arc_text_questions.jsonl")
STATS_FILE = os.path.join(OUTPUT_DIR, "generation_stats_arc_text.json")
USED_COMBINATIONS_FILE = os.path.join(OUTPUT_DIR, "used_arc_combinations.json")

# --- DYNAMIC PROMPTS V9.3 (Bi-Directional + 3D-Hard) ---

DIMENSION_INSTRUCTIONS = {
    1: "You must invent a rule for manipulating a **1D sequence of characters** (a string).",
    2: "You must invent a rule for transforming a **2D grid of characters** (a list of strings).",
    3: "You must invent a rule for transforming a **3D cube of characters** (a list of lists of strings). Standard layer-by-layer operations are acceptable.",
    "3D-Hard": """You must invent a **Object-Oriented & Topological 3D Rule**.
**CRITICAL CHANGE:**
Stop using simple "neighbor checks" or "physics simulations" (like raycasting).
Instead, your rule must treat the 3D grid as a collection of **Discrete Objects** or analyze **Global Topology**.

**REQUIRED MECHANISMS (Must use at least one):**
1.  **Object Identification (The "Blob" Logic):**
    *   "Identify all connected groups (blobs) of character 'X'. If a blob has a volume > 5, move it to the bottom layer. If volume < 3, delete it."
    *   *Why:* Requires Breadth-First Search (BFS) or grouping logic, which is much harder than coordinate loops.

2.  **Topological Properties (Holes & Enclosures):**
    *   "Find 3D structures that form a 'cage' or 'shell' around other cells. Fill the inside of the cage with 'W'."
    *   "Detect if a line of symbols 'knots' or forms a closed loop in 3D space."

3.  **High-Entropy Conditional Logic (The "Arbitrary" Logic):**
    *   "Analyze each Z-layer. If Layer N has more 'A's than Layer N+1, Swap them. BUT, if Layer N contains a 'Z', leave it frozen."
    *   Use complex `if-else` chains based on abstract properties (counts, symmetry, containment) rather than simple position.

**GOAL:** The solver should NOT be able to solve this by just simulating a physical process. They must deduce abstract properties of 3D shapes."""
}

TASK_INSTRUCTIONS = {
"SymbolicRule": """The rule must operate on the structure, position, or counts of characters, COMPLETELY INDEPENDENT of their semantic meaning.
**CRITICAL:** The logic must be fully reversible. Do not use lossy operations like 'delete all vowels' (you can't know what was deleted) or 'replace numbers with #'.""",
"SemanticRule": """The rule must rely on real-world knowledge (linguistic, mathematical, cultural).
**CRITICAL:** To be reversible, the semantic relationship must be 1-to-1 (Bijective).
*   BAD (Not Reversible): 'Replace city with country' (Paris->France is easy, but France->? is ambiguous).
*   GOOD (Reversible): 'Replace letter with its symmetrical opposite in the alphabet (A<->Z)' or a specific Cipher."""
}

PROMPT_CREATE_RULE = {
    "system": "You are a logical puzzle designer specializing in reversible algorithms. Your task is to invent a novel transformation rule and its corresponding inverse (decoding) rule. Your entire JSON output should be under 8000 tokens.",
    "user_template": """
**Inspiration Rules (2D Grid Examples - FOR INSPIRATION ONLY):**
---
{sampled_rules_str}
---
**Your Task:**
Invent a new rule for the task type: **{task_type}**.

**CRITICAL DIMENSION REQUIREMENT:**
{dimension_instructions}

**CRITICAL REQUIREMENT: REVERSIBILITY (BIJECTIVITY)**
You are designing a **Encoder/Decoder** pair.
1.  **NO INFORMATION LOSS:** The rule must not destroy information. Every input must map to a unique output, and that output must be able to map back to the *exact* original input.
2.  **Forward Rule:** Describes how to transform Input A -> Output B.
3.  **Inverse Rule:** Describes how to verify/reconstruct Input A <- Output B exactly.

**Rule Constraints for {task_type}:**
{task_specific_instructions}

{feedback_section}

**Output Format (Strict JSON):**
```json
{{
  "reasoning_of_creation": "My reasoning for this reversible {dimensionality} rule...",
  "rule_description": "Clear description of the Forward transformation (Input -> Output).",
  "inverse_rule_description": "Clear description of the Inverse transformation (Output -> Input) to prove solvability."
}}```
"""
}

PROMPT_GENERATE_CODE = {
    "system": "You are a meticulous Chief Software Architect. Your goal is to implement a perfect, reversible transformation system in Python. Your output must be self-contained and valid JSON.",
    "user_template": """
**Primary Mission:** Implement the provided rule as a pair of Python functions: an Encoder and a Decoder.

**The Rules You MUST Implement:**
1.  **Forward Rule:** `{rule_description}`
2.  **Inverse Rule:** `{inverse_rule_description}`

**Data Dimension:** {dimensionality} Structure

---
**Requirements:**

**Part 1: The Twin Functions**
You must implement TWO functions:
1.  `def transform_grid(grid):` -> Returns the transformed output.
2.  `def inverse_transform_grid(grid):` -> Takes the output of `transform_grid` and returns the *exact original input*.

**Part 2: Robustness**
*   Handle edge cases (empty lists, single items) gracefully.
*   **Cycle Consistency:** The logic MUST satisfy `inverse_transform_grid(transform_grid(x)) == x` for all valid inputs.

**Part 3: The Puzzle**
*   Create a `question_plaintext` (valid input) and compute `answer_ciphertext` (correct output).
*   Create diagnostic `examples` showing the transformation.

---
{feedback_section}
**Output Format (Strict JSON with Python string):**
```json
{{
  "reasoning": "I have implemented both functions and verified the cycle consistency...",
  "python_code": "import json\\n\\n# 1. Forward Function\\ndef transform_grid(grid):\\n    # Implementation...\\n    return result\\n\\n# 2. Inverse Function\\ndef inverse_transform_grid(grid):\\n    # Implementation...\\n    return original_grid\\n\\n# 3. Examples and Data\\nexamples = [{{'input': ..., 'output': ...}}]\\nquestion_plaintext = ... # A non-trivial input case\\nanswer_ciphertext = transform_grid(question_plaintext)\\n\\n# 4. Self-Validation (Important)\\nreconstructed = inverse_transform_grid(answer_ciphertext)\\nif reconstructed != question_plaintext:\\n    raise ValueError('Cycle Check Failed! Code is logically flawed.')\\n\\n# 5. Final Output\\nprint(json.dumps({{\\n    'examples': examples,\\n    'question_plaintext': question_plaintext,\\n    'answer_ciphertext': answer_ciphertext\\n}}))"
}}```
"""
}

PROMPT_JUDGE_CODE = {
    "system": "You are a pragmatist logic judge. Your job is to verify that the generated puzzle is solvable and clearly defined.",
    "user_template": """
**Validation Task**
We have a puzzle generated by Python code.

**Rule Definitions:**
- Forward: `{rule_description}`
- Inverse: `{inverse_rule_description}`

**Python Code:**
```python
{python_code}
```

**Generated Puzzle Data:**
```json
{code_output_str}
```

**Your Judgement Criteria:**
1.  **Consistency:** Does the Python code actually implement the Forward and Inverse rules described?
2.  **Solvability:** Are the provided examples sufficient for a human/AI to deduce the rule? (i.e., The rule isn't "random" or hidden inside the code without external logic).
3.  **Quality:** Is the puzzle non-trivial?

**Output Format (Strict JSON):**
```json
{{
  "reasoning": "Analysis of rule consistency and example quality...",
  "is_valid": boolean
}}
```
"""
}

# --- Helper Functions ---
def load_arc_rules(filepath: str) -> list[str]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return [item['Rule Description'] for item in json.load(f)]
    except Exception as e: print(f"[Fatal Error] Could not load ARC rules: {e}"); exit()

def load_used_combinations(filepath: str) -> set:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return {tuple(item) for item in json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError): return set()

def save_used_combinations(filepath: str, combinations_set: set):
    with open(filepath, 'w', encoding='utf-8') as f: json.dump([list(item) for item in combinations_set], f)

def generate_unique_rule_combination(all_rules: list, used_combinations: set) -> tuple[list[str], tuple] | None:
    if len(used_combinations) >= math.comb(len(all_rules), NUM_SAMPLED_RULES): return None
    while True:
        indices = random.sample(range(len(all_rules)), NUM_SAMPLED_RULES)
        combo_signature = tuple(sorted(indices))
        if combo_signature not in used_combinations:
            return [all_rules[i] for i in indices], combo_signature

def execute_code(code_string: str, timeout_seconds: int = 10) -> tuple[bool, dict | str]:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
        tmp_file_path = tmp_file.name
        tmp_file.write(code_string)
    
    try:
        process = subprocess.run(
            ['python', tmp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=True,
            encoding='utf-8'
        )
        return True, json.loads(process.stdout)
    except subprocess.TimeoutExpired:
        return False, "Execution timed out."
    except subprocess.CalledProcessError as e:
        error_message = f"Execution failed with exit code {e.returncode}.\nStderr:\n{e.stderr}"
        return False, error_message
    except json.JSONDecodeError:
        return False, "Execution succeeded, but the output was not valid JSON."
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def validate_puzzle_with_code(python_code: str, question: any, expected_answer: any) -> tuple[bool, str]:
    """
    Executes the python_code to strictly verify Cycle Consistency with explicit Debug Prints.
    Cycle: input == inverse_transform_grid(transform_grid(input))
    """
    try:
        scope = {}
        exec(python_code, scope)
        
        transform_func = scope.get('transform_grid')
        inverse_func = scope.get('inverse_transform_grid')
        
        if not callable(transform_func):
            return False, "Missing function: 'transform_grid' not defined."
        if not callable(inverse_func):
            return False, "Missing function: 'inverse_transform_grid' not defined."
            
        print(f"      [Debug] Valdating Input Type: {type(question)}")
        
        # --- 1. Forward Check ---
        try:
            actual_answer = transform_func(question)
        except Exception as e:
            return False, f"Forward execution crashed: {e}"

        # Visual Debug
        q_str = str(question)
        a_str = str(expected_answer)
        act_str = str(actual_answer)
        print(f"      [Debug] Forward Check:")
        print(f"          Input:    {q_str if len(q_str)<100 else q_str[:100]+'...'}")
        print(f"          Expected: {a_str if len(a_str)<100 else a_str[:100]+'...'}")
        print(f"          Actual:   {act_str if len(act_str)<100 else act_str[:100]+'...'}")

        if actual_answer != expected_answer:
            return False, f"Forward mismatch.\nExpected: {json.dumps(expected_answer)}\nActual: {json.dumps(actual_answer)}"
            
        # --- 2. Backward Check (The Gold Standard) ---
        try:
            reconstructed_question = inverse_func(actual_answer)
        except Exception as e:
            return False, f"Inverse execution crashed: {e}"

        rec_str = str(reconstructed_question)
        print(f"      [Debug] Cycle Check:")
        print(f"          Reconstructed: {rec_str if len(rec_str)<100 else rec_str[:100]+'...'}")

        if reconstructed_question != question:
            return False, (f"Cycle Consistency Failed (The rule is not reversible).\n"
                           f"Original Input: {json.dumps(question)}\n"
                           f"Reconstructed:  {json.dumps(reconstructed_question)}")

        return True, "Validation successful: Forward and Inverse logic is consistent."
            
    except Exception as e:
        return False, f"Code validation crashed with syntax/runtime error: {e}"

# --- Pipeline Stage Functions V9.3 ---

def generate_rule(client, task_type, sampled_rules, dimensionality, previous_attempts=None):
    feedback_section = ""
    if previous_attempts:
        feedback_section += "\n\n--- PREVIOUS FAILED ATTEMPTS (AVOID THESE MISTAKES) ---\n"
        for i, attempt in enumerate(previous_attempts):
            feedback_section += f"Attempt #{i+1} Failed: {attempt['feedback']}\n"
            feedback_section += f"Rejected Rule: {attempt['rule_description']}\n"
    
    # --- 3D-Hard Injection Logic ---
    hard_mode_instruction = ""
    if dimensionality == "3D-Hard":
        hard_mode_instruction = """
**HARD MODE ACTIVE - STRICT VALIDATION:**
You are currently designing for the '3D-Hard' tier. The Judge will **REJECT** trivial global transformations or simple layer-by-layer operations.

**CORE REQUIREMENT: ITERATIVE & INTERACTIVE LOGIC**
The rule must describe a dynamic system or a complex calculation, NOT just a geometric mapping.

*   **REJECTED (Too Simple/Linear):** 
    *   "Reverse every string in every layer." (1D logic)
    *   "Rotate the entire cube 90 degrees." (Pure math, too easy for LLMs)
    *   "Swap Layer 1 and Layer 2." (No voxel interaction)

*   **ACCEPTED (Complex, Iterative, Non-Linear):**
    *   **3D Cellular Automata:** "Run 3 steps of a simulation: A cell becomes 'X' only if exactly 4 neighbors in its 3x3x3 neighborhood are 'X', otherwise it decays to '.'." (Requires tracking state updates).
    *   **Particle Physics / Collision:** "Particles 'A' fall down (z+1) and Particles 'B' float up (z-1). If they collide at the same voxel, they merge into 'C' and freeze." (Requires handling movement and interaction).
    *   **Ray Casting with Reflection:** "Cast 'light' from the top layer. If it hits a '/' mirror, it reflects horizontally. Mark all illuminated paths." (Requires tracing complex paths).
    *   **Volumetric Encryption:** "The value of cell (x,y,z) is determined by XORing the values of its 6 orthogonal neighbors, then shifting based on the value at (z,y,x)." (High calculation density).

**GOAL:** Create a rule that requires the solver to *mentally simulate* a process step-by-step, rather than just applying a formula.
"""

    user_prompt = PROMPT_CREATE_RULE["user_template"].format(
        sampled_rules_str="\n".join(f"- {rule}" for rule in sampled_rules),
        task_type=task_type,
        dimensionality=dimensionality,
        dimension_instructions=DIMENSION_INSTRUCTIONS[dimensionality],
        task_specific_instructions=TASK_INSTRUCTIONS[task_type] + hard_mode_instruction,
        feedback_section=feedback_section
    )
    response = client.make_request(PROMPT_CREATE_RULE["system"], user_prompt, temperature=0.5) # Increased temp slightly for Hard mode creativity
    return response.get("action") if response else None

def judge_rule(client, rule_candidate):
    if "inverse_rule_description" not in rule_candidate:
        return {"is_valid": False, "reasoning": "Missing 'inverse_rule_description'."}

    user_prompt = f"Review this rule pair.\nForward: `{rule_candidate['rule_description']}`\nInverse: `{rule_candidate['inverse_rule_description']}`\n\nIs this rule logical, unambiguous, and truly reversible (lossless)?"
    system_prompt = "You are a strict logic judge. Respond in JSON with {'is_valid': boolean, 'reasoning': '...'}"
    response = client.make_request(system_prompt, user_prompt, temperature=0.1)
    return response.get("action") if response else None

def generate_code(client, rule_desc, inv_rule_desc, dimensionality, previous_attempts=None):
    feedback_section = ""
    if previous_attempts:
        feedback_section += "\n\n--- PREVIOUS FAILED CODE ATTEMPTS ---\n"
        for i, attempt in enumerate(previous_attempts):
            feedback_section += f"Attempt #{i+1} Failed: {attempt['feedback']}\n"
    
    user_prompt = PROMPT_GENERATE_CODE["user_template"].format(
        rule_description=rule_desc,
        inverse_rule_description=inv_rule_desc,
        dimensionality=dimensionality,
        feedback_section=feedback_section
    )
    response = client.make_request(PROMPT_GENERATE_CODE["system"], user_prompt, temperature=0.2)
    return response.get("action") if response else None

def judge_code_and_output(client, rule_desc, inv_rule_desc, python_code, code_output):
    code_output_str = json.dumps(code_output, indent=2)
    user_prompt = PROMPT_JUDGE_CODE["user_template"].format(
        rule_description=rule_desc,
        inverse_rule_description=inv_rule_desc,
        python_code=python_code,
        code_output_str=code_output_str
    )
    response = client.make_request(PROMPT_JUDGE_CODE["system"], user_prompt, temperature=0.1)
    return response.get("action") if response else None

# --- Main Orchestrator V9.3 ---
def main():
    print("--- Starting Puzzle Generation (V9.3 - With 3D-Hard Mode) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_arc_rules = load_arc_rules(ARC_RULE_FILE)
    used_combinations = load_used_combinations(USED_COMBINATIONS_FILE)

    if START_FRESH:
        for f in [OUTPUT_FILE, STATS_FILE, USED_COMBINATIONS_FILE]:
            if os.path.exists(f): os.remove(f)
        used_combinations = set()

    # Load Progress
    progress = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                # dim can be string "3D-Hard", ensure consistent key type
                progress[record['author_model']][record['task_type']][record['dimensionality']] += 1
    except: pass
        
    # Load Stats
    generation_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    try:
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            loaded_stats = json.load(f)
            for author, tasks in loaded_stats.items():
                for task, dims in tasks.items():
                    for dim, stages in dims.items():
                        for stage, value in stages.items():
                            generation_stats[author][task][dim][stage] = value
        print("Stats loaded successfully.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("Stats file not found or invalid. Starting fresh stats.")
    
    author_clients = {name: LLMClient(config["provider"], config["model"], max_tokens=15000) for name, config in AUTHOR_MODELS.items()}
    judge_client = LLMClient(JUDGE_MODEL["provider"], JUDGE_MODEL["model"], max_tokens=15000,max_retries=10)
    
    task_combinations = list(itertools.product(author_clients.items(), TASK_TYPES, DIMENSIONALITIES))
    task_cycler = itertools.cycle(task_combinations)

    consecutive_skips = 0
    total_combinations = len(task_combinations)

    while True:
        (author_name, author_client), task_type, dimensionality = next(task_cycler)

        # Check completion
        if progress[author_name][task_type][dimensionality] >= TARGET_PER_COMBO:
            consecutive_skips += 1
            if consecutive_skips > total_combinations + 2:
                print("\n[Finished] All target combinations have been met.")
                break
            continue
        
        consecutive_skips = 0
        
        # Display dimensionality clearly (str() handles "3D-Hard")
        print(f"\n--- New Task: {author_name} | {task_type} | {dimensionality} ---")
        stats = generation_stats[author_name][task_type][str(dimensionality)]

        # --- Stage 1: Rule Generation ---
        print("  > Stage 1: Generating Reversible Rule...")
        rule_attempts = []
        validated_rule_desc, validated_inv_rule_desc = None, None
        
        combo_result = generate_unique_rule_combination(all_arc_rules, used_combinations)
        if not combo_result:
            print("[Warning] No more unique ARC rule combinations available.")
            break
        sampled_rules, new_combo_signature = combo_result

        for _ in range(MAX_ATTEMPTS):
            stats["rule_generation_attempts"] += 1
            rule_candidate = generate_rule(author_client, task_type, sampled_rules, dimensionality, rule_attempts)

            if not (rule_candidate and "rule_description" in rule_candidate and "inverse_rule_description" in rule_candidate):
                rule_attempts.append({"rule_description": "N/A", "feedback": "JSON missing rule or inverse rule."})
                continue
            
            rule_judgement = judge_rule(judge_client, rule_candidate)
            if rule_judgement and rule_judgement.get("is_valid"):
                validated_rule_desc = rule_candidate["rule_description"]
                validated_inv_rule_desc = rule_candidate["inverse_rule_description"]
                stats["rule_generation_success"] += 1
                print("    [Success] Rule accepted.")
                break
            else:
                reason = rule_judgement.get('reasoning', 'Unknown')
                print(f"      [Reject] {reason}")
                rule_attempts.append({"rule_description": rule_candidate.get("rule_description", ""), "feedback": reason})
        
        if not validated_rule_desc: continue

        # --- Stages 2-5: Code & Validation ---
        print("  > Stages 2-5: Coding & Cycle Validation...")
        code_attempts = []
        final_record = None

        for _ in range(MAX_ATTEMPTS):
            stats["code_generation_attempts"] += 1
            
            # 2. Generate
            code_response = generate_code(author_client, validated_rule_desc, validated_inv_rule_desc, dimensionality, code_attempts)
            if not code_response or "python_code" not in code_response:
                code_attempts.append({"python_code": "", "feedback": "Invalid JSON/Code."})
                continue
            python_code = code_response["python_code"]
            stats["code_generation_success"] += 1

            # 3. Execute
            stats["code_execution_attempts"] += 1
            is_success, result = execute_code(python_code)
            if not is_success:
                print(f"      [Exec Fail] {result}")
                code_attempts.append({"python_code": python_code, "feedback": f"Execution Error: {result}"})
                continue
            stats["code_execution_success"] += 1
            code_output = result

            # 4. Self-Correction
            stats["puzzle_self_validation_attempts"] += 1
            question = code_output.get('question_plaintext')
            answer = code_output.get('answer_ciphertext')
            
            is_puzzle_valid, val_reason = validate_puzzle_with_code(python_code, question, answer)
            if not is_puzzle_valid:
                print(f"      [Validation Fail] {val_reason}")
                code_attempts.append({"python_code": python_code, "feedback": val_reason})
                continue
            stats["puzzle_self_validation_success"] += 1
            print("    [Success] Cycle check passed.")

            # 5. Final Judge
            stats["final_judging_attempts"] += 1
            final_judgement = judge_code_and_output(judge_client, validated_rule_desc, validated_inv_rule_desc, python_code, code_output)
            if final_judgement and final_judgement.get("is_valid"):
                stats["final_judging_success"] += 1
                
                # Format dimension string for ID (handle "3D-Hard" -> "D3DHard")
                dim_str = str(dimensionality).replace("-", "")
                question_id = f"{task_type[:3].upper()}_{author_name.replace('Author_', '')}_D{dim_str}_{progress[author_name][task_type][dimensionality]:03d}"
                
                final_record = {
                    "question_id": question_id,
                    "author_model": author_name,
                    "task_type": task_type,
                    "dimensionality": dimensionality,
                    "rule_description": validated_rule_desc,
                    "inverse_rule_description": validated_inv_rule_desc,
                    "python_code": python_code,
                    "puzzle_data": code_output,
                    "judgement_reasoning": final_judgement.get("reasoning")
                }
                print(f"    [Final Success] {question_id} Saved.")
                break
            else:
                reason = final_judgement.get('reasoning', 'Unknown')
                print(f"      [Judge Fail] {reason}")
                code_attempts.append({"python_code": python_code, "feedback": reason})

        if final_record:
            progress[author_name][task_type][dimensionality] += 1
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(final_record, ensure_ascii=False) + '\n')
            used_combinations.add(new_combo_signature)
            save_used_combinations(USED_COMBINATIONS_FILE, used_combinations)
            with open(STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(generation_stats, f, indent=4)
            time.sleep(1)

    print("\n--- Process Complete ---")

if __name__ == "__main__":
    main()