# classify_failures.py (Version 2.0.2 - Fixed Task Counting for P1 Perturbations)

import os
import json
import pandas as pd
from llm_client import LLMClient
from tqdm import tqdm

# --- CONFIGURATION ---

# --- Input Files ---
INPUT_RESULTS_FILE = os.path.join("results_arc_text_v9", "results_raw.jsonl")
INPUT_QUESTIONS_FILE = os.path.join("questions_arc_text_v9", "expanded_arc_text_questions.jsonl")

# --- Output File ---
OUTPUT_DIR = "cognitive_analysis_v9"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "classified_analysis.jsonl")

# --- Analyst LLM Configuration (Use your most capable model) ---
ANALYST_MODEL_CONFIG = {
    "provider": "all",
    "model": "gpt-5-mini",
    "max_tokens": 4096
}

# --- Robust Data Loading Function ---
def load_jsonl_robustly(filepath: str) -> pd.DataFrame:
    """
    Reads a .jsonl file line by line, robustly skipping any corrupted or empty lines.
    """
    print(f"Robustly loading data from: {filepath}")
    records = []
    if not os.path.exists(filepath):
        print(f"  [Error] File not found: {filepath}")
        return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  [Warning] Skipping a corrupted line at line number {i+1} in {filepath}")

    print(f"  Successfully loaded {len(records)} valid records.")
    return pd.DataFrame(records)


# --- V2.0 PROMPT TEMPLATE ---
PROMPT_TEMPLATE = """
System: You are a meticulous and insightful AI cognitive analyst. Your mission is to dissect a model's thought process to understand not just *if* it was right, but *how* it arrived at its conclusion. You must follow the structured classification guide below.

User:
**Analysis Task:**
You are given a case file containing a puzzle, the ground truth, and a model's attempt to solve it. Your job is to perform a two-part analysis:
1.  **Outcome & Reasoning Analysis:** Categorize the final outcome and the underlying reasoning quality.
2.  **Reasoning Style Analysis:** Classify the overall style of the thought process.

**Case File:**
- **Original Rule Description:** {rule_description}
- **Puzzle Dimensionality:** {dimensionality}
- **Ground Truth Answer:** {ground_truth}
- **Model's Answer:** {model_answer}
- **Model's Reasoning (Chain of Thought):**
```
{model_cot}
```

---
**PART 1: Outcome & Reasoning Analysis Guide**

First, determine if the model's answer is correct. Then, follow the appropriate path below.

**PATH A: IF THE MODEL'S ANSWER IS INCORRECT**
Choose ONE error category that is the primary cause of failure:
-   **Abstraction_Failure-Operator_Inference:** The model failed to infer the correct underlying rule/operation from the examples (e.g., inferred 'OR' instead of 'AND'). This is the most common type of abstraction failure.
-   **Abstraction_Failure-Scope_Condition:** The model inferred the correct operation but failed to identify the correct scope or conditions for applying it (e.g., applied it to the whole grid instead of just one row).
-   **Reasoning_Failure-Procedural_Error:** The model seemed to understand the rule but made a mistake in the step-by-step execution (e.g., calculation error, off-by-one error).
-   **Reasoning_Failure-Spatial_Error:** The model specifically failed to handle spatial relationships correctly in 2D/3D grids (e.g., found the wrong neighbor, confused coordinates).
-   **Format_Or_Collapse-Output_Formatting:** The reasoning seems correct, but the final answer's format is wrong (e.g., string instead of list, missing brackets).
-   **Format_Or_Collapse-Reasoning_Collapse:** The reasoning is nonsensical, hallucinatory, or gives up prematurely.

**PATH B: IF THE MODEL'S ANSWER IS CORRECT**
Analyze the quality of the model's reasoning by comparing its CoT to the ground truth rule. Choose ONE success category:
-   **Success-Type_A-Surface_Fitting:** The CoT does not describe a general, abstract rule. Instead, it describes operations specific to the test input's values (e.g., "I see 'ABC' so I move 'ABC' to the end"). The reasoning is not generalizable.
-   **Success-Type_B-Inferior_Rule:** The CoT describes a general rule that works for the examples and the question, but it is overly complex or less general than the true rule (violates Occam's Razor).
-   **Success-Type_C-Correct_Generalization:** The CoT describes a general, abstract rule that is logically equivalent to (or better than) the ground truth rule. This represents true understanding.

---
**PART 2: Reasoning Style Analysis Guide**

Independently of the outcome, classify the style of the CoT. Choose ONE:
-   **Style-Direct_Deduction:** The model proceeds linearly from observation to conclusion with few or no alternative hypotheses.
-   **Style-Hypothesis_Testing:** The model explicitly proposes one or more candidate rules, tests them against the examples, and then selects one. This is a scientific approach.
-   **Style-Chaotic_Guessing:** The reasoning lacks clear structure, jumps between ideas, or appears to be guessing randomly.

---
**Your Response (Strict JSON format):**
Provide your complete analysis in the following JSON format ONLY. Use `null` for fields that are not applicable.

```json
{{
  "justification": "A brief, one-sentence explanation for your choices, referencing the model's CoT.",
  "outcome_category": "CHOSEN_CATEGORY_FROM_GUIDE (e.g., 'Abstraction_Failure-Operator_Inference' or 'Success-Type_C-Correct_Generalization')",
  "reasoning_style": "CHOSEN_STYLE_FROM_GUIDE (e.g., 'Style-Hypothesis_Testing')"
}}
```
"""

def main():
    print("--- Starting Cognitive Process Analysis (V2.0.2 - Corrected Task Filtering) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Data ---
    df_results = load_jsonl_robustly(INPUT_RESULTS_FILE)
    df_questions = load_jsonl_robustly(INPUT_QUESTIONS_FILE)

    if df_results.empty or df_questions.empty:
        print("[FATAL] Input data is empty or could not be loaded. Aborting.")
        return
    
    # --- FIXED FILTERING LOGIC ---
    # 定义唯一任务标识：ID + 扰动类型 (这样 P0 和 P1 会被分开计算，总数应为 1052)
    df_results['task_instance_key'] = df_results['question_id'].astype(str) + "_" + df_results['perturbation_type']
    
    # 总任务数应为结果集中所有出现过的 (ID, Perturbation) 组合的并集
    total_task_instances = df_results['task_instance_key'].nunique()
    
    if total_task_instances > 0:
        # 计算每个模型完成的唯一任务实例数
        model_counts = df_results.groupby('answerer_model')['task_instance_key'].nunique()
        
        # 找出完成率 >= 99% 的模型
        valid_models = model_counts[model_counts / total_task_instances >= 0.99].index.tolist()
        
        print(f"\n[FILTERING] Total Unique Task Instances (P0 + P1): {total_task_instances}")
        print(f"Models passing 99% completion threshold: {len(valid_models)}")
        
        # 剔除不合格模型
        df_results = df_results[df_results['answerer_model'].isin(valid_models)].copy()
        
        if df_results.empty:
            print("[FATAL] No data remains after filtering incomplete models.")
            return

    # Merge dataframes
    df_results['question_id'] = df_results['question_id'].astype(str)
    df_questions['question_id'] = df_questions['question_id'].astype(str)
    df_merged = pd.merge(df_results, df_questions.drop(columns=['python_code', 'puzzle_data'], errors='ignore'), on='question_id', suffixes=('_res', '_q'))
    print(f"Loaded a total of {len(df_merged)} cases to analyze.")

    # --- 2. Handle Resumability ---
    processed_tasks = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from existing output file: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    task_id = (record['question_id'], record['answerer_model'], record['perturbation_type'])
                    processed_tasks.add(task_id)
                except (json.JSONDecodeError, KeyError):
                    continue

        print(f"Found {len(processed_tasks)} already classified cases. Skipping them.")
        df_merged['task_id'] = df_merged.apply(lambda row: (row['question_id'], row['answerer_model'], row['perturbation_type']), axis=1)
        df_to_process = df_merged[~df_merged['task_id'].isin(processed_tasks)].drop(columns=['task_id'])
    else:
        df_to_process = df_merged

    # --- 3. Classify Cognitive Processes ---
    if not df_to_process.empty:
        print(f"\nInitializing Analyst LLM: {ANALYST_MODEL_CONFIG['model']}")
        analyst_client = LLMClient(provider=ANALYST_MODEL_CONFIG['provider'], model=ANALYST_MODEL_CONFIG['model'], max_tokens=ANALYST_MODEL_CONFIG['max_tokens'])

        with open(OUTPUT_FILE, 'a', encoding='utf-8', errors='surrogatepass') as f:
            for row in tqdm(df_to_process.to_dict('records'), total=len(df_to_process), desc="Analyzing Cognitive Processes"):
                prompt = PROMPT_TEMPLATE.format(
                    rule_description=row.get('rule_description_q', row.get('rule_description', 'N/A')),
                    dimensionality=row.get('dimensionality_q', row.get('dimensionality', 'N/A')),
                    ground_truth=json.dumps(row['ground_truth']),
                    model_answer=json.dumps(row['model_answer']),
                    model_cot=str(row['model_cot'])
                )
                response = analyst_client.make_request("You are an expert AI cognitive analyst.", prompt, temperature=1e-7)
                output_record = row
                action = response.get("action")
                if action and isinstance(action, dict):
                    output_record['outcome_category'] = action.get("outcome_category", "CLASSIFICATION_FAILED")
                    output_record['reasoning_style'] = action.get("reasoning_style", "STYLE_UNKNOWN")
                    output_record['analyst_justification'] = action.get("justification", "No justification provided.")
                else:
                    output_record['outcome_category'] = "CLASSIFICATION_FAILED"
                    output_record['reasoning_style'] = "STYLE_UNKNOWN"
                    output_record['analyst_justification'] = response.get("response_text", "LLM response was empty or invalid.")
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')

    print("\n--- Classification process complete. ---")

    # --- 4. Final Analysis ---
    try:
        df_classified_raw = load_jsonl_robustly(OUTPUT_FILE) 
        if df_classified_raw.empty:
            return

        # 再次进行严格过滤（针对已分类数据）
        df_classified_raw['task_instance_key'] = df_classified_raw['question_id'].astype(str) + "_" + df_classified_raw['perturbation_type']
        total_task_instances = df_classified_raw['task_instance_key'].nunique()
        
        model_stats = df_classified_raw.groupby('answerer_model')['task_instance_key'].nunique()
        valid_models = model_stats[model_stats / total_task_instances >= 0.99].index.tolist()
        df_classified = df_classified_raw[df_classified_raw['answerer_model'].isin(valid_models)].copy()

        if df_classified.empty:
            print(f"\n[Warning] No models passed the 99% threshold ({total_task_instances} tasks).")
            return

        print(f"\n[1] Overall Distribution of Outcomes (N={total_task_instances} per model):")
        print(df_classified['outcome_category'].value_counts(normalize=True).apply('{:.2%}'.format))

        print("\n[2] Overall Distribution of Reasoning Styles:")
        print(df_classified['reasoning_style'].value_counts(normalize=True).apply('{:.2%}'.format))

        print("\n[3] Cognitive Profile for Each Model:")
        for model in sorted(valid_models):
            model_df = df_classified[df_classified['answerer_model'] == model]
            processed_tasks = model_df['task_instance_key'].nunique()
            completion_percentage = (processed_tasks / total_task_instances) * 100
            
            print(f"\n--- {model.replace('Answerer_', '')} ({processed_tasks}/{total_task_instances} | {completion_percentage:.1f}%) ---")
            print("  a) Outcome Profile:")
            print(model_df['outcome_category'].value_counts(normalize=True).apply('{:.2%}'.format).to_string())
            print("\n  b) Reasoning Style Profile:")
            print(model_df['reasoning_style'].value_counts(normalize=True).apply('{:.2%}'.format).to_string())

        print("\n[4] Correlation between Reasoning Style and Success Quality:")
        successful_cases = df_classified[df_classified['is_correct'] == True]
        if not successful_cases.empty:
            style_vs_success_quality = pd.crosstab(successful_cases['reasoning_style'], successful_cases['outcome_category'], normalize='index')
            print(style_vs_success_quality.applymap('{:.2%}'.format))

    except Exception as e:
        print(f"\n[Error] An error occurred during final analysis: {e}")

if __name__ == "__main__":
    main()