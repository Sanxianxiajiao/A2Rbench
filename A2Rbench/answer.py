# answer.py (Version 9.0.1 - Multi-Dimensional ARC-Text Solver)
# Compatible with question.py V9.3 (Supports "3D-Hard" dimensionality)

import os
import json
import random
import string
import time
from llm_client import LLMClient

# --- CONFIGURATION ---
ANSWERER_MODELS = {
    "Answerer_O4_mini": {"provider": "all", "model": "o4-mini"}, #366 403
    "Answerer_GPT4o_Mini": {"provider": "all", "model": "gpt-4o-mini"}, #403 405
    "Answerer_GPT5_Mini": {"provider": "all", "model": "gpt-5-mini"}, #405  419
    "Answerer_GPT5": {"provider": "all", "model": "gpt-5"}, # 520 559
    "Answerer_Gemini_Flash": {"provider": "all", "model": "gemini-2.5-flash"}, #419 441
    # "Answerer_Gemini_Pro": {"provider": "all", "model": "gemini-2.5-pro"}, #
    "Answerer_Gemini3_Flash": {"provider": "all", "model": "gemini-3-flash-preview"},#441 482
    "Answerer_Qwen3_8B": {"provider": "all", "model": "Qwen/Qwen3-8B"},  #491 
    "Answerer_Qwen3_14B": {"provider": "all", "model": "Qwen/Qwen3-14B"},  #502 
    "Answerer_Qwen3_32B": {"provider": "all", "model": "Qwen/Qwen3-32B"},  #      505
    "Answerer_GLM_4.6": {"provider": "all", "model": "zai-org/GLM-4.6"},  #495 502
    # "Answerer_Deepseek_V3_2": {"provider": "all", "model": "deepseek-v3.2-251201"},
    "Answerer_GPT_5_2": {"provider": "all", "model": "gpt-5.2"},# 510 520
    # "Answerer_Claude_Sonnet_4_5": {"provider": "all", "model": "claude-sonnet-4-5-20250929"}
    # "Answerer_Grok4": {"provider": "all", "model": "grok-4-0709"},#690 793
    "Answerer_Gemini3_Pro": {"provider": "all", "model": "gemini-3-pro-preview"},#793
}

JUDGE_MODEL_CONFIG = {"provider": "all", "model": "gpt-5-mini"}

# --- PROMPTS (通用设计，无需修改即可处理多维数据，包括 3D-Hard) ---
PROMPTS = {
    "answerer": {
        "system": "You are a powerful, EFFICIENT, and logical AI specializing in abstract reasoning. Your goal is to deduce the hidden rule and solve the puzzle concisely. Your output MUST be a single, raw JSON object.",
        "user": """
Your task is to analyze the examples, deduce the rule, and solve the question. The input/output can be strings (1D), lists of strings (2D grids), or lists of lists of strings (3D cubes).

---
**CRITICAL INSTRUCTIONS:**
1.  **CONCISENESS GOAL:** Your entire JSON output should aim to be well under 8000 tokens.
2.  **FAILURE PROTOCOL:** If the rule is too complex to be explained concisely, you MUST follow this protocol:
    - In the `reasoning` field, write: "The pattern is too complex to be explained concisely within the token limit."
    - In the `final_answer` field, return `null`.

---
**DEMONSTRATION (1D String Example):**
**Puzzle Examples:**
Examples:
  - Example: "ABC" -> "CBA"
  - Example: "1234" -> "4321"
Question: Solve for "apple"
**Correct Output Format:**
{{
    "reasoning": "1. **Observation:** The output is always the reverse of the input string.\\n2. **Hypothesis:** The rule is to reverse the sequence.\\n3. **Verification:** 'ABC' reversed is 'CBA'. '1234' reversed is '4321'.\\n4. **Application:** Applying the rule to 'apple' results in 'elppa'.",
    "final_answer": "elppa"
}}
---
**DEMONSTRATION (2D Grid Example):**
**Puzzle Examples:**
Examples:
  - Example: ["AB", "CD"] -> ["CD", "AB"]
Question: Solve for ["12", "34"]
**Correct Output Format:**
{{
    "reasoning": "The rule is to reverse the order of the rows in the 2D grid.",
    "final_answer": ["34", "12"]
}}
---

**NOW, SOLVE THE FOLLOWING PUZZLE:**

**Puzzle Examples:**
{puzzle_examples}

**Your Mission:**
Provide your solution in the exact same JSON format. The `final_answer` field should match the data type of the puzzle (string, list of strings, etc.).

**Output Format (Strictly adhere to this JSON structure):**
{{
    "reasoning": "<Your concise, step-by-step thought process, or the failure message if applicable>",
    "final_answer": <The final answer here (string, list, or list of lists), or null if applicable>
}}
"""
    },
    "judge": {
        "system": "You are a meticulous and impartial AI judge. Your task is to determine if a model's answer is correct based on the provided ground truth. Your output MUST be a single, raw JSON object.",
        "user": """
You are judging the correctness of an AI's answer to a puzzle.

---
**Puzzle Information:**
- **Task Type:** {task_type}
- **Rule Description:** {rule_description}
- **Question:** {question_text}

---
**Ground Truth (The official correct answer):**
`{ground_truth}`

---
**Model's Answer:**
`{model_answer}`

---
**Judging Instructions:**
1.  **Primary Goal:** Determine if the `Model's Answer` is equivalent to the `Ground Truth`.
2.  **Equivalence:** The answer is correct if it is an exact match. For lists or nested lists, the structure, order, and content must match exactly.
3.  **Reasoning vs. Answer:** Your judgment should be based on the final answer, not the reasoning. A correct answer with flawed reasoning is still correct.
4.  **Format:** Provide your verdict in the specified JSON format.

---
**Output Format (Strictly adhere to this JSON structure):**
{{
    "justification": "<A brief, one-sentence explanation for your decision.>",
    "is_correct": <true or false>
}}
"""
    }
}

# --- FILE PATHS (与 V9 版本的 question.py 对应) ---
INPUT_FILE = os.path.join("questions_arc_text_v9", "expanded_arc_text_questions.jsonl")
OUTPUT_DIR = "results_arc_text_v9"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results_raw.jsonl")

class Judge:
    def __init__(self, client: LLMClient):
        self.client = client

    def evaluate(self, task_type: str, rule_description: str, question_text: any, ground_truth: any, model_answer: any, model_cot: str) -> tuple[bool, str]:
        system_prompt = PROMPTS["judge"]["system"]
        # 中文注释：使用 json.dumps 可以确保无论数据是一维、二维还是三维(包括3D-Hard)，都能被正确地序列化为字符串。
        user_prompt = PROMPTS["judge"]["user"].format(
            task_type=task_type,
            rule_description=rule_description,
            question_text=json.dumps(question_text),
            ground_truth=json.dumps(ground_truth),
            model_answer=json.dumps(model_answer),
            model_cot=model_cot
        )
        response = self.client.make_request(system_prompt, user_prompt, temperature=1e-7)
        action = response.get("action")

        if action and isinstance(action, dict):
            is_correct = action.get("is_correct", False)
            justification = action.get("justification", "Error: No justification provided.")
            return bool(is_correct), str(justification)
        else:
            print(f"    [Judge Warning] Judge returned invalid JSON. Falling back to direct comparison.")
            # 中文注释：如果 Judge 模型返回格式错误，则进行最严格的直接比较。
            return (model_answer == ground_truth), "Fallback: Judge failed to produce valid JSON."

class PerturbationEngine:
    # 中文注释：json.dumps 自动处理 list of lists 结构。
    def _format_puzzle_text(self, examples: list, question_plaintext: any) -> str:
        formatted_examples = "\n".join([f"  - Example: {json.dumps(ex['input'])} -> {json.dumps(ex['output'])}" for ex in examples])
        return (
            f"Examples:\n{formatted_examples}\n\n"
            f"Question: Solve for {json.dumps(question_plaintext)}"
        )
        
    # 中文注释：递归映射函数，完美支持 3D-Hard 的嵌套列表结构。
    def _apply_mapping_recursive(self, data: any, mapping: dict) -> any:
        if isinstance(data, str):
            return "".join([mapping.get(char, char) for char in data])
        elif isinstance(data, list):
            return [self._apply_mapping_recursive(item, mapping) for item in data]
        else:
            return data # 其他类型（如数字）保持不变

    def generate_for_puzzle(self, question_record: dict) -> dict:
        task_type = question_record['task_type']
        puzzle_data = question_record['puzzle_data']
        
        # 中文注释：P0 代表原始谜题。
        p0_prompt, p0_answer, p0_question_text = self._generate_p0_original(puzzle_data)
        perturbations = {"P0_Original": {"prompt": p0_prompt, "answer": p0_answer, "question_text": p0_question_text}}

        # 中文注释：只有符号性规则的谜题才需要进行符号映射的扰动测试。
        if task_type == "SymbolicRule":
            perturbations["P1_SymbolMapping"] = self._generate_p1_symbol_mapping(puzzle_data)
            
        return perturbations

    def _generate_p0_original(self, puzzle_data: dict) -> tuple[str, any, any]:
        question_text = puzzle_data['question_plaintext']
        prompt_text = self._format_puzzle_text(
            puzzle_data['examples'], question_text
        )
        return prompt_text, puzzle_data['answer_ciphertext'], question_text

    def _generate_p1_symbol_mapping(self, puzzle_data: dict) -> dict:
        original_chars = string.ascii_letters + string.digits
        new_symbols = list("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ" + "çñòúàéíøæåëï")
        random.shuffle(new_symbols)
        
        mapping_dict = dict(zip(original_chars, new_symbols[:len(original_chars)]))
        
        mapped_examples = [
            {"input": self._apply_mapping_recursive(ex["input"], mapping_dict), 
             "output": self._apply_mapping_recursive(ex["output"], mapping_dict)}
            for ex in puzzle_data["examples"]
        ]
        
        mapped_question = self._apply_mapping_recursive(puzzle_data["question_plaintext"], mapping_dict)
        mapped_answer = self._apply_mapping_recursive(puzzle_data["answer_ciphertext"], mapping_dict)
        
        prompt_text = self._format_puzzle_text(mapped_examples, mapped_question)
        return {"prompt": prompt_text, "answer": mapped_answer, "question_text": mapped_question}

class Answerer:
    def __init__(self, client: LLMClient):
        self.client = client

    def get_answer(self, prompt: str) -> tuple[any, str]:
        system_prompt = PROMPTS["answerer"]["system"]
        user_prompt = PROMPTS["answerer"]["user"].format(puzzle_examples=prompt)
        response = self.client.make_request(system_prompt, user_prompt, temperature=1e-7) 
        
        action = response.get("action")
        if action and isinstance(action, dict):
            final_answer = action.get("final_answer", "Error: No answer provided.")
            reasoning = action.get("reasoning", "Error: No reasoning provided.")
            return final_answer, str(reasoning)
        else:
            return "Error: Invalid JSON response.", response.get("response_text", "")

def main():
    print(f"--- Starting ARC-Text Puzzle Evaluation (V9.0.1) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing LLM clients...")
    answerer_clients = {
        name: LLMClient(config["provider"], config["model"], max_tokens=10000, max_retries=1)
        for name, config in ANSWERER_MODELS.items()
    }
    judge_client = LLMClient(JUDGE_MODEL_CONFIG["provider"], JUDGE_MODEL_CONFIG["model"], max_tokens=15000, max_retries=10)
    judge = Judge(judge_client)
    
    perturbation_engine = PerturbationEngine()

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_questions = [json.loads(line) for line in f]
        print(f"Successfully loaded {len(all_questions)} validated puzzles from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"[Error] Input file not found: {INPUT_FILE}. Please run question.py (V9.3) first.")
        return

    completed_tasks = set()
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                task_id = (result['question_id'], result['answerer_model'], result['perturbation_type'])
                completed_tasks.add(task_id)
        print(f"Found {len(completed_tasks)} completed tasks. Resuming...")
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing results file found. Starting a new run.")

    total_evals = 0
    for q in all_questions:
        num_perturbations = 2 if q['task_type'] == 'SymbolicRule' else 1
        total_evals += num_perturbations * len(answerer_clients)

    current_eval = 0
    for question_record in all_questions:
        perturbations = perturbation_engine.generate_for_puzzle(question_record)
        
        for p_type, p_data in perturbations.items():
            for answerer_name, client in answerer_clients.items():
                current_eval += 1
                
                task_id = (question_record['question_id'], answerer_name, p_type)
                if task_id in completed_tasks:
                    print(f"  ({current_eval}/{total_evals}) SKIPPING task {task_id} - already completed.")
                    continue

                print(f"  ({current_eval}/{total_evals}) Evaluating task {task_id}...")
                
                answerer = Answerer(client)
                model_answer, model_cot = answerer.get_answer(p_data['prompt'])
                ground_truth = p_data['answer']

                print(f"    > Submitting to Judge...")
                is_correct, justification = judge.evaluate(
                    task_type=question_record['task_type'],
                    rule_description=question_record['rule_description'],
                    question_text=p_data['question_text'],
                    ground_truth=ground_truth,
                    model_answer=model_answer,
                    model_cot=model_cot
                )
                print(f"    > Judge Verdict: {is_correct}. Justification: {justification}")
                # 打印预览时限制长度，防止日志刷屏
                gt_preview = json.dumps(ground_truth)
                ans_preview = json.dumps(model_answer)
                if len(gt_preview) > 100: gt_preview = gt_preview[:100] + "..."
                if len(ans_preview) > 100: ans_preview = ans_preview[:100] + "..."
                print(f"    > GT: {gt_preview} | Model: {ans_preview}")
                
                result_record = {
                    "question_id": question_record['question_id'],
                    "author_model": question_record['author_model'],
                    "task_type": question_record['task_type'],
                    "dimensionality": question_record.get('dimensionality', 'N/A'), # 3D-Hard 字符串会被正常保存
                    "rule_description": question_record['rule_description'],
                    "answerer_model": answerer_name,
                    "perturbation_type": p_type,
                    "is_correct": is_correct,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer,
                    "model_cot": model_cot,
                    "judge_model": JUDGE_MODEL_CONFIG['model'],
                    "judge_justification": justification
                }

                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_record, ensure_ascii=True) + '\n')
                completed_tasks.add(task_id)
                
                time.sleep(1)

    print("\n--- Evaluation Complete ---")
    print(f"All {total_evals} targeted evaluations have been processed.")
    print(f"Raw results, including Judge verdicts, saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()