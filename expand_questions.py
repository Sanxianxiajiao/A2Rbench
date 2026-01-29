# expand_questions.py (V9.3.1 - Robust Data Augmentation)

import os
import json
import copy
import time
from collections import defaultdict
from llm_client import LLMClient

# --- CONFIGURATION ---
INPUT_FILE = os.path.join("questions_arc_text_v10", "validated_arc_text_questions.jsonl")
OUTPUT_DIR = "questions_arc_text_v10"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "expanded_arc_text_questions.jsonl")

# 推荐使用逻辑能力较强的模型进行数据扩充 (如 GPT-4o, GPT-5-mini)
EXPANDER_MODEL_CONFIG = {
    "provider": "all",
    "model": "gpt-5-mini", 
    "max_tokens": 4096
}

TARGET_VARIATIONS = 9 # 除了 V0 之外，尝试生成的最大变体数量 (V1-V9)
MAX_RETRIES_PER_STEP = 3 # 格式错误或校验失败时的重试次数

# --- PROMPT TEMPLATES ---

PROMPT_EXPAND_INPUT = """
**Role:** You are a Lead QA Engineer specializing in Abstract Logic Puzzles.
**Task:** We have a specific transformation rule and its Python implementation. Your job is to generate **NEW, DIVERSE input test cases** for this rule to verify its robustness and the solver's generalization capability.

---
**Rule Information:**
*   **Dimensionality:** {dimensionality}
*   **Rule Description:** {rule_description}

**Reference Code (Do not modify, use logic for reference):**
```python
{python_code}
```

**Existing Input History (DO NOT REPEAT THESE):**
{input_history}

---
**Current Objective (Variation {variation_index}/9):**
{variation_guidance}

**Constraints:**
1.  **Validity:** The new input MUST be valid for the provided code. (e.g., if the code expects a 3x3 grid, don't provide a string).
2.  **Diversity:** The new input must be strictly different from any in the History.
3.  **Low Entropy Check:** If the rule is extremely restrictive (e.g., "Input must be exactly string 'ABC'"), and you cannot generate anymore unique, valid inputs, you MUST return status "SKIPPED_LOW_ENTROPY".

**Output Format (Strict JSON):**
```json
{{
  "reasoning": "Explain your strategy for this test case (e.g., testing empty edge case).",
  "variation_type": "Label for this case (e.g., 'Edge_Case_Empty', 'Complexity_High', 'Adversarial_Pattern')",
  "new_input": <YOUR_NEW_INPUT_HERE>,
  "status": "CONTINUE"  // or "SKIPPED_LOW_ENTROPY" if impossible to generate new unique inputs
}}
```
"""

# 分阶段生成策略
VARIATION_STRATEGIES = {
    "early": "Generate a **Standard Variation**. Change the content but keep structure similar to original. Focus on verifying general logic.",
    "mid": "Generate an **Edge Case**. Think about: Empty structures, single elements, repetitive characters, max/min reasonable lengths.",
    "late": "Generate a **Complex or Adversarial Case**. Combine edge cases, use 'tricky' characters that might confuse the logic (if applicable), or create larger/denser inputs."
}

class QuestionExpander:
    def __init__(self):
        self.client = LLMClient(
            EXPANDER_MODEL_CONFIG["provider"],
            EXPANDER_MODEL_CONFIG["model"],
            max_tokens=EXPANDER_MODEL_CONFIG["max_tokens"]
        )

    def _get_strategy_text(self, index):
        if index <= 3: return VARIATION_STRATEGIES["early"]
        if index <= 6: return VARIATION_STRATEGIES["mid"]
        return VARIATION_STRATEGIES["late"]

    def _validate_and_execute(self, python_code, new_input):
        """
        利用原始代码验证新输入的合法性和循环一致性。
        使用 deepcopy 防止原地修改导致的数据污染。
        """
        scope = {}
        try:
            # 1. 动态编译并执行原始代码以获取函数定义
            exec(python_code, scope)
            transform_func = scope.get('transform_grid')
            inverse_func = scope.get('inverse_transform_grid')

            if not transform_func or not inverse_func:
                return False, None, "Missing functions in code."

            # 2. Forward Transform (Deepcopy 输入，防止函数修改原数据)
            try:
                input_copy_for_fwd = copy.deepcopy(new_input) 
                answer_ciphertext = transform_func(input_copy_for_fwd)
            except Exception as e:
                return False, None, f"Forward Execution Error: {e}"

            # 3. Inverse Transform (Cycle Check)
            try:
                input_copy_for_inv = copy.deepcopy(answer_ciphertext)
                reconstructed_input = inverse_func(input_copy_for_inv)
            except Exception as e:
                return False, None, f"Inverse Execution Error: {e}"

            # 4. Equality Check (验证是否无损还原)
            if new_input != reconstructed_input:
                # 截断过长的错误信息
                input_str = str(new_input)
                rec_str = str(reconstructed_input)
                if len(input_str) > 50: input_str = input_str[:50] + "..."
                if len(rec_str) > 50: rec_str = rec_str[:50] + "..."
                return False, None, f"Cycle Mismatch: {input_str} != {rec_str}"
            print('\nvalidate_and_execute\n')
            return True, answer_ciphertext, None

        except Exception as e:
            return False, None, f"Code Parsing/System Error: {e}"

    def process_file(self):
        print(f"--- Starting Data Augmentation (V9.3.1 - Robust Resume) ---")
        print(f"Reading Input: {INPUT_FILE}")
        
        # 1. 加载已有进度 (细粒度断点续传)
        # 结构: existing_variations[parent_id] = {0, 1, 2, ...}
        existing_variations = defaultdict(set)
        
        if os.path.exists(OUTPUT_FILE):
            print(f"Found existing output file. Loading progress...")
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        qid = rec.get("question_id", "")
                        # 解析 ID 结构: "XXX_V0", "XXX_V5"
                        if "_V" in qid:
                            parts = qid.rsplit("_V", 1)
                            parent_id = parts[0]
                            try:
                                v_index = int(parts[1])
                                existing_variations[parent_id].add(v_index)
                            except ValueError: pass
                    except: pass
            print(f"Resuming: Found progress for {len(existing_variations)} parent tasks.")

        # 2. 读取原始问题
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                all_questions = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"[Fatal Error] Input file {INPUT_FILE} not found.")
            return

        # 3. 主循环
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            for q_idx, original_record in enumerate(all_questions):
                original_id = original_record['question_id']
                current_existing = existing_variations[original_id]

                # 检查是否全部完成 (0 到 TARGET_VARIATIONS)
                if len(current_existing) >= TARGET_VARIATIONS + 1:
                    # 如果这题已经满了10个变体，跳过
                    continue

                print(f"\n[{q_idx+1}/{len(all_questions)}] Expanding Task: {original_id}")
                
                # --- Step A: 处理种子 V0 (如果不存在) ---
                if 0 not in current_existing:
                    v0_record = copy.deepcopy(original_record)
                    v0_record['parent_id'] = None # V0 没有父节点
                    v0_record['question_id'] = f"{original_id}_V0"
                    v0_record['variation_type'] = "Seed_Original"
                    
                    f_out.write(json.dumps(v0_record, ensure_ascii=False) + '\n')
                    f_out.flush()
                    current_existing.add(0)
                    print("  > Saved V0 (Seed)")
                
                # 准备 Prompt 用的历史记录 (包含 V0)
                input_history = [original_record['puzzle_data']['question_plaintext']]
                python_code_ref = original_record['python_code']
                
                # --- Step B: 生成 V1 - V9 ---
                low_entropy_stop = False
                
                for i in range(1, TARGET_VARIATIONS + 1):
                    # 如果该变体已存在，跳过
                    if i in current_existing:
                        continue
                    
                    # 如果之前的变体判定该规则熵过低，则停止生成后续变体
                    if low_entropy_stop:
                        break
                    
                    print(f"  > Generating V{i}...", end="", flush=True)
                    
                    # 构建 Prompt，只传入最近的5个历史输入以节省 Token
                    user_prompt = PROMPT_EXPAND_INPUT.format(
                        dimensionality=original_record['dimensionality'],
                        rule_description=original_record['rule_description'],
                        python_code=python_code_ref,
                        input_history=json.dumps(input_history[-5:], ensure_ascii=False), 
                        variation_index=i,
                        variation_guidance=self._get_strategy_text(i)
                    )

                    valid_variation = None
                    
                    # 重试循环 (用于处理 JSON 格式错误或校验失败)
                    for attempt in range(MAX_RETRIES_PER_STEP):
                        response = self.client.make_request(
                            "You are a strict QA Test Engineer.", 
                            user_prompt, 
                            temperature=0.5 # 较高的温度鼓励多样性
                        )
                        
                        action = response.get("action")
                        if not action:
                            print(".", end="", flush=True)
                            continue
                        
                        # 1. 检查 Low Entropy 信号
                        if action.get("status") == "SKIPPED_LOW_ENTROPY":
                            print(" [SKIPPED - Low Entropy Rule]")
                            low_entropy_stop = True
                            break
                        
                        new_input = action.get("new_input")
                        if new_input is None: 
                            continue

                        # 2. 严格代码校验 (The Gold Standard)
                        is_valid, answer_output, err = self._validate_and_execute(python_code_ref, new_input)
                        
                        if is_valid:
                            # 3. 简单查重 (避免和历史记录完全一样)
                            if new_input in input_history:
                                print(" [Dup]", end="", flush=True)
                                continue
                                
                            valid_variation = {
                                "input": new_input,
                                "output": answer_output,
                                "reasoning": action.get("reasoning", ""),
                                "variation_type": action.get("variation_type", f"Variation_V{i}")
                            }
                            break # 成功拿到一个有效变体，跳出重试循环
                        else:
                            # 校验失败 (LLM 生成的输入导致报错或不可逆)
                            # print(f"(Err: {err})", end="", flush=True)
                            pass
                    
                    # 如果是因为熵过低跳出重试，则也跳出 V 循环
                    if low_entropy_stop:
                        break

                    # 如果拿到了有效变体，保存
                    if valid_variation:
                        print(f" [OK] {valid_variation['variation_type']}")
                        
                        new_record = copy.deepcopy(original_record)
                        new_record['question_id'] = f"{original_id}_V{i}"
                        new_record['parent_id'] = original_id
                        new_record['variation_type'] = valid_variation['variation_type']
                        new_record['expansion_reasoning'] = valid_variation['reasoning']
                        
                        # 更新具体的题目数据
                        new_record['puzzle_data']['question_plaintext'] = valid_variation['input']
                        new_record['puzzle_data']['answer_ciphertext'] = valid_variation['output']
                        
                        # 实时写入磁盘
                        f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                        f_out.flush()
                        
                        # 加入历史
                        input_history.append(valid_variation['input'])
                    else:
                        print(" [Fail]") # 重试多次后依然失败，本轮 V(i) 放弃，尝试 V(i+1) 或继续

        print("\n--- Data Augmentation Complete ---")
        print(f"Expanded dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    expander = QuestionExpander()
    expander.process_file()