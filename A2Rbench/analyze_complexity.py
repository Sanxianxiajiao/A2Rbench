# analyze_complexity.py (Version 3.4.3 - With Corrected Task Counting)

import os
import json
import ast
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
QUESTIONS_FILE = os.path.join("questions_arc_text_v9", "validated_arc_text_questions.jsonl")
RESULTS_FILE = os.path.join("results_arc_text_v9", "results_raw.jsonl")
OUTPUT_DIR = "complexity_analysis_v9"

# --- NEW: Robust Data Loading Function (FIX for the ValueError) ---
def load_jsonl_robustly(filepath: str) -> pd.DataFrame:
    """
    Reads a .jsonl file line by line, skipping any corrupted or empty lines.
    This prevents crashes from malformed JSON files.
    """
    print(f"Robustly loading data from: {filepath}")
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Only process non-empty lines
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  [Warning] Skipping a corrupted line in {filepath}")
    
    print(f"  Successfully loaded {len(records)} valid records.")
    return pd.DataFrame(records)

# --- AST Visitor for Comprehensive Code Pattern Analysis (V3.3 Logic, Unchanged) ---
class PatternVisitorV3_Comprehensive(ast.NodeVisitor):
    def __init__(self):
        self.function_loc = 0
        self.total_if_statements = 0
        self.local_variable_count = 0
        self.max_loop_depth = 0
        self.spatial_dependency_score = 0
        self.max_conditional_complexity = 0
        self.state_mutability_score = 0
        self.max_nested_conditional_depth = 0
        self.data_structure_creation_count = 0
        self.helper_function_count = 0
        self.return_complexity = 0
        self.concatenation_count = 0
        self.builtin_function_call_count = 0
        self._current_loop_depth = 0
        self._current_conditional_depth = 0
        self._loop_vars = set()
        self._main_function_visited = False
        self._builtins = set(dir(__builtins__))

    def _get_node_complexity(self, node):
        if node is None: return 0
        return 1 + sum(self._get_node_complexity(child) for child in ast.iter_child_nodes(node))

    def visit_FunctionDef(self, node):
        if node.name == 'transform_grid' and not self._main_function_visited:
            self._main_function_visited = True
            self.local_variable_count = len({
                n.id for n in ast.walk(node) 
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            })
            self.generic_visit(node)
        elif self._main_function_visited:
            self.helper_function_count += 1

    def visit_For(self, node):
        if isinstance(node.target, ast.Name): self._loop_vars.add(node.target.id)
        self._current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self._current_loop_depth)
        self.generic_visit(node)
        self._current_loop_depth -= 1
        if isinstance(node.target, ast.Name): self._loop_vars.remove(node.target.id)

    def visit_If(self, node):
        self.total_if_statements += 1
        self._current_conditional_depth += 1
        self.max_nested_conditional_depth = max(self.max_nested_conditional_depth, self._current_conditional_depth)
        self.max_conditional_complexity = max(self.max_conditional_complexity, self._get_node_complexity(node.test))
        self.generic_visit(node)
        self._current_conditional_depth -= 1

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Subscript): self.state_mutability_score += 1
        self.generic_visit(node)

    def visit_Subscript(self, node):
        index_node = getattr(node.slice, 'value', node.slice)
        for sub_node in ast.walk(index_node):
            if isinstance(sub_node, ast.BinOp) and any(isinstance(n, ast.Name) and n.id in self._loop_vars for n in ast.walk(sub_node)):
                self.spatial_dependency_score += 1
                break
        self.generic_visit(node)

    def visit_Dict(self, node): self.data_structure_creation_count += 1; self.generic_visit(node)
    def visit_Set(self, node): self.data_structure_creation_count += 1; self.generic_visit(node)
    def visit_ListComp(self, node): self.data_structure_creation_count += 1; self.generic_visit(node)
    def visit_SetComp(self, node): self.data_structure_creation_count += 1; self.generic_visit(node)
    def visit_DictComp(self, node): self.data_structure_creation_count += 1; self.generic_visit(node)
    def visit_Return(self, node): self.return_complexity = self._get_node_complexity(node.value)
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add): self.concatenation_count += 1
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self._builtins:
            self.builtin_function_call_count += 1
        self.generic_visit(node)

# --- Complexity & Fingerprint Analysis Functions ---

def analyze_code_complexity(code_string: str) -> dict:
    """Analyzes Python code using AST to extract complexity metrics."""
    try:
        tree = ast.parse(code_string)
        visitor = PatternVisitorV3_Comprehensive()
        visitor.visit(tree)
        return {
            "max_loop_depth": visitor.max_loop_depth,
            "total_if_statements": visitor.total_if_statements,
            "local_variable_count": visitor.local_variable_count,
            "spatial_dependency_score": visitor.spatial_dependency_score,
            "max_conditional_complexity": visitor.max_conditional_complexity,
            "state_mutability_score": visitor.state_mutability_score,
            "max_nested_conditional_depth": visitor.max_nested_conditional_depth,
            "data_structure_creation_count": visitor.data_structure_creation_count,
            "helper_function_count": visitor.helper_function_count,
            "return_complexity": visitor.return_complexity,
            "concatenation_count": visitor.concatenation_count,
            "builtin_function_call_count": visitor.builtin_function_call_count,
        }
    except Exception:
        return {} # Return empty dict on parsing error

def classify_rule_style(code_string: str, complexity_metrics: dict) -> str:
    """
    Classifies the 'style' of a rule based on its code patterns.
    This is the core of the "Author Fingerprint" analysis.
    """
    if complexity_metrics.get("max_loop_depth", 0) >= 2:
        return "Spatial_Transformation"
    
    sequence_patterns = r"\[::-1\]|\.join\(|\.split\(|reversed\("
    if re.search(sequence_patterns, code_string):
        return "Sequence_Manipulation"

    if complexity_metrics.get("max_conditional_complexity", 0) > 10 or \
       (code_string.count('+') + code_string.count('-') + code_string.count('*')) > 3:
        return "Arithmetic_Logical"

    if complexity_metrics.get("data_structure_creation_count", 0) > 4:
         return "Data_Structure_Heavy"

    return "General_Other"

def create_analysis_plot(df, metric, output_dir, title_prefix):
    """Generates and saves a box plot for a given metric."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    df['dimensionality'] = df['dimensionality'].astype(str)
    dim_order = sorted(df['dimensionality'].unique(), key=lambda x: (x.isdigit(), x))
    
    sns.boxplot(data=df, x='dimensionality', y=metric, hue='is_correct', order=dim_order, palette='Set2')
    
    plt.title(f'{title_prefix}: "{metric}" by Puzzle Outcome', fontsize=16, weight='bold')
    plt.xlabel("Puzzle Dimensionality", fontsize=12)
    plt.ylabel(f"Value of {metric}", fontsize=12)
    plt.legend(title='Answer Correctness', loc='upper right')
    plt.tight_layout()
    
    filename = f"{title_prefix.lower().replace(' ', '_')}_{metric}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Saved plot to: {os.path.join(output_dir, filename)}")

# --- Main Orchestrator ---
def main():
    print("--- Starting Comprehensive Analysis (V3.4.3 - Fixed Task Counting) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)

    # --- 1. Data Loading & Filtering ---
    try:
        df_questions = load_jsonl_robustly(QUESTIONS_FILE)
        df_results = load_jsonl_robustly(RESULTS_FILE)
    except FileNotFoundError as e:
        print(f"[FATAL] Input file not found: {e}.")
        return

    # *** MODIFICATION START: Filter models that completed >= 99% of tasks (P0 + P1) ***
    # 创建任务实例唯一标识：ID + 扰动类型 (保证分母包含映射变体)
    df_results['task_instance_key'] = df_results['question_id'].astype(str) + "_" + df_results['perturbation_type']
    
    # 总任务数应为结果集中所有出现的 (ID, Perturbation) 组合的总数
    total_task_instances = df_results['task_instance_key'].nunique()
    
    if total_task_instances > 0:
        # 计算每个模型完成的唯一任务实例数
        model_completion = df_results.groupby('answerer_model')['task_instance_key'].nunique()
        completion_ratio = model_completion / total_task_instances
        models_to_keep = completion_ratio[completion_ratio >= 0.99].index.tolist()
        
        print(f"\nTotal unique task instances (including P0 + P1): {total_task_instances}")
        print(f"Found {len(models_to_keep)} models with >= 99% completion: {models_to_keep}")
        
        df_results_filtered = df_results[df_results['answerer_model'].isin(models_to_keep)].copy()
        
        if df_results_filtered.empty:
            print("[FATAL] No data remains after filtering for high-completion models. Aborting.")
            return
        print(f"Proceeding with {len(df_results_filtered)} records from high-completion models.\n")
    else:
        print("\nWarning: No task instances found. Skipping model completion filter.\n")
        df_results_filtered = df_results
    # *** MODIFICATION END ***
    
    # --- 2. Data Integration ---
    df_full = pd.merge(df_results_filtered, df_questions, on='question_id', suffixes=('_res', '_q'))
    df_full['dimensionality'] = df_full['dimensionality_q']
    
    print(f"Successfully merged {len(df_full)} records for analysis.")

    # --- 3. Feature Engineering (Complexity, Fingerprint, Efficiency) ---
    print("Analyzing unique puzzles for complexity and rule style...")
    unique_questions = df_questions.drop_duplicates(subset=['question_id'])
    
    complexity_metrics_list = unique_questions['python_code'].apply(analyze_code_complexity)
    df_complexity = pd.json_normalize(complexity_metrics_list)
    
    temp_df = pd.concat([unique_questions.reset_index(drop=True), df_complexity], axis=1)
    
    temp_df['rule_style'] = temp_df.apply(
        lambda row: classify_rule_style(row['python_code'], row.to_dict()), axis=1
    )
    
    df_questions_with_features = temp_df
    
    cols_to_merge = ['question_id', 'rule_style'] + list(df_complexity.columns)
    df_final = pd.merge(df_full, df_questions_with_features[cols_to_merge], on='question_id')
    
    df_final['cot_length'] = df_final['model_cot'].fillna('').astype(str).str.len()
    
    metric_cols = list(df_complexity.columns)

    # --- 4. Analysis & Reporting ---
    
    print("\n" + "="*80)
    print("PART A: Author Model Bias (Complexity of Generated Rules)")
    print("="*80)
    author_bias_analysis = df_questions_with_features.groupby(['author_model', 'dimensionality'])[metric_cols].mean()
    print(author_bias_analysis.round(2))

    print("\n" + "="*80)
    print("PART B: Author Fingerprint Analysis (Style of Generated Rules)")
    print("="*80)
    author_style_crosstab = pd.crosstab(
        df_questions_with_features['author_model'],
        df_questions_with_features['rule_style'],
        normalize='index'
    ).applymap('{:.2%}'.format)
    print("B1: Author Model Style Preferences (Distribution of Rule Types):\n")
    print(author_style_crosstab)
    
    # This part now uses the filtered data
    solver_style_performance = df_final.groupby(['answerer_model', 'rule_style'])['is_correct'].mean().unstack(fill_value=0)
    print("\nB2: Solver Model Accuracy on Different Rule Styles (High-Completion Models Only):\n")
    print(solver_style_performance.applymap('{:.2%}'.format))

    print("\n" + "="*80)
    print("PART C: Fine-Grained Failure Cases (Complexity vs. Outcome, High-Completion Models Only)")
    print("="*80)
    # This part now uses the filtered data
    failure_analysis_complexity = df_final.groupby(['dimensionality', 'is_correct'])[metric_cols].mean()
    print(failure_analysis_complexity.round(2))

    print("\n" + "="*80)
    print("PART D: Reasoning Efficiency Analysis (CoT Length vs. Outcome, High-Completion Models Only)")
    print("="*80)
    # This part now uses the filtered data
    efficiency_analysis = df_final.groupby(['answerer_model', 'is_correct'])['cot_length'].describe()
    print(efficiency_analysis.round(2))

    # --- 5. Visualization ---
    print("\n" + "="*80)
    print("PART E: Generating Visualizations (High-Completion Models Only)")
    print("="*80)
    
    key_metrics_to_plot = [
        "spatial_dependency_score", "max_conditional_complexity", 
        "state_mutability_score", "max_nested_conditional_depth",
        "return_complexity", "max_loop_depth"
    ]
    for metric in key_metrics_to_plot:
        if metric in df_final.columns:
             create_analysis_plot(df_final, metric, OUTPUT_DIR, "Complexity Failure Analysis")
    
    if 'cot_length' in df_final.columns:
        create_analysis_plot(df_final, 'cot_length', OUTPUT_DIR, "Reasoning Efficiency Analysis")

    print("\n--- Analysis Complete ---")
    print(f"Check the '{OUTPUT_DIR}' directory for output plots.")

if __name__ == "__main__":
    main()