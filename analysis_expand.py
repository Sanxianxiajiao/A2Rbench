# analysis_expand.py (V2.2.2 - Fixed Task Counting for P1 Perturbations)

import os
import json
import pandas as pd
import numpy as np
import itertools
from scipy.stats import entropy
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gzip

# --- CONFIGURATION ---
# 输入文件
INPUT_RESULTS_FILE = os.path.join("results_arc_text_v9", "results_raw.jsonl")
INPUT_QUESTIONS_FILE = os.path.join("questions_arc_text_v9", "validated_arc_text_questions.jsonl")

# 输出目录
OUTPUT_DIR = "expand_analysis_v9"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---

def safe_json_dumps(obj):
    """将任何对象安全地转换为一个规范的JSON字符串，用于比较或编码。"""
    try:
        return json.dumps(obj, sort_keys=True)
    except TypeError:
        return str(obj)

def robust_load_jsonl(filepath):
    """使用健壮的方式从jsonl文件加载数据到DataFrame。"""
    print(f"Robustly loading data from: {filepath}")
    data_list = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Reading {os.path.basename(filepath)}"):
                if not line.strip():
                    continue
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping a malformed JSON line: {line[:100]}...")
    except FileNotFoundError:
        print(f"[FATAL] Input file not found at '{filepath}'. Please check the path.")
        return None
    
    if not data_list:
        print(f"[FATAL] No valid JSON data found in '{filepath}'.")
        return None
        
    df = pd.DataFrame(data_list)
    print(f"Loaded {len(df)} records.\n")
    return df
    
def get_version_from_id(question_id):
    """从question_id中安全地提取版本号，处理不含'_V'的原始ID。"""
    if '_V' in question_id:
        return int(question_id.rsplit('_V', 1)[1])
    else:
        # 如果ID不含'_V'，我们认定它是原始种子问题，版本为0。
        return 0

# --- MODEL COMPLETION FILTER ---
def filter_models_by_completion(df_results, df_questions):
    """
    根据模型是否完成了99%以上的问题（包含P0和P1变体）来过滤结果。
    """
    if df_results is None or df_questions is None:
        return None
        
    # *** MODIFICATION START: 使用 ID + Perturbation 作为唯一任务标识以处理 P1 变体 ***
    df_results['task_instance_key'] = df_results['question_id'].astype(str) + "_" + df_results['perturbation_type']
    
    # 总任务实例数应为结果集中所有出现的 (ID, Perturbation) 组合的并集
    total_task_instances = df_results['task_instance_key'].nunique()
    
    if total_task_instances == 0:
        print("Warning: No task instances found in results file. Cannot filter.")
        return df_results
        
    model_completion = df_results.groupby('answerer_model')['task_instance_key'].nunique()
    completion_ratio = model_completion / total_task_instances
    
    models_to_keep = completion_ratio[completion_ratio >= 0.99].index.tolist()
    
    print(f"Total unique task instances (including P0 + P1): {total_task_instances}")
    print(f"Found {len(completion_ratio)} models in results file.")
    print(f"Filtering for models with >= 99% completion. Keeping {len(models_to_keep)} models: {models_to_keep}")
    
    if len(models_to_keep) < len(completion_ratio):
        models_to_exclude = completion_ratio[completion_ratio < 0.99].index.tolist()
        print(f"Excluding {len(models_to_exclude)} models with < 99% completion: {models_to_exclude}")
    
    filtered_df = df_results[df_results['answerer_model'].isin(models_to_keep)].copy()
    
    if filtered_df.empty:
        print("\n[FATAL] No results remain after filtering for model completion. Aborting analysis for this scheme.")
        return None
        
    return filtered_df

# --- SCHEME 1: FAILURE MODE DIVERSITY ---

def analyze_failure_group(group):
    """计算一组失败答案的多样性指标。"""
    answers = group['model_answer'].apply(safe_json_dumps)
    if len(answers) <= 1:
        return pd.Series({'Unique_Errors': 1, 'Entropy': 0.0, 'Avg_Edit_Distance': 0.0})

    unique_error_count = answers.nunique()
    probabilities = answers.value_counts(normalize=True).values
    shannon_entropy = entropy(probabilities, base=2)
    
    unique_answers = answers.unique()
    if len(unique_answers) <= 1:
        avg_distance = 0.0
    else:
        pairs = list(itertools.combinations(unique_answers, 2))
        distances = [100 - fuzz.ratio(p1, p2) for p1, p2 in pairs]
        avg_distance = np.mean(distances) if distances else 0.0
        
    return pd.Series({'Unique_Errors': unique_error_count, 'Entropy': shannon_entropy, 'Avg_Edit_Distance': avg_distance})

def run_scheme_1_failure_diversity():
    """执行方案一：失败模式多样性分析。"""
    print("--- Running Scheme 1: Failure Mode Diversity Analysis ---")
    df_results = robust_load_jsonl(INPUT_RESULTS_FILE)
    df_questions = robust_load_jsonl(INPUT_QUESTIONS_FILE)
    
    # ** MODIFICATION: Filter models by completion rate **
    df = filter_models_by_completion(df_results, df_questions)
    if df is None: return

    df_failures = df[df['is_correct'] == False].copy()
    if df_failures.empty:
        print("No failure cases found for the filtered models. Scheme 1 analysis cannot be performed.")
        return
        
    print(f"Found {len(df_failures)} failure cases to analyze for Scheme 1 (from high-completion models).")
    df_failures['version'] = df_failures['question_id'].apply(get_version_from_id)
    
    tqdm.pandas(desc="Analyzing failure groups")
    diversity_metrics = df_failures.groupby('question_id').progress_apply(analyze_failure_group).reset_index()
    diversity_metrics = pd.merge(diversity_metrics, df_failures[['question_id', 'version']].drop_duplicates(), on='question_id')
    
    aggregated_results = diversity_metrics.groupby('version').mean(numeric_only=True).reset_index()
    aggregated_results = aggregated_results.sort_values('version')
    aggregated_results['version_str'] = 'V' + aggregated_results['version'].astype(str)
    
    print("\n--- [Scheme 1] Final Aggregated Results (for high-completion models) ---")
    print("This table shows the average 'ambiguity' (failure diversity) for each question version.")
    print(aggregated_results[['version_str', 'Unique_Errors', 'Entropy', 'Avg_Edit_Distance']].round(3))
    print("-" * 50)

# --- SCHEME 2: INFORMATION-THEORETIC COMPLEXITY ---

def calculate_compression_ratio(data):
    """计算给定数据的Gzip压缩率，作为信息熵的代理指标。"""
    serialized_str = safe_json_dumps(data)
    encoded_bytes = serialized_str.encode('utf-8')
    original_size = len(encoded_bytes)
    
    if original_size == 0:
        return 1.0
        
    compressed_bytes = gzip.compress(encoded_bytes)
    compressed_size = len(compressed_bytes)
    
    return compressed_size / original_size

def generate_compression_visualization(df_aggregated, output_dir):
    """为方案二（压缩率分析）生成并保存图表。"""
    print("\n--- Generating Visualization for Scheme 2 ---")
    sns.set_theme(style="whitegrid", palette="plasma")
    plt.figure(figsize=(12, 7))
    
    ax = sns.barplot(data=df_aggregated, x='version_str', y='Compression_Ratio')
    
    title = "Information Complexity Analysis: Avg. Gzip Compression Ratio\n(Higher ratio suggests more complexity/information, less predictability)"
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel("Question Version", fontsize=12)
    ax.set_ylabel("Average Compression Ratio", fontsize=12)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "information_complexity_analysis.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to: {output_path}")

def run_scheme_2_info_complexity():
    """执行方案二：信息论复杂度分析。"""
    print("\n--- Running Scheme 2: Information-Theoretic Complexity Analysis ---")
    expanded_questions_file = os.path.join("questions_arc_text_v9", "expanded_arc_text_questions.jsonl")
    if os.path.exists(expanded_questions_file):
        df = robust_load_jsonl(expanded_questions_file)
    else:
        print(f"Warning: '{expanded_questions_file}' not found. Falling back to '{INPUT_QUESTIONS_FILE}'.")
        print("Scheme 2 will only be able to analyze seed questions (V0).")
        df = robust_load_jsonl(INPUT_QUESTIONS_FILE)

    if df is None: return

    print(f"Analyzing information complexity for {len(df)} questions.")
    df['version'] = df['question_id'].apply(get_version_from_id)
    
    tqdm.pandas(desc="Calculating compression ratios")
    df['Compression_Ratio'] = df['puzzle_data'].apply(lambda x: calculate_compression_ratio(x.get('question_plaintext')))
    
    aggregated_results = df.groupby('version')['Compression_Ratio'].mean().reset_index()
    aggregated_results = aggregated_results.sort_values('version')
    aggregated_results['version_str'] = 'V' + aggregated_results['version'].astype(str)

    print("\n--- [Scheme 2] Final Aggregated Results ---")
    print("This table shows the average information complexity (Gzip ratio) for each question version.")
    print(aggregated_results[['version_str', 'Compression_Ratio']].round(4))
    print("-" * 50)
    
    generate_compression_visualization(aggregated_results, OUTPUT_DIR)

# --- NEW: SCHEME 3 - ACCURACY BY VERSION ---
def run_scheme_3_accuracy_by_version():
    """执行方案三：按版本号分析准确率。"""
    print("\n--- Running Scheme 3: Accuracy Analysis by Version ---")
    
    df_results = robust_load_jsonl(INPUT_RESULTS_FILE)
    df_questions = robust_load_jsonl(INPUT_QUESTIONS_FILE)

    # ** MODIFICATION: Filter models by completion rate **
    df = filter_models_by_completion(df_results, df_questions)
    if df is None: return

    # 从 question_id 中提取版本号
    df['version'] = df['question_id'].apply(get_version_from_id)

    # 按版本号分组并计算准确率 (is_correct列的均值)
    accuracy_by_version = df.groupby('version')['is_correct'].mean().reset_index()
    accuracy_by_version.rename(columns={'is_correct': 'Accuracy'}, inplace=True)
    
    # 将准确率转换为百分比格式
    accuracy_by_version['Accuracy'] = accuracy_by_version['Accuracy'] * 100
    
    # 排序并添加版本字符串用于显示
    accuracy_by_version = accuracy_by_version.sort_values('version')
    accuracy_by_version['version_str'] = 'V' + accuracy_by_version['version'].astype(str)

    # 打印最终结果
    print("\n--- [Scheme 3] Final Aggregated Results (for high-completion models) ---")
    print("This table shows the overall accuracy for each question version.")
    
    # 为了美观，格式化输出
    print_df = accuracy_by_version[['version_str', 'Accuracy']].copy()
    print_df['Accuracy'] = print_df['Accuracy'].apply(lambda x: f"{x:.2f}%")
    print(print_df.to_string(index=False))
    print("-" * 50)

# --- MAIN ORCHESTRATOR ---
def main():
    """主函数，协调并执行所有分析方案。"""
    
    # 方案一：失败模式多样性（歧义性）分析
    run_scheme_1_failure_diversity()
    
    # 方案二：信息论复杂度分析
    run_scheme_2_info_complexity()

    # 新增方案三：按版本计算准确率
    run_scheme_3_accuracy_by_version()
    
    print("\n--- Analysis Complete ---")
    print(f"Check the '{OUTPUT_DIR}' directory for the output plot from Scheme 2.")

if __name__ == "__main__":
    main()