# analysis_comprehensive.py (V9.6 - Added Symbolic Dependency, kept others same)

import pandas as pd
import json
import os
import numpy as np
from collections import defaultdict

# --- CONFIGURATION (与原版一致) ---
INPUT_FILE_COT = os.path.join("results_arc_text_v9", "results_raw.jsonl")
INPUT_FILE_NO_COT = os.path.join("results_arc_text_v9_nocot", "results_raw.jsonl")
OUTPUT_MARKDOWN_FILE = os.path.join("results_arc_text_v9", "analysis_final_report.md")

# 严格筛选：必须跑完 99% 以上的题目才计入统计
COMPLETION_THRESHOLD = 0.99

def load_jsonl_robust(filepath):
    print(f"Loading {filepath} ...")
    data = []
    if not os.path.exists(filepath):
        print(f"[WARNING] File not found: {filepath}")
        return pd.DataFrame()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: pass
    return pd.DataFrame(data)

def preprocess_data(df):
    if df.empty: return df
    df['is_seed'] = df['question_id'].apply(lambda x: str(x).endswith('_V0'))
    df['dimensionality'] = df['dimensionality'].astype(str)
    return df

def filter_incomplete_models(df):
    if df.empty: return df
    counts = df.groupby(['answerer_model', 'reasoning_method']).size().reset_index(name='count')
    max_q = counts['count'].max()
    threshold = max_q * COMPLETION_THRESHOLD
    
    print("\n" + "="*50)
    print(f"COMPLETION CHECK (Threshold: {int(threshold)})")
    print(f"{'Model':<25} | {'Method':<8} | {'Count':<6} | {'Status'}")
    print("-" * 50)
    
    valid_keys = set()
    for _, row in counts.iterrows():
        m, met, c = row['answerer_model'], row['reasoning_method'], row['count']
        if c >= threshold:
            print(f"{m.replace('Answerer_', ''):<25} | {met:<8} | {c:<6} | ✅ PASS")
            valid_keys.add((m, met))
        else:
            print(f"{m.replace('Answerer_', ''):<25} | {met:<8} | {c:<6} | ❌ FAIL")
    print("="*50 + "\n")
    
    mask = df.apply(lambda x: (x['answerer_model'], x['reasoning_method']) in valid_keys, axis=1)
    return df[mask].copy()

def calc_acc(df):
    return (df['is_correct'].sum() / len(df) * 100) if len(df) > 0 else 0.0

# --- 原有的核心指标生成逻辑 (保持不变) ---
def generate_metrics_row(group, author_list):
    row = {}
    row['Total Acc'] = calc_acc(group)
    group['model_cot'] = group['model_cot'].fillna("")
    crashes = group[group['model_cot'].str.contains("Exceeded max retries|Error:", na=False)]
    row['Collapse'] = (len(crashes) / len(group) * 100) if len(group) > 0 else 0
    row['Sym Acc'] = calc_acc(group[group['task_type'] == 'SymbolicRule'])
    row['Sem Acc'] = calc_acc(group[group['task_type'] == 'SemanticRule'])
    seed_df = group[group['is_seed']]
    exp_df = group[~group['is_seed']]
    row['Seed Acc'] = calc_acc(seed_df)
    row['Exp Acc'] = calc_acc(exp_df)
    row['Gap'] = row['Seed Acc'] - row['Exp Acc']
    for auth in author_list:
        auth_sub = group[group['author_model'] == auth]
        row[f"vs {auth.replace('Author_', '')}"] = calc_acc(auth_sub)
    return row

# --- 原有的表格写入逻辑 (保持不变) ---
def write_markdown_table(f, df_clean, title):
    f.write(f"### {title}\n\n")
    authors = sorted(df_clean['author_model'].unique())
    cols = ["Model", "Method", "Total Acc", "Sym Acc", "Sem Acc", "Seed Acc", "Exp Acc", "Gap", "Collapse"]
    cols += [f"vs {a.replace('Author_', '')}" for a in authors]
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "---|" * len(cols) + "\n")
    
    groups = df_clean.groupby(['answerer_model', 'reasoning_method'])
    rows_data = []
    for (model, method), group in groups:
        metrics = generate_metrics_row(group, authors)
        metrics['Model'] = f"**{model.replace('Answerer_', '')}**"
        metrics['Method'] = method
        rows_data.append(metrics)
    
    rows_data.sort(key=lambda x: x['Total Acc'], reverse=True)
    for r in rows_data:
        gap_val = r['Gap']
        gap_str = f"{gap_val:.1f}%"
        if gap_val > 10: gap_str = f"**{gap_str}**"
        
        line = [
            r['Model'], r['Method'], f"{r['Total Acc']:.1f}%",
            f"{r['Sym Acc']:.1f}%", f"{r['Sem Acc']:.1f}%",
            f"{r['Seed Acc']:.1f}%", f"{r['Exp Acc']:.1f}%", gap_str,
            f"{r['Collapse']:.1f}%"
        ]
        for auth in authors:
            key = f"vs {auth.replace('Author_', '')}"
            val = r.get(key, 0.0)
            line.append(f"{val:.1f}%")
        f.write("| " + " | ".join(line) + " |\n")
    f.write("\n")

# --- 新增：符号依赖分析表 ---
def write_symbol_dependency_table(f, df_clean):
    f.write("## 2. Symbolic Dependency Analysis\n")
    f.write("> **Symbolic Dependency** = (Original Acc - Mapped Acc). A higher value suggests the model relies on common alphanumeric patterns rather than pure logic.\n\n")
    
    df_sym = df_clean[df_clean['task_type'] == 'SymbolicRule']
    if df_sym.empty:
        f.write("No SymbolicRule tasks found.\n\n")
        return

    cols = ["Model", "Method", "Original Acc (P0)", "Mapped Acc (P1)", "Symbolic Dependency"]
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "---|" * len(cols) + "\n")
    
    groups = df_sym.groupby(['answerer_model', 'reasoning_method'])
    rows_data = []
    for (model, method), group in groups:
        acc_p0 = calc_acc(group[group['perturbation_type'] == 'P0_Original'])
        acc_p1 = calc_acc(group[group['perturbation_type'] == 'P1_SymbolMapping'])
        dependency = acc_p0 - acc_p1
        rows_data.append({
            "Model": f"**{model.replace('Answerer_', '')}**",
            "Method": method, "P0": acc_p0, "P1": acc_p1, "Dep": dependency
        })
    
    # 按依赖度从高到低排序
    rows_data.sort(key=lambda x: x['Dep'], reverse=True)
    for r in rows_data:
        dep_str = f"{r['Dep']:.1f}%"
        if r['Dep'] > 15: dep_str = f"**{dep_str}**"
        f.write(f"| {r['Model']} | {r['Method']} | {r['P0']:.1f}% | {r['P1']:.1f}% | {dep_str} |\n")
    f.write("\n")

# --- 主逻辑 (结构微调以插入新表格) ---
def main():
    print("--- Starting Comprehensive Analysis (V9.6) ---")
    df_cot = load_jsonl_robust(INPUT_FILE_COT)
    if not df_cot.empty: df_cot['reasoning_method'] = 'CoT'
    df_nocot = load_jsonl_robust(INPUT_FILE_NO_COT)
    if not df_nocot.empty: df_nocot['reasoning_method'] = 'Direct'
    
    if df_cot.empty and df_nocot.empty: return
    df = pd.concat([df_cot, df_nocot], ignore_index=True)
    df = preprocess_data(df)
    df_clean = filter_incomplete_models(df)
    
    if df_clean.empty: return

    with open(OUTPUT_MARKDOWN_FILE, 'w', encoding='utf-8') as f:
        f.write("# ARC-Text Final Comprehensive Report\n\n")
        f.write("> **Note:** Only models that completed >99% of tasks are shown.\n\n")
        
        # 1. 全局表现 (保持不变)
        f.write("## 1. Global Performance\n")
        write_markdown_table(f, df_clean, "All Dimensions Combined")
        
        # 2. 符号依赖 (新加入)
        write_symbol_dependency_table(f, df_clean)
        
        # 3. 按维度分析 (保持不变)
        f.write("## 3. Performance by Dimensionality\n")
        dims = sorted(df_clean['dimensionality'].unique(), key=lambda x: (str(x).isdigit(), x))
        for dim in dims:
            dim_df = df_clean[df_clean['dimensionality'] == dim]
            if len(dim_df) > 0:
                write_markdown_table(f, dim_df, f"Dimensionality: {dim}")
                
    print(f"\nReport generated: {OUTPUT_MARKDOWN_FILE}")

if __name__ == "__main__":
    main()