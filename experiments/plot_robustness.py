import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_robustness_chart():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 12
    
    df = pd.read_csv("data/robustness_results.csv")
    summary = df.groupby(['Algorithm', 'Condition']).mean(numeric_only=True).reset_index()
    
    algorithms = ['Baseline-Rule', 'Baseline-GA', 'Proposed-DRL-HIO']
    conditions = ['Normal', 'Breakdown (300s)']
    
    normal_spans = [summary[(summary['Algorithm']==alg) & (summary['Condition']=='Normal')]['Make_span'].values[0] for alg in algorithms]
    breakdown_spans = [summary[(summary['Algorithm']==alg) & (summary['Condition']=='Breakdown (300s)')]['Make_span'].values[0] for alg in algorithms]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(algorithms))
    width = 0.35

    bars1 = ax.bar(x - width/2, normal_spans, width, label='Normal Execution', color='#2ca02c', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, breakdown_spans, width, label='With Machine Breakdown', color='#d62728', edgecolor='black', alpha=0.8)

    ax.set_ylabel('System Make-span (s)')
    ax.set_title('Robustness Analysis under Sudden Machine Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    for i in range(len(algorithms)):
        degradation = ((breakdown_spans[i] - normal_spans[i]) / normal_spans[i]) * 100
        ax.text(x[i] + width/2, breakdown_spans[i] + 10, f"+{degradation:.1f}%", ha='center', va='bottom', fontweight='bold', color='darkred')

    plt.tight_layout()
    os.makedirs("data/figures", exist_ok=True)
    plt.savefig("data/figures/fig_robustness.png", dpi=300)
    print("File saved to: data/figures/fig_robustness.png")

if __name__ == "__main__":
    plot_robustness_chart()