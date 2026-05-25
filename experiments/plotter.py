import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def set_ieee_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = '--'

def plot_experimental_results(csv_path=None, save_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if csv_path is None:
        csv_path = os.path.join(project_root, "data", "experiment_results.csv")
    if save_dir is None:
        save_dir = os.path.join(project_root, "data", "figures")
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found {csv_path}")
        return

    set_ieee_style()

    style_map = {
        'Baseline-Rule': {'color': '#1f77b4', 'label': 'Baseline (Rule)'},
        'Baseline-GA': {'color': '#8c564b', 'label': 'Baseline (GA)'},
        'Ablation-NoAStar': {'color': '#2ca02c', 'label': 'Ablation (w/o A*)'},
        'Ablation-NoShield': {'color': '#ff7f0e', 'label': 'Ablation (w/o Shield)'},
        'Proposed-CDRL-HIO': {'color': '#d62728', 'label': 'Proposed DRL-HIO'}
    }

    algorithms = ['Baseline-Rule', 'Baseline-GA', 'Ablation-NoAStar', 'Ablation-NoShield', 'Proposed-DRL-HIO']
    algorithms = [alg for alg in algorithms if alg in df['Algorithm'].unique()]
    
    task_loads = sorted(df['Task_Load'].unique())
    x = np.arange(len(task_loads))  
    total_algs = len(algorithms)
    width = 0.15  

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    for i, alg in enumerate(algorithms):
        alg_data = df[df['Algorithm'] == alg]
        y_values = []
        for load in task_loads:
            val = alg_data[alg_data['Task_Load'] == load]['Make_span_Mean'].values
            y_values.append(val[0] if len(val) > 0 else 0)
            
        offset = (i - total_algs / 2 + 0.5) * width
        ax1.bar(x + offset, y_values, width, 
                color=style_map[alg]['color'], 
                edgecolor='black',
                alpha=0.9,
                label=style_map[alg]['label'])

    ax1.set_xlabel('Task Load (Number of global tasks)')
    ax1.set_ylabel('Make-span (s)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_loads)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    makespan_png_path = os.path.join(save_dir, "fig_makespan.png")
    fig1.savefig(makespan_png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f"File saved to: {makespan_png_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    for i, alg in enumerate(algorithms):
        alg_data = df[df['Algorithm'] == alg]
        y_values = []
        for load in task_loads:
            val = alg_data[alg_data['Task_Load'] == load]['Deadlocks_Mean'].values
            y_values.append(val[0] if len(val) > 0 else 0)
            
        offset = (i - total_algs / 2 + 0.5) * width
        ax2.bar(x + offset, y_values, width, 
                color=style_map[alg]['color'], 
                edgecolor='black',
                alpha=0.9,
                label=style_map[alg]['label'])

    ax2.set_xlabel('Task Load (Number of global tasks)')
    ax2.set_ylabel('Conflict Interventions / Deadlocks')
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_loads)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    deadlocks_png_path = os.path.join(save_dir, "fig_deadlocks.png")
    fig2.savefig(deadlocks_png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print(f"File saved to: {deadlocks_png_path}")

    max_load = df['Task_Load'].max()
    df_time = df[df['Task_Load'] == max_load].copy()
    
    time_col = 'Avg_Decision_Time_ms'
    if time_col not in df_time.columns and 'Time_ms' in df_time.columns:
        df_time[time_col] = df_time['Time_ms'].astype(float)
        
    if not df_time.empty and time_col in df_time.columns:
        df_time['Algorithm'] = pd.Categorical(df_time['Algorithm'], categories=algorithms, ordered=True)
        df_time = df_time.sort_values('Algorithm')
        
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        
        colors = [style_map[alg]['color'] for alg in df_time['Algorithm']]
        labels = [style_map[alg]['label'] for alg in df_time['Algorithm']]
        
        plot_times = np.maximum(df_time[time_col], 0.01)
        
        bars = ax3.bar(labels, plot_times, color=colors, edgecolor='black', alpha=0.9, width=0.6)
        
        for bar, real_val in zip(bars, df_time[time_col]):
            yval = bar.get_height()
            display_text = f"{real_val:.2f} ms" if real_val >= 0.01 else "<0.01 ms"
            ax3.text(bar.get_x() + bar.get_width()/2, yval * 1.15, display_text, 
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        ax3.set_ylabel('Avg Decision Time per Assignment (ms)')
        ax3.set_title(f'Computational Efficiency (Task Load = {max_load})')
        
        ax3.set_yscale('log')
        ax3.set_ylim(bottom=0.005, top=max(plot_times) * 10)
        
        plt.xticks(rotation=15)
        ax3.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        time_png_path = os.path.join(save_dir, "fig_decision_time.png")
        fig3.savefig(time_png_path, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig3)
        print(f"File saved to: {time_png_path}")

if __name__ == "__main__":
    plot_experimental_results()