import matplotlib.pyplot as plt
import numpy as np
import os

def run_cma_idle_yielding_case_study():
    dt = 0.1
    total_time = 20.0
    
    time_history = []
    c1_pos_history = []
    c2_pos_history = []
    
    c1_pos = 2.0
    c1_target = 16.0
    c1_vel_nom = 2.0
    
    c2_pos = 9.0
    c2_vel_nom = 0.0 
    
    safe_dist = 2.5 
    
    for step in range(int(total_time / dt)):
        t = step * dt
        time_history.append(t)
        
        if c1_pos < c1_target:
            c1_vel = c1_vel_nom
        else:
            c1_vel = 0.0 
            
        next_c1_nom = c1_pos + c1_vel * dt
        next_c2_nom = c2_pos + c2_vel_nom * dt
        
        if next_c2_nom - next_c1_nom < safe_dist:
            c2_vel = (next_c1_nom + safe_dist - c2_pos) / dt
        else:
            c2_vel = 0.0 
            
        c1_pos += c1_vel * dt
        c2_pos += c2_vel * dt
            
        c1_pos_history.append(c1_pos)
        c2_pos_history.append(c2_pos)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time_history, c1_pos_history, label='Crane 1_1 (Active Task, Moving Up)', color='#d62728', linewidth=3.5, zorder=4)
    ax.plot(time_history, c2_pos_history, label='Crane 1_2 (Idle, Blocking Path)', color='#1f77b4', linewidth=3.5, linestyle='-', zorder=4)
    
    ax.fill_between(time_history, c1_pos_history, c2_pos_history, color='gray', alpha=0.25, label=f'Safe Distance Maintained ($d_{{safe}}={safe_dist}m$)', zorder=2)
    
    push_start, push_end = None, None
    for i in range(1, len(c2_pos_history)):
        if c2_pos_history[i] > c2_pos_history[i-1] + 0.001 and push_start is None:
            push_start = time_history[i]
        if c2_pos_history[i] <= c2_pos_history[i-1] + 0.001 and push_start is not None and push_end is None:
            push_end = time_history[i]
            
    ax.axvspan(0, push_start, color='#e6f2ff', alpha=0.4, label='Phase 1: C1 Approaching', zorder=1)
    ax.axvspan(push_start, push_end, color='#ff7f0e', alpha=0.25, label='Phase 2: C2 Forced to Yield', zorder=1)
    ax.axvspan(push_end, 20.0, color='#2ca02c', alpha=0.25, label='Phase 3: C1 Working & C2 Waits', zorder=1)
    
    ax.annotate('Idle C2 Awakened &\nPushed Upward', xy=((push_start+push_end)/2, 14.5), xytext=(1.5, 16.0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                ha='center', va='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle="round", fc="white", ec="black"))
    
    ax.annotate('C1 Arrives & Works\nC2 Stays Outside Safe Dist', xy=(12.0, 16.0), xytext=(13.0, 10.0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                ha='center', va='center', fontsize=11, fontweight='bold', color='darkred', bbox=dict(boxstyle="round", fc="#ffeeee", ec="darkred"))

    ax.set_xlabel('Simulation Time (s)', fontsize=13)
    ax.set_ylabel('Spatial Position (Coordinate)', fontsize=13)
    ax.set_title('Micro-Conflict Resolution: Active Yielding of Idle Equipment ($C_3$ Modality)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_xlim(0, 20)
    
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 2, 3, 4, 5]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
              loc='lower right', fontsize=10, framealpha=0.95, edgecolor='black', ncol=2)
    
    plt.tight_layout()
    os.makedirs("data/figures", exist_ok=True)
    save_path = "data/figures/fig_cma_idle_yielding.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"File saved to: {save_path}")

if __name__ == "__main__":
    run_cma_idle_yielding_case_study()