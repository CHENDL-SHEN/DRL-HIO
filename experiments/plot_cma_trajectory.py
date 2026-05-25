import matplotlib.pyplot as plt
import numpy as np
import os

def run_cma_mutual_yielding_case_study():
    dt = 0.1
    total_time = 28.0
    
    time_history = []
    c1_pos_history = []
    c2_pos_history = []
    
    c1_pos = 2.0
    c2_pos = 25.0
    
    safe_dist = 2.5 
    
    for step in range(int(total_time / dt)):
        t = step * dt
        time_history.append(t)
        
        if t < 12.0:
            c1_priority = 100
            c2_priority = 10
            c1_vel_nom = 2.0 if c1_pos < 18.0 else 0.0  
            c2_vel_nom = -2.0                          
        else:
            c1_priority = 10
            c2_priority = 100
            c1_vel_nom = 0.0                           
            c2_vel_nom = -2.0 if c2_pos > 4.0 else 0.0 
            
        next_c1_nom = c1_pos + c1_vel_nom * dt
        next_c2_nom = c2_pos + c2_vel_nom * dt
        
        c1_vel = c1_vel_nom
        c2_vel = c2_vel_nom
        
        if next_c2_nom - next_c1_nom < safe_dist:
            if c1_priority >= c2_priority:
                c1_vel = c1_vel_nom
                c2_vel = (c1_pos + c1_vel * dt + safe_dist - c2_pos) / dt
            else:
                c2_vel = c2_vel_nom
                c1_vel = (c2_pos + c2_vel * dt - safe_dist - c1_pos) / dt

        c1_pos += c1_vel * dt
        c2_pos += c2_vel * dt
            
        c1_pos_history.append(c1_pos)
        c2_pos_history.append(c2_pos)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time_history, c1_pos_history, label='Crane 1_1 (Initial High Priority)', color='#d62728', linewidth=3.5, zorder=4)
    ax.plot(time_history, c2_pos_history, label='Crane 1_2 (Initial Low Priority)', color='#1f77b4', linewidth=3.5, linestyle='-', zorder=4)
    
    ax.fill_between(time_history, c1_pos_history, c2_pos_history, color='gray', alpha=0.25, label=f'Safe Distance Maintained ($d_{{safe}}={safe_dist}m$)', zorder=2)
    
    ax.axvspan(5.2, 8.0, color='#ff7f0e', alpha=0.2, label='Phase 1: C2 Forced to Retreat', zorder=1)
    ax.axvspan(8.0, 12.0, color='#ffe119', alpha=0.2, label='Phase 2: C2 Yield & Wait', zorder=1)
    ax.axvspan(12.0, 20.2, color='#2ca02c', alpha=0.2, label='Phase 3: Priority Swap & C1 Yields', zorder=1)
    
    ax.annotate('C2 Forced Upward', xy=(6.5, 17.5), xytext=(2.0, 21.0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                ha='center', va='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle="round", fc="white", ec="black"))
    
    ax.annotate('Priority Swap!\nC1 Forced Downward', xy=(15.0, 12.0), xytext=(19.0, 15.0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                ha='center', va='center', fontsize=11, fontweight='bold', color='darkred', bbox=dict(boxstyle="round", fc="#ffeeee", ec="darkred"))
    
    ax.axvline(x=12.0, color='black', linestyle='-.', alpha=0.6, linewidth=1.5)
    ax.text(12.2, 2.5, 'C1 task done\nC2 gets right-of-way', color='black', fontsize=10, fontweight='bold', style='italic')

    ax.set_xlabel('Simulation Time (s)', fontsize=13)
    ax.set_ylabel('Spatial Position (Coordinate)', fontsize=13)
    ax.set_title('Micro-Conflict Resolution: Dynamic Priority Swap & Mutual Yielding ($C_2$ Modality)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_xlim(0, 25)
    
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 2, 3, 4, 5]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
              loc='lower left', fontsize=10, framealpha=0.95, edgecolor='black', ncol=2)
    
    plt.tight_layout()
    os.makedirs("data/figures", exist_ok=True)
    save_path = "data/figures/fig_cma_mutual_yielding.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"File saved to: {save_path}")

if __name__ == "__main__":
    run_cma_mutual_yielding_case_study()