import matplotlib.pyplot as plt
import numpy as np
import os

def run_cma_rear_end_case_study():
    dt = 0.1
    total_time = 15.0
    
    time_history = []
    c1_pos_history = []
    c2_pos_history = []
    c1_vel_history = []
    
    c2_pos = 10.0
    c2_vel_nom = 1.2
    
    c1_pos = 0.0
    c1_vel_nom = 3.5
    
    safe_dist = 2.5 
    kappa = 1.5     

    current_time = 0.0
    
    for _ in range(int(total_time / dt)):
        time_history.append(current_time)
        c1_pos_history.append(c1_pos)
        c2_pos_history.append(c2_pos)

        actual_dist = c2_pos - c1_pos
        c2_vel = c2_vel_nom 
        
        if actual_dist < safe_dist + 2.0:
            v_cmd = c2_vel + kappa * (actual_dist - safe_dist)
            c1_vel = max(0.0, min(v_cmd, c1_vel_nom)) 
        else:
            c1_vel = c1_vel_nom
            
        c1_vel_history.append(c1_vel)
        
        c1_pos += c1_vel * dt
        c2_pos += c2_vel * dt
        current_time += dt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time_history, c1_pos_history, label='Rear Crane 1_1 (Fast, Approaching)', color='#d62728', linewidth=3.5, zorder=4)
    ax.plot(time_history, c2_pos_history, label='Front Crane 1_2 (Slow, Moving Ahead)', color='#1f77b4', linewidth=3.5, linestyle='-', zorder=4)
    
    intervention_start = None
    for i, v in enumerate(c1_vel_history):
        if v < c1_vel_nom - 0.01:
            intervention_start = time_history[i]
            break
            
    ax.axvspan(intervention_start, total_time, color='#ffe119', alpha=0.25, label='Phase 2: CMA Adaptive Tracking', zorder=1)
    ax.axvspan(0, intervention_start, color='#e6f2ff', alpha=0.4, label='Phase 1: Free Approaching', zorder=1)
    
    ax.fill_between(time_history, c1_pos_history, c2_pos_history, where=np.array(time_history) >= intervention_start, 
                     color='gray', alpha=0.3, label=f'Safe Distance Maintained ($d_{{safe}}={safe_dist}m$)', zorder=2)
    
    interv_pos = c1_pos_history[int(intervention_start/dt)]
    ax.annotate('CMA Intervenes:\nSmooth Deceleration',
                 xy=(intervention_start, interv_pos), xytext=(1.5, 15.0),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                 ha='center', va='center', fontsize=11, fontweight='bold', zorder=5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    mid_time = (intervention_start + total_time) / 2
    stop_pos = c1_pos_history[int(mid_time/dt)]
    ax.annotate('Velocity Matched ($v_{rear} = v_{front}$)\nDistance Locked at $d_{safe}$',
                 xy=(mid_time, stop_pos), xytext=(11.0, 5.0),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                 ha='center', va='center', fontsize=11, fontweight='bold', color='darkred', zorder=5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ffeeee", ec="darkred", lw=1))

    ax.set_xlabel('Simulation Time (s)', fontsize=13)
    ax.set_ylabel('Spatial Position (Coordinate)', fontsize=13)
    ax.set_title('Micro-Conflict Resolution: Adaptive Velocity Tracking ($C_1$ Modality)', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, max(c2_pos_history) + 2)
    
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 4, 2, 3] 
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
              loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    os.makedirs("data/figures", exist_ok=True)
    save_path = "data/figures/fig_cma_rear_end.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"File saved to: {save_path}")

if __name__ == "__main__":
    run_cma_rear_end_case_study()