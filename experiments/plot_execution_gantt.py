import os
import json
import torch
import copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from env.scc_env import SCCEnv
from scheduler.upper_planner import ConstraintAwareAStar
from scheduler.middle_agent import ShieldedDQNAgent
from scheduler.baselines import RuleBasedScheduler
from utils.task_generator import TaskGenerator

def run_and_log_execution(agent_type, tasks):
    env = SCCEnv(config_path="data/env.yaml")
    planner = ConstraintAwareAStar(env.topology)
    
    if agent_type == 'Proposed-DRL-HIO':
        agent = ShieldedDQNAgent(env.observation_space.shape[0], env.action_space.n)
        try:
            agent.load_state_dict(torch.load("data/models/dqn_best.pth", weights_only=True))
            agent.eval()
        except: pass
        use_shielding = True
    else:
        agent = RuleBasedScheduler(env.topology)
        use_shielding = False

    state, _ = env.reset(tasks=[])
    global_tasks_queue = copy.deepcopy(tasks)
    
    execution_log = []
    
    def plan_next():
        if not global_tasks_queue: return False
        task = global_tasks_queue.pop(0)
        pono = task['pono']
        waypoints = [task['start_ld']]
        if task.get('lf_station'): waypoints.append(task['lf_station'])
        if task.get('rh_station'): waypoints.append(task['rh_station'])
        waypoints.append(task['end_cc'])
        
        real_track_loads = {t: 0.1 for t in env.topology.tracks.keys()}
        if agent_type == 'Proposed-DRL-HIO':
            for v in env.vehicles_state.values():
                if v['status'] != 'IDLE': real_track_loads[v['track']] += 2.0
                
        subtasks = []
        for i in range(len(waypoints) - 1):
            urg = 0.3 if agent_type == 'Proposed-DRL-HIO' else 1.0
            path = planner.find_path(waypoints[i], waypoints[i+1], urg, set(), real_track_loads)
            subtasks.extend(planner.decompose_task(pono, path))
        env.pending_subtasks.extend(subtasks)
        return True

    for _ in range(3): plan_next()
    
    done = False
    vehicle_active_start = {}

    while not done:
        if len(env.pending_subtasks) < 2 and global_tasks_queue:
            plan_next()

        for vid, v_data in env.vehicles_state.items():
            if v_data['status'] in ['MOVING', 'LOADING', 'UNLOADING'] and vid not in vehicle_active_start:
                pono = env.active_subtasks[vid]['global_id'] if vid in env.active_subtasks else -1
                vehicle_active_start[vid] = {'start': env.current_time, 'pono': pono, 'status': v_data['status']}

        if agent_type == 'Proposed-DRL-HIO':
            subtask_track = env.pending_subtasks[0]['required_track'] if env.pending_subtasks else None
            st_pos = env.topology.graph.nodes[env.pending_subtasks[0]['start_node']]['pos'] if env.pending_subtasks else None
            ed_pos = env.topology.graph.nodes[env.pending_subtasks[0]['end_node']]['pos'] if env.pending_subtasks else None
            action = agent.select_action(torch.FloatTensor(state), subtask_track, env.vehicles_state, 0.0, st_pos, ed_pos, use_shielding)
        else:
            action = agent.select_action(env.pending_subtasks, env.vehicles_state)

        state, _, env_done, _, _ = env.step(action)
        
        for vid in list(vehicle_active_start.keys()):
            if env.vehicles_state[vid]['status'] == 'IDLE':
                record = vehicle_active_start.pop(vid)
                execution_log.append({
                    'Vehicle': vid,
                    'Start': record['start'],
                    'End': env.current_time,
                    'Task_ID': record['pono']
                })

        if env_done:
            if env.current_time >= 3000.0 or not global_tasks_queue: done = True
            else: plan_next()
            
    return pd.DataFrame(execution_log), env.current_time

def plot_comparative_gantt():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    generator = TaskGenerator(seed=2029)
    tasks_objs = generator.generate_tasks(task_num=30)
    
    temp_file = "data/temp_gantt_tasks.json"
    generator.save_tasks_to_json(tasks_objs, temp_file)
    with open(temp_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)
        
    print("Running Baseline-Rule ...")
    df_rule, _ = run_and_log_execution('Baseline-Rule', tasks)
    print("Running Proposed-DRL-HIO ...")
    df_chsf, _ = run_and_log_execution('Proposed-DRL-HIO', tasks)
    
    target_vehicles = ['crane1_1', 'crane1_2'] 
    df_rule_filtered = df_rule[df_rule['Vehicle'].isin(target_vehicles)]
    df_chsf_filtered = df_chsf[df_chsf['Vehicle'].isin(target_vehicles)]
    
    span_rule_local = df_rule_filtered['End'].max() if not df_rule_filtered.empty else 0
    span_chsf_local = df_chsf_filtered['End'].max() if not df_chsf_filtered.empty else 0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    
    color_odd = '#2b5b84'
    color_even = '#d65f5f'
    color_idle = '#f0f0f0'
    
    def draw_focused_gantt(ax, df, title, local_span):
        y_ticks = []
        y_labels = []
        
        for i, vid in enumerate(target_vehicles):
            v_data = df[df['Vehicle'] == vid].sort_values(by='Start')
            total_work_time = sum(row['End'] - row['Start'] for _, row in v_data.iterrows())
            
            ax.barh(i, local_span, left=0, height=0.5, color=color_idle, hatch='////', edgecolor='#cccccc', alpha=0.6)
            
            for _, row in v_data.iterrows():
                color = color_even if row['Task_ID'] % 2 == 0 else color_odd
                ax.barh(i, row['End'] - row['Start'], left=row['Start'], height=0.6, color=color, edgecolor='black', linewidth=1.2, zorder=3)
                if row['End'] - row['Start'] > 8:
                    ax.text(row['Start'] + (row['End']-row['Start'])/2, i, f"T{row['Task_ID']}", 
                            ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=4)
            
            y_ticks.append(i)
            y_labels.append(f"{vid}\n(Work: {total_work_time:.1f}s)")
            
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=12)
        ax.set_title(f"{title}", fontsize=14, fontweight='bold', pad=15)
        
        ax.grid(True, axis='x', linestyle='--', alpha=0.5, zorder=0)
        
        ax.axvline(x=local_span, color='red', linestyle='-.', linewidth=2.5, zorder=5)
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5, alpha=0.9)
        ax.text(local_span - 2, -0.4, f"Make-span:\n{local_span:.1f}s", color='red', 
                fontsize=11, fontweight='bold', ha='right', va='top', bbox=bbox_props, zorder=6)
        
        ax.set_xlim(-5, max(span_rule_local, span_chsf_local) + 15)

    draw_focused_gantt(ax1, df_rule_filtered, "(a) Baseline-Rule: Focused Execution Gantt Chart", span_rule_local)
    draw_focused_gantt(ax2, df_chsf_filtered, "(b) Proposed-DRL-HIO: Focused Execution Gantt Chart", span_chsf_local)
    
    ax2.set_xlabel('Simulation Time (s)', fontsize=13)
    
    legend_elements = [
        mpatches.Patch(facecolor=color_odd, edgecolor='black', label='Task Execution (Odd)'),
        mpatches.Patch(facecolor=color_even, edgecolor='black', label='Task Execution (Even)'),
        mpatches.Patch(facecolor=color_idle, hatch='////', edgecolor='#cccccc', label='Idle / Blocked Wait')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12, frameon=True, shadow=True)
    
    plt.tight_layout()
    os.makedirs("data/figures", exist_ok=True)
    plt.savefig("data/figures/fig_focused_gantt_comparison_v2.png", dpi=300, bbox_inches='tight')
    print("File saved to: data/figures/fig_focused_gantt_comparison_v2.png")

if __name__ == "__main__":
    plot_comparative_gantt()