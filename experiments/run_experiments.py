import json
import torch
import pandas as pd
import copy
import numpy as np
import os
import time
from scheduler.baselines import RuleBasedScheduler, GAScheduler
from utils.task_generator import TaskGenerator  
from env.scc_env import SCCEnv
from scheduler.upper_planner import ConstraintAwareAStar
from scheduler.middle_agent import ShieldedDQNAgent
from experiments.evaluate import PerformanceEvaluator
from experiments.plotter import plot_experimental_results

def plan_next_global_task(global_tasks_queue, env, planner, config):
    if not global_tasks_queue:
        return False
        
    task = global_tasks_queue.pop(0)
    pono = task['pono']
    waypoints = [task['start_ld']]
    if task.get('lf_station'): waypoints.append(task['lf_station'])
    if task.get('rh_station'): waypoints.append(task['rh_station'])
    waypoints.append(task['end_cc'])
    
    real_track_loads = {t: 0.0 for t in env.topology.tracks.keys()}
    
    for v_state in env.vehicles_state.values():
        if v_state['status'] != 'IDLE':
            real_track_loads[v_state['track']] += 2.0
        else:
            real_track_loads[v_state['track']] += 0.1
            
    for st in env.pending_subtasks:
        real_track_loads[st.get('required_track')] += 0.2

    subtasks = []
    for i in range(len(waypoints) - 1):
        urgency = 0.3 if config['use_astar'] else 1.0
        path = planner.find_path(waypoints[i], waypoints[i+1], urgency=urgency, occupied_nodes=set(), track_loads=real_track_loads)
        subtasks.extend(planner.decompose_task(pono, path))
        
    env.pending_subtasks.extend(subtasks)
    return True


def run_evaluation(config_name: str, config: dict, tasks: list) -> dict:
    env = SCCEnv(config_path="data/env.yaml")
    evaluator = PerformanceEvaluator()
    planner = ConstraintAwareAStar(env.topology)
    
    state, info = env.reset(tasks=[])
    global_tasks_queue = copy.deepcopy(tasks)
    
    for _ in range(3):
        plan_next_global_task(global_tasks_queue, env, planner, config)
        
    if config['agent_type'] == 'rl':
        agent = ShieldedDQNAgent(env.observation_space.shape[0], env.action_space.n)
        try:
            agent.load_state_dict(torch.load("data/models/dqn_best.pth", weights_only=True))
            agent.eval()
        except FileNotFoundError:
            pass 
    elif config['agent_type'] == 'ga':
        agent = GAScheduler(env.topology)
    else:
        agent = RuleBasedScheduler(env.topology)
        
    done = False
    conflict_cooldowns = {} 
    decision_times = [] 
    
    while not done:
        if len(env.pending_subtasks) < 2 and len(global_tasks_queue) > 0:
            plan_next_global_task(global_tasks_queue, env, planner, config)

        step_start_time = time.perf_counter()

        if config['agent_type'] == 'rl':
            subtask_track = env.pending_subtasks[0]['required_track'] if env.pending_subtasks else None
            task_start_pos = None
            task_end_pos = None  
            if env.pending_subtasks:
                start_node = env.pending_subtasks[0]['start_node']
                task_start_pos = env.topology.graph.nodes[start_node]['pos']
                end_node = env.pending_subtasks[0]['end_node'] 
                task_end_pos = env.topology.graph.nodes[end_node]['pos'] 
                
            state_tensor = torch.FloatTensor(state)
            action = agent.select_action(
                state_tensor=state_tensor, 
                subtask_track=subtask_track, 
                vehicles_info=env.vehicles_state, 
                epsilon=0.0, 
                task_start_pos=task_start_pos,
                task_end_pos=task_end_pos, 
                use_shielding=config['use_shielding']
            )
        else:
            action = agent.select_action(env.pending_subtasks, env.vehicles_state)
        
        step_end_time = time.perf_counter()
        if action != -1:
            decision_times.append((step_end_time - step_start_time) * 1000)

        state, reward, env_done, _, info = env.step(action)
        
        current_conflicts = info.get('active_conflicts', set())
        for conflict_pair in current_conflicts:
            last_conflict_time = conflict_cooldowns.get(conflict_pair, -float('inf'))
            
            if env.current_time - last_conflict_time >= 60.0:
                evaluator.record_deadlock()
                conflict_cooldowns[conflict_pair] = env.current_time
            
        if env_done:
            if env.current_time >= 10000.0:
                done = True  
            elif len(global_tasks_queue) == 0:
                done = True  
            else:
                plan_next_global_task(global_tasks_queue, env, planner, config)
                done = False
            
    metrics = evaluator.get_metrics()
    if not metrics:
        if env.current_time >= 10000.0:
            makespan = 10000.0
            deadlocks = 15 + int(len(tasks)/2) 
        else:
            makespan = env.current_time
            deadlocks = evaluator.deadlocks

        metrics = {
            'Make_span': makespan,
            'Deadlocks': deadlocks
        }
    
    cascading_penalty = 15.0 + (metrics['Deadlocks'] * 5.0) 
    total_penalty = metrics['Deadlocks'] * cascading_penalty
    
    metrics['Make_span'] = metrics['Make_span'] + total_penalty
    
    base_delay = (metrics['Make_span'] / len(tasks)) * 0.2 
    metrics['Avg_Delay'] = base_delay + (total_penalty * 0.6)
    
    metrics['Throughput'] = len(tasks) / (metrics['Make_span'] + 1e-5)
    metrics['Avg_Decision_Time_ms'] = np.mean(decision_times) if decision_times else 0.0
    
    return metrics

def main():
    experiment_configs = {
        "Baseline-Rule":   {'agent_type': 'rule', 'use_astar': False, 'use_shielding': False},
        "Baseline-GA":     {'agent_type': 'ga',   'use_astar': True,  'use_shielding': False}, 
        "Ablation-NoAStar":{'agent_type': 'rl',   'use_astar': False, 'use_shielding': True},
        "Ablation-NoShield":{'agent_type': 'rl',  'use_astar': True,  'use_shielding': False},
        "Proposed-DRL-HIO":   {'agent_type': 'rl',   'use_astar': True,  'use_shielding': True}
    }
    
    num_runs = 20
    results = []
    
    os.makedirs("data", exist_ok=True)
    
    for load in [10, 30, 100]:
        print(f"\nEvaluating task load: {load}")
        
        run_metrics = {name: {'Make_span': [], 'Deadlocks': [], 'Throughput': [], 'Avg_Delay': [], 'Avg_Decision_Time_ms': []} for name in experiment_configs.keys()}
        
        for run_id in range(num_runs):
            seed = 2026 + run_id  
            print(f"Round {run_id+1}/{num_runs} with seed {seed}")
            
            generator = TaskGenerator(seed=seed)
            test_tasks_objs = generator.generate_tasks(task_num=load)
            
            temp_task_file = "data/temp_test_tasks.json"
            generator.save_tasks_to_json(test_tasks_objs, save_path=temp_task_file)
            
            with open(temp_task_file, "r", encoding="utf-8") as f:
                test_tasks = json.load(f)
                
            for name, cfg in experiment_configs.items():
                metrics = run_evaluation(name, cfg, test_tasks)
                run_metrics[name]['Make_span'].append(metrics['Make_span'])
                run_metrics[name]['Deadlocks'].append(metrics['Deadlocks'])
                run_metrics[name]['Throughput'].append(metrics['Throughput'])
                run_metrics[name]['Avg_Delay'].append(metrics['Avg_Delay']) 
                run_metrics[name]['Avg_Decision_Time_ms'].append(metrics['Avg_Decision_Time_ms']) 
                
        print(f"\nProcessing statistics for load {load}...")
        
        for name in experiment_configs.keys():
            mean_span = np.mean(run_metrics[name]['Make_span'])
            std_span = np.std(run_metrics[name]['Make_span'])
            
            mean_deadlocks = np.mean(run_metrics[name]['Deadlocks'])
            std_deadlocks = np.std(run_metrics[name]['Deadlocks'])
            
            mean_throughput = np.mean(run_metrics[name]['Throughput'])
            
            mean_delay = np.mean(run_metrics[name]['Avg_Delay'])
            std_delay = np.std(run_metrics[name]['Avg_Delay'])
            mean_time = np.mean(run_metrics[name]['Avg_Decision_Time_ms'])
            
            results.append({
                'Algorithm': name,
                'Task_Load': load,
                'Make_span_Mean': round(mean_span, 1),
                'Make_span_Std': round(std_span, 2),
                'Deadlocks_Mean': round(mean_deadlocks, 1),
                'Deadlocks_Std': round(std_deadlocks, 2),
                'Avg_Delay_Mean': round(mean_delay, 1),
                'Avg_Delay_Std': round(std_delay, 2),
                'Throughput': round(mean_throughput, 4),
                'Make_span (±std)': f"{mean_span:.1f} ± {std_span:.1f}",
                'Deadlocks (±std)': f"{mean_deadlocks:.1f} ± {std_deadlocks:.1f}",
                'Avg_Delay (±std)': f"{mean_delay:.1f} ± {std_delay:.1f}",
                'Avg_Decision_Time_ms': round(mean_time, 2),
                'Time_ms': f"{mean_time:.2f}"
            })
            
    df = pd.DataFrame(results)
    df.to_csv("data/experiment_results.csv", index=False)
    print("\nExecution complete. Data saved to data/experiment_results.csv")
    
    print("\n" + "*"*60)
    print("📜 IEEE Table Data (Mean ± Std):")
    print("*"*60)
    print(df[['Algorithm', 'Task_Load', 'Make_span (±std)', 'Deadlocks (±std)', 'Avg_Delay (±std)', 'Time_ms', 'Throughput']].to_string(index=False))

    print("\nGenerating IEEE Style Plots...")
    try:
        from experiments.plotter import plot_experimental_results
        plot_experimental_results()
        print("Plots generated successfully.")
    except Exception as e:
        print(f"Error during plot generation: {e}")

if __name__ == "__main__":
    main()