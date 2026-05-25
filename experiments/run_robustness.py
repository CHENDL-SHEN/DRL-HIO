import json
import torch
import pandas as pd
import copy
import numpy as np
import os
import time
from utils.task_generator import TaskGenerator
from env.scc_env import SCCEnv
from scheduler.upper_planner import ConstraintAwareAStar
from scheduler.middle_agent import ShieldedDQNAgent
from scheduler.baselines import RuleBasedScheduler, GAScheduler
from experiments.evaluate import PerformanceEvaluator

def plan_next_global_task_robust(global_tasks_queue, env, planner, config):
    if not global_tasks_queue:
        return False
        
    task = global_tasks_queue.pop(0)
    pono = task['pono']
    waypoints = [task['start_ld']]
    if task.get('lf_station'): waypoints.append(task['lf_station'])
    if task.get('rh_station'): waypoints.append(task['rh_station'])
    waypoints.append(task['end_cc'])
    
    real_track_loads = {t: 0.0 for t in env.topology.tracks.keys()}
    occupied_nodes = set()
    
    for vid, v_state in env.vehicles_state.items():
        if v_state['status'] == 'ERROR':
            real_track_loads[v_state['track']] += 100.0 
            if v_state['current_node']:
                occupied_nodes.add(v_state['current_node'])
        elif v_state['status'] != 'IDLE':
            real_track_loads[v_state['track']] += 2.0
        else:
            real_track_loads[v_state['track']] += 0.1
            
    for st in env.pending_subtasks:
        real_track_loads[st.get('required_track')] += 0.2

    subtasks = []
    for i in range(len(waypoints) - 1):
        urgency = 0.3 if config['use_astar'] else 1.0
        path = planner.find_path(waypoints[i], waypoints[i+1], urgency=urgency, occupied_nodes=occupied_nodes, track_loads=real_track_loads)
        subtasks.extend(planner.decompose_task(pono, path))
        
    env.pending_subtasks.extend(subtasks)
    return True

def run_evaluation(config_name: str, config: dict, tasks: list, enable_disturbance: bool) -> dict:
    env = SCCEnv(config_path="data/env.yaml")
    evaluator = PerformanceEvaluator()
    planner = ConstraintAwareAStar(env.topology)
    
    state, info = env.reset(tasks=[])
    global_tasks_queue = copy.deepcopy(tasks)
    
    for _ in range(3):
        plan_next_global_task_robust(global_tasks_queue, env, planner, config)
        
    if config['agent_type'] == 'rl':
        agent = ShieldedDQNAgent(env.observation_space.shape[0], env.action_space.n)
        try:
            agent.load_state_dict(torch.load("data/models/dqn_best.pth", weights_only=True))
            agent.eval()
        except: pass 
    elif config['agent_type'] == 'ga':
        agent = GAScheduler(env.topology)
    else:
        agent = RuleBasedScheduler(env.topology)
        
    done = False
    conflict_cooldowns = {} 
    disturbance_triggered = False
    
    while not done:
        if enable_disturbance and not disturbance_triggered and env.current_time >= 200.0:
            env.trigger_breakdown("crane3_1", duration=300.0)
            disturbance_triggered = True

        if len(env.pending_subtasks) < 2 and len(global_tasks_queue) > 0:
            plan_next_global_task_robust(global_tasks_queue, env, planner, config)

        if config['agent_type'] == 'rl':
            subtask_track = env.pending_subtasks[0]['required_track'] if env.pending_subtasks else None
            task_start_pos = None
            task_end_pos = None  
            if env.pending_subtasks:
                task_start_pos = env.topology.graph.nodes[env.pending_subtasks[0]['start_node']]['pos']
                task_end_pos = env.topology.graph.nodes[env.pending_subtasks[0]['end_node']]['pos'] 
                
            state_tensor = torch.FloatTensor(state)
            action = agent.select_action(state_tensor, subtask_track, env.vehicles_state, 0.0, task_start_pos, task_end_pos, config['use_shielding'])
        else:
            action = agent.select_action(env.pending_subtasks, env.vehicles_state)
            
        state, reward, env_done, _, info = env.step(action)
        
        current_conflicts = info.get('active_conflicts', set())
        for conflict_pair in current_conflicts:
            last_time = conflict_cooldowns.get(conflict_pair, -float('inf'))
            if env.current_time - last_time >= 60.0:
                evaluator.record_deadlock()
                conflict_cooldowns[conflict_pair] = env.current_time
            
        if env_done:
            if env.current_time >= 10000.0 or len(global_tasks_queue) == 0:
                done = True  
            else:
                plan_next_global_task_robust(global_tasks_queue, env, planner, config)
                done = False
            
    metrics = evaluator.get_metrics()
    if not metrics:
        metrics = {'Make_span': min(env.current_time, 10000.0), 'Deadlocks': evaluator.deadlocks}
    
    cascading_penalty = 15.0 + (metrics['Deadlocks'] * 5.0) 
    metrics['Make_span'] += (metrics['Deadlocks'] * cascading_penalty)
    
    return metrics

def main():
    configs = {
        "Baseline-Rule": {'agent_type': 'rule', 'use_astar': False, 'use_shielding': False},
        "Baseline-GA":   {'agent_type': 'ga',   'use_astar': False, 'use_shielding': False},
        "Proposed-DRL-HIO": {'agent_type': 'rl',   'use_astar': True,  'use_shielding': True}
    }
    
    load = 30 
    num_runs = 10
    results = []
    
    print("Starting robustness comparative experiments...")
    
    for run_id in range(num_runs):
        seed = 2026 + run_id
        generator = TaskGenerator(seed=seed)
        test_tasks = generator.generate_tasks(task_num=load)
        temp_file = "data/temp_robust_tasks.json"
        generator.save_tasks_to_json(test_tasks, temp_file)
        
        with open(temp_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
            
        for name, cfg in configs.items():
            metrics_normal = run_evaluation(name, cfg, tasks, enable_disturbance=False)
            metrics_disturb = run_evaluation(name, cfg, tasks, enable_disturbance=True)
            
            results.append({
                'Algorithm': name, 'Run_ID': run_id, 'Condition': 'Normal',
                'Make_span': metrics_normal['Make_span'], 'Deadlocks': metrics_normal['Deadlocks']
            })
            results.append({
                'Algorithm': name, 'Run_ID': run_id, 'Condition': 'Breakdown (300s)',
                'Make_span': metrics_disturb['Make_span'], 'Deadlocks': metrics_disturb['Deadlocks']
            })
            
    df = pd.DataFrame(results)
    df.to_csv("data/robustness_results.csv", index=False)
    
    summary = df.groupby(['Algorithm', 'Condition']).mean(numeric_only=True).round(1).reset_index()
    print("\n📊 Robustness Performance Summary Table (Mean):")
    print(summary[['Algorithm', 'Condition', 'Make_span', 'Deadlocks']].to_string(index=False))

    from experiments.plot_robustness import plot_robustness_chart
    plot_robustness_chart()

if __name__ == "__main__":
    main()