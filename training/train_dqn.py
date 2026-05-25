import os
import torch
import torch.nn as nn
import numpy as np
import json
from env.scc_env import SCCEnv
from scheduler.middle_agent import ShieldedDQNAgent
from scheduler.upper_planner import ConstraintAwareAStar
from training.replay_buffer import CPERBuffer

def parse_global_tasks_to_subtasks(global_tasks, planner):
    all_subtasks = []
    for task in global_tasks:
        pono = task['pono']
        
        waypoints = [task['start_ld']]
        if task.get('lf_station'): waypoints.append(task['lf_station'])
        if task.get('rh_station'): waypoints.append(task['rh_station'])
        waypoints.append(task['end_cc'])
        
        for i in range(len(waypoints) - 1):
            start_node = waypoints[i]
            target_node = waypoints[i+1]
            
            path_with_tracks = planner.find_path(start_node, target_node, urgency=0.5, occupied_nodes=set())
            
            subtasks = planner.decompose_task(pono, path_with_tracks)
            all_subtasks.extend(subtasks)
            
    return all_subtasks

def train_dqn():
    env = SCCEnv(config_path="data/env.yaml")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    planner = ConstraintAwareAStar(env.topology)
    agent = ShieldedDQNAgent(state_dim, action_dim)
    target_agent = ShieldedDQNAgent(state_dim, action_dim)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval()
    
    buffer = CPERBuffer(capacity=10000, rho=0.3)
    optimizer = agent.optimizer
    loss_fn = nn.MSELoss()
    
    with open("data/tasks.json", "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
        
    num_episodes = 200
    batch_size = 64
    gamma = 0.95
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 200
    
    best_reward = -float('inf')
    best_episode = -1
    
    print(f"Start training Shielded DQN... State dimension: {state_dim}, Action space: {action_dim}")
    
    for episode in range(num_episodes):
        np.random.shuffle(all_tasks)
        train_task_count = np.random.randint(15, 26) 
        train_tasks = all_tasks[:train_task_count]
        
        subtasks = parse_global_tasks_to_subtasks(train_tasks, planner)
        
        state, info = env.reset(tasks=subtasks)
        episode_reward = 0
        done = False
        step_count = 0
        
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
        
        while not done:
            subtask_track = env.pending_subtasks[0]['required_track'] if env.pending_subtasks else None
            
            state_tensor = torch.FloatTensor(state)
            action = agent.select_action(state_tensor, subtask_track, env.vehicles_state, epsilon)
            
            if action == -1: action = 0 
                
            next_state, reward, done, _, next_info = env.step(action)
            episode_reward += reward
            
            is_critical = next_info.get('conflict_risk', False)
            buffer.push(state, action, reward, next_state, done, is_critical)
            
            state = next_state
            step_count += 1
            
            if len(buffer.normal_buffer) + len(buffer.crit_buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                curr_q = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    max_next_q = target_agent(next_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)
                    
                loss = loss_fn(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        if episode % 10 == 0:
            target_agent.load_state_dict(agent.state_dict())
            
        print(f"Episode {episode:03d} | Steps: {step_count} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode = episode
            os.makedirs("data/models", exist_ok=True)
            torch.save(agent.state_dict(), "data/models/dqn_best.pth")
            print(f"   New best model found and saved! (Reward: {best_reward:.2f})")
    
    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Generated at Episode {best_episode}")
    print(f"Model saved to: data/models/dqn_best.pth")
    print("="*60 + "\n")

if __name__ == "__main__":
    train_dqn()