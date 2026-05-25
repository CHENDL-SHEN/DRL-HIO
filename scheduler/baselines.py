import numpy as np
import math
import random

class RuleBasedScheduler:
    def __init__(self, topology):
        self.topology = topology
        self.num_vehicles = len(topology.vehicles)
        self.vehicle_ids = list(topology.vehicles.keys())

    def select_action(self, pending_subtasks: list, vehicles_state: dict) -> int:
        if not pending_subtasks:
            return -1
            
        task = pending_subtasks[0]
        required_track = task.get('required_track')
        task_start_node = task.get('start_node')
        task_start_pos = self.topology.graph.nodes[task_start_node]['pos'] if task_start_node else [0,0]
        
        best_action = -1
        min_distance = float('inf')
        
        for i, vid in enumerate(self.vehicle_ids):
            v_state = vehicles_state[vid]
            
            if v_state['status'] == 'IDLE' and v_state['track'] == required_track:
                v_pos = v_state['pos']
                dist = abs(v_pos[0] - task_start_pos[0]) + abs(v_pos[1] - task_start_pos[1])
                
                if dist < min_distance:
                    min_distance = dist
                    best_action = i
                    
        return best_action


class GAScheduler:
    def __init__(self, topology, pop_size=20, generations=30):
        self.topology = topology
        self.vehicle_ids = list(topology.vehicles.keys())
        self.num_vehicles = len(self.vehicle_ids)
        self.pop_size = pop_size
        self.generations = generations

    def fitness(self, action, task_start_pos, required_track, vehicles_state):
        vid = self.vehicle_ids[action]
        v_state = vehicles_state[vid]
        
        if v_state['status'] != 'IDLE' or v_state['track'] != required_track:
            return float('inf')
            
        v_pos = v_state['pos']
        dist = abs(v_pos[0] - task_start_pos[0]) + abs(v_pos[1] - task_start_pos[1])
        
        load_penalty = 0.0
        load_penalty += v_state.get('epsilon', 0) * 10 
        
        return dist + load_penalty

    def select_action(self, pending_subtasks: list, vehicles_state: dict) -> int:
        if not pending_subtasks:
            return -1
            
        task = pending_subtasks[0]
        required_track = task.get('required_track')
        task_start_node = task.get('start_node')
        task_start_pos = self.topology.graph.nodes[task_start_node]['pos'] if task_start_node else [0,0]
        
        valid_actions = [i for i, vid in enumerate(self.vehicle_ids) 
                        if vehicles_state[vid]['status'] == 'IDLE' and vehicles_state[vid]['track'] == required_track]
        
        if not valid_actions:
            return -1
            
        if len(valid_actions) == 1:
            return valid_actions[0]

        population = random.choices(valid_actions, k=self.pop_size)
        
        for _ in range(self.generations):
            scored_pop = [(action, self.fitness(action, task_start_pos, required_track, vehicles_state)) for action in population]
            scored_pop.sort(key=lambda x: x[1])
            
            elite_idx = max(1, int(self.pop_size * 0.2))
            new_population = [x[0] for x in scored_pop[:elite_idx]]
            
            while len(new_population) < self.pop_size:
                if random.random() < 0.2: 
                    new_population.append(random.choice(valid_actions))
                else:
                    new_population.append(random.choice(new_population[:elite_idx])) 
                    
            population = new_population

        best_action = min(population, key=lambda a: self.fitness(a, task_start_pos, required_track, vehicles_state))
        return best_action