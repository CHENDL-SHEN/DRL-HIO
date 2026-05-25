import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class ShieldedDQNAgent(nn.Module):
    def __init__(self, state_dim: int, num_vehicles: int):
        super(ShieldedDQNAgent, self).__init__()
        self.num_vehicles = num_vehicles
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_vehicles)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        return self.fc(state)

    def select_action(self, state_tensor: torch.Tensor, subtask_track: str, vehicles_info: dict, epsilon: float, task_start_pos: list = None, task_end_pos: list = None, use_shielding: bool = True) -> int:
        
        base_mask = np.zeros(self.num_vehicles)
        vehicle_ids = list(vehicles_info.keys())
        for i, v_id in enumerate(vehicle_ids):
            v_data = vehicles_info[v_id]
            if v_data['track'] != subtask_track or v_data['status'] != 'IDLE':
                base_mask[i] = -np.inf 

        shield_mask = np.copy(base_mask)
        if use_shielding and task_start_pos is not None and task_end_pos is not None:
            for i, v_id in enumerate(vehicle_ids):
                if shield_mask[i] == -np.inf: continue
                v_data = vehicles_info[v_id]
                v_pos = v_data['pos']
                is_blocked = False
                
                min_x = min(v_pos[0], task_start_pos[0], task_end_pos[0]) - 0.5
                max_x = max(v_pos[0], task_start_pos[0], task_end_pos[0]) + 0.5
                min_y = min(v_pos[1], task_start_pos[1], task_end_pos[1]) - 0.5
                max_y = max(v_pos[1], task_start_pos[1], task_end_pos[1]) + 0.5
                
                for other_id, other_data in vehicles_info.items():
                    if other_id != v_id and other_data['track'] == subtask_track:
                        other_pos = other_data['pos']
                        if (min_x <= other_pos[0] <= max_x) and (min_y <= other_pos[1] <= max_y):
                            dist_to_other = abs(v_pos[0] - other_pos[0]) + abs(v_pos[1] - other_pos[1])
                            if dist_to_other < 0.1:
                                if v_id > other_id:
                                    is_blocked = True
                                    break
                            else:
                                is_blocked = True
                                break
                if is_blocked:
                    shield_mask[i] = -np.inf

        valid_indices = np.where(shield_mask > -np.inf)[0]
        used_mask = shield_mask
        
        if len(valid_indices) == 0:
            valid_indices = np.where(base_mask > -np.inf)[0]
            used_mask = base_mask
            
        if len(valid_indices) == 0:
            return -1 

        if random.random() < epsilon:
            return int(random.choice(valid_indices))
            
        with torch.no_grad():
            q_values = self.forward(state_tensor).numpy()
            
            if not use_shielding:
                masked_q = q_values + used_mask
                return int(np.argmax(masked_q))

            q_valid = q_values[valid_indices]
            if np.max(q_valid) > np.min(q_valid):
                q_norm = (q_valid - np.min(q_valid)) / (np.max(q_valid) - np.min(q_valid))
            else:
                q_norm = np.ones_like(q_valid)

            dists = []
            for idx in valid_indices:
                v_pos = vehicles_info[list(vehicles_info.keys())[idx]]['pos']
                d = abs(v_pos[0] - task_start_pos[0]) + abs(v_pos[1] - task_start_pos[1])
                dists.append(d)
            dists = np.array(dists)
            
            if np.max(dists) > np.min(dists):
                dist_norm = (np.max(dists) - dists) / (np.max(dists) - np.min(dists))
            else:
                dist_norm = np.ones_like(dists)
                
            final_scores = 0.5 * q_norm + 0.5 * dist_norm
            best_idx = valid_indices[np.argmax(final_scores)]
            
            return int(best_idx)