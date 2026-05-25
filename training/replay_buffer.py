from collections import deque
import numpy as np
import random
import torch

class CPERBuffer:
    def __init__(self, capacity=10000, rho=0.3):
        self.normal_buffer = deque(maxlen=capacity)
        self.crit_buffer = deque(maxlen=capacity // 5) 
        self.rho = rho 

    def push(self, state, action, reward, next_state, done, is_critical: bool):
        experience = (state, action, reward, next_state, done)
        if is_critical:
            self.crit_buffer.append(experience)
        else:
            self.normal_buffer.append(experience)

    def sample(self, batch_size):
        crit_batch_size = int(batch_size * self.rho)
        norm_batch_size = batch_size - crit_batch_size
        
        if len(self.normal_buffer) < norm_batch_size:
            norm_batch_size = len(self.normal_buffer)
            crit_batch_size = batch_size - norm_batch_size

        if len(self.crit_buffer) < crit_batch_size:
            crit_batch_size = len(self.crit_buffer)
            norm_batch_size = batch_size - crit_batch_size
            
        batch = []
        if crit_batch_size > 0:
            batch.extend(random.sample(self.crit_buffer, crit_batch_size))
        if norm_batch_size > 0:
            batch.extend(random.sample(self.normal_buffer, norm_batch_size))
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (torch.FloatTensor(np.array(states, dtype=np.float32)), 
                torch.LongTensor(actions), 
                torch.FloatTensor(rewards), 
                torch.FloatTensor(np.array(next_states, dtype=np.float32)), 
                torch.FloatTensor(dones))