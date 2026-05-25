import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import math
from typing import Tuple, Dict, List, Optional

from env.topology import SCCTopology
from env.cma_controller import ConflictModalityAnalysis

class SCCEnv(gym.Env):
    def __init__(self, config_path: str):
        super(SCCEnv, self).__init__()

        self.topology = SCCTopology(config_path)
        self.cma_controller = ConflictModalityAnalysis(safe_dist=2.0, kappa=0.5)

        self.vehicle_ids = list(self.topology.vehicles.keys())
        self.num_vehicles = len(self.vehicle_ids)
        self.node_ids = list(self.topology.graph.nodes())

        self.action_space = spaces.Discrete(self.num_vehicles)

        self.state_dim = self.num_vehicles * 3 + len(self.topology.tracks) + 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        self.dt = 1.0
        self.current_time = 0.0

        self.vehicles_state = {}
        self.pending_subtasks = []
        self.active_subtasks = {}
        self.completed_tasks = []

    def reset(
        self,
        seed: Optional[int] = None,
        tasks: List[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:

        super().reset(seed=seed)

        self.current_time = 0.0
        self.current_time = 0.0

        self.breakdowns = {}

        self.vehicles_state = {
            vid: {
                'pos': v_data['init_pos'] if 'init_pos' in v_data else [0, 0],
                'vel': 0.0,
                'track': v_data['track'],
                'status': 'IDLE',
                'loaded': False,
                'current_node': None,
                'target_node': None,
                'path': [],
                'epsilon': 0.0
            }
            for vid, v_data in self.topology.vehicles.items()
        }

        self.pending_subtasks = copy.deepcopy(tasks) if tasks else []
        self.active_subtasks = {}
        self.completed_tasks = []

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:

        reward = 0.0

        if len(self.pending_subtasks) > 0 and 0 <= action < self.num_vehicles:

            assigned_vehicle = self.vehicle_ids[action]
            task = self.pending_subtasks[0]

            if (
                self.vehicles_state[assigned_vehicle]['status'] == 'IDLE'
                and
                self.vehicles_state[assigned_vehicle]['track']
                == task['required_track']
            ):

                self.pending_subtasks.pop(0)

                self.active_subtasks[assigned_vehicle] = task

                self.vehicles_state[assigned_vehicle]['status'] = 'MOVING'

                self.vehicles_state[assigned_vehicle]['target_node'] = task['end_node']

            else:
                reward -= 10.0

        event_triggered = False

        while not event_triggered:

            for vid in list(self.breakdowns.keys()):

                self.breakdowns[vid] -= self.dt

                if self.breakdowns[vid] <= 0:

                    self.vehicles_state[vid]['status'] = \
                        self.vehicles_state[vid].get('prev_status', 'IDLE')

                    del self.breakdowns[vid]

            v_cmds = self.cma_controller.resolve_conflicts(
                self.vehicles_state
            )

            for vid, v_state in self.vehicles_state.items():

                if v_state['status'] == 'MOVING':

                    v_state['vel'] = v_cmds.get(vid, 2.0)

                    target_pos = self.topology.graph.nodes[
                        v_state['target_node']
                    ]['pos']

                    curr_pos = v_state['pos']

                    dx = target_pos[0] - curr_pos[0]
                    dy = target_pos[1] - curr_pos[1]

                    dist = math.hypot(dx, dy)

                    step_dist = v_state['vel'] * self.dt

                    if dist <= step_dist:

                        v_state['pos'] = list(target_pos)

                        v_state['status'] = 'UNLOADING'

                        v_state['current_node'] = \
                            v_state['target_node']

                    else:

                        v_state['pos'][0] += \
                            (dx / dist) * step_dist

                        v_state['pos'][1] += \
                            (dy / dist) * step_dist

                elif v_state['status'] in ['LOADING', 'UNLOADING']:

                    v_state['status'] = 'IDLE'

                    if vid in self.active_subtasks:

                        reward += 50.0

                        del self.active_subtasks[vid]

            self.current_time += self.dt

            is_done = (
                len(self.pending_subtasks) == 0
                and
                len(self.active_subtasks) == 0
            )

            can_assign = False

            if len(self.pending_subtasks) > 0:

                req_track = self.pending_subtasks[0]['required_track']

                for v in self.vehicles_state.values():

                    if (
                        v['status'] == 'IDLE'
                        and
                        v['track'] == req_track
                    ):

                        can_assign = True
                        break

            if is_done or can_assign:
                event_triggered = True

            if self.current_time > 10000.0:

                event_triggered = True

                done = True

                reward -= 1000.0

        is_timeout = self.current_time > 10000.0

        is_finished = (
            len(self.pending_subtasks) == 0
            and
            len(self.active_subtasks) == 0
        )

        done = is_finished or is_timeout

        info = self._get_info()

        reward += self._calculate_step_reward()

        if info.get('conflict_risk'):
            reward -= 50.0

        return (
            self._get_obs(),
            reward,
            done,
            False,
            self._get_info()
        )

    def trigger_breakdown(
        self,
        vehicle_id: str,
        duration: float
    ):

        if (
            vehicle_id in self.vehicles_state
            and
            self.vehicles_state[vehicle_id]['status'] != 'ERROR'
        ):

            self.breakdowns[vehicle_id] = duration

            self.vehicles_state[vehicle_id]['prev_status'] = \
                self.vehicles_state[vehicle_id]['status']

            self.vehicles_state[vehicle_id]['status'] = 'ERROR'

            self.vehicles_state[vehicle_id]['vel'] = 0.0

    def _get_obs(self) -> np.ndarray:

        obs = []

        for vid in self.vehicle_ids:

            v = self.vehicles_state[vid]

            obs.extend([
                v['pos'][0] / 30.0,
                v['pos'][1] / 30.0,
                1.0 if v['loaded'] else 0.0,
                1.0 if v['status'] == 'IDLE' else 0.0
            ])

        track_loads = {
            t_id: 0
            for t_id in self.topology.tracks.keys()
        }

        for v in self.vehicles_state.values():

            if v['track'] in track_loads:
                track_loads[v['track']] += 1

        obs.extend([
            count / self.num_vehicles
            for count in track_loads.values()
        ])

        if self.pending_subtasks:

            task = self.pending_subtasks[0]

            start_pos = self.topology.graph.nodes[
                task['start_node']
            ]['pos']

            end_pos = self.topology.graph.nodes[
                task['end_node']
            ]['pos']

            obs.extend([
                start_pos[0] / 30.0,
                start_pos[1] / 30.0,
                end_pos[0] / 30.0,
                end_pos[1] / 30.0,
                task.get('urgency', 0.5),
                1.0
            ])

        else:

            obs.extend([
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ])

        obs = np.array(obs, dtype=np.float32)

        if len(obs) < self.state_dim:

            obs = np.pad(
                obs,
                (0, self.state_dim - len(obs)),
                'constant'
            )

        return obs[:self.state_dim]

    def _calculate_step_reward(self) -> float:

        penalty = 0.0

        penalty -= 0.1

        for vid, v in self.vehicles_state.items():

            if v['epsilon'] > 0.5:
                penalty -= 2.0

        return penalty

    def _get_info(self) -> Dict:

        active_conflicts = set()

        v_ids = list(self.vehicles_state.keys())

        for i in range(len(v_ids)):

            for j in range(i + 1, len(v_ids)):

                v1 = self.vehicles_state[v_ids[i]]
                v2 = self.vehicles_state[v_ids[j]]

                if (
                    v1['track'] == v2['track']
                    and
                    (
                        v1['status'] in ['MOVING', 'ERROR']
                        and
                        v2['status'] in ['MOVING', 'ERROR']
                    )
                ):

                    if (
                        v1['status'] == 'ERROR'
                        and
                        v2['status'] == 'ERROR'
                    ):
                        continue

                    dist = math.hypot(
                        v1['pos'][0] - v2['pos'][0],
                        v1['pos'][1] - v2['pos'][1]
                    )

                    if dist < 2.0:

                        active_conflicts.add(
                            tuple(sorted((v_ids[i], v_ids[j])))
                        )

        return {
            'time': self.current_time,
            'active_conflicts': active_conflicts,
            'conflict_risk': len(active_conflicts) > 0,
            'active_tasks_count': len(self.active_subtasks),
            'pending_tasks_count': len(self.pending_subtasks)
        }