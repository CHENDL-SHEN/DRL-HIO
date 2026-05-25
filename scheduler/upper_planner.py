import networkx as nx
import heapq
from env.topology import SCCTopology

class ConstraintAwareAStar:
    def __init__(self, topology: SCCTopology):
        self.topo = topology
        self.G = topology.graph
        self.v_avg = 2.0 
        self.p_trans = 50.0 

    def _heuristic(self, curr_node: str, target_node: str, urgency: float, current_track: str, next_track: str, track_loads: dict) -> float:
        pos_c = self.G.nodes[curr_node]['pos']
        pos_t = self.G.nodes[target_node]['pos']
        dist = abs(pos_c[0] - pos_t[0]) + abs(pos_c[1] - pos_t[1])
        
        w1 = urgency
        w2 = 1.0 - urgency
        
        transfer_penalty = self.p_trans if current_track != next_track else 0.0
        gamma_load = track_loads.get(next_track, 0.0) if track_loads else 0.0
        congestion_penalty = gamma_load * 10.0
        
        return w1 * (dist / self.v_avg) + w2 * congestion_penalty + transfer_penalty

    def find_path(self, start: str, target: str, urgency: float, occupied_nodes: set, track_loads: dict = None) -> list:
        if track_loads is None:
            track_loads = {}
            
        open_set = []
        heapq.heappush(open_set, (0, start, ""))
        g_score = {start: 0}
        came_from = {}
        
        while open_set:
            _, current, curr_track = heapq.heappop(open_set)
            
            if current == target:
                return self._reconstruct_path(came_from, current)
                
            for neighbor in self.G.neighbors(current):
                if neighbor in occupied_nodes:
                    continue 
                    
                edge_data = self.G[current][neighbor]
                next_track = edge_data['track']
                
                dist = edge_data['weight']
                gamma_load = track_loads.get(next_track, 0.0)
                
                congestion_cost = gamma_load * 20.0 
                transfer_cost = self.p_trans if curr_track and curr_track != next_track else 0.0
                
                w1 = urgency
                w2 = 1.0 - urgency
                
                actual_edge_cost = w1 * (dist / self.v_avg) + w2 * congestion_cost + transfer_cost
                tentative_g = g_score[current] + actual_edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, next_track)
                    g_score[neighbor] = tentative_g
                    
                    pos_n = self.G.nodes[neighbor]['pos']
                    pos_t = self.G.nodes[target]['pos']
                    h_dist = abs(pos_n[0] - pos_t[0]) + abs(pos_n[1] - pos_t[1])
                    h = w1 * (h_dist / self.v_avg)
                    
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, neighbor, next_track))
                    
        return []

    def _reconstruct_path(self, came_from: dict, current: str) -> list:
        path = [current]
        tracks = []
        while current in came_from:
            current, track = came_from[current]
            path.append(current)
            tracks.append(track)
        path.reverse()
        tracks.reverse()
        return list(zip(path, tracks + [tracks[-1]] if tracks else []))

    def decompose_task(self, global_task_id: int, path_with_tracks: list) -> list:
        subtasks = []
        if not path_with_tracks: return subtasks
        
        current_subtask_start = path_with_tracks[0][0]
        current_track = path_with_tracks[0][1]
        
        for i in range(1, len(path_with_tracks)):
            node, track = path_with_tracks[i]
            if track != current_track: 
                subtasks.append({
                    'global_id': global_task_id,
                    'start_node': current_subtask_start,
                    'end_node': path_with_tracks[i-1][0],
                    'required_track': current_track
                })
                current_subtask_start = path_with_tracks[i-1][0]
                current_track = track
                
        subtasks.append({
            'global_id': global_task_id,
            'start_node': current_subtask_start,
            'end_node': path_with_tracks[-1][0],
            'required_track': current_track
        })
        return subtasks