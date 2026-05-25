import math

class ConflictModalityAnalysis:
    def __init__(self, safe_dist=2.0, kappa=0.5):
        self.safe_dist = safe_dist
        self.kappa = kappa

    def calculate_priority(self, vehicle: dict, is_loaded: bool, urgency: float) -> float:
        
        if vehicle.get('status') == 'IDLE':
            return -100.0
        alpha_1, alpha_2 = 10.0, 5.0
        return alpha_1 * int(is_loaded) + alpha_2 * urgency

    def resolve_conflicts(self, vehicles_state: dict) -> dict:
        v_cmds = {vid: (2.0 if v['status'] == 'MOVING' else 0.0) for vid, v in vehicles_state.items()}
        v_ids = list(vehicles_state.keys())
        
        for i in range(len(v_ids)):
            for j in range(i + 1, len(v_ids)):
                v1, v2 = vehicles_state[v_ids[i]], vehicles_state[v_ids[j]]
                if v1['track'] != v2['track'] or (v1['status'] != 'MOVING' and v2['status'] != 'MOVING'):
                    continue
                dist = math.dist(v1['pos'], v2['pos'])
                if dist < self.safe_dist:
                   
                    p1 = self.calculate_priority(v1, v1.get('loaded', False), v1.get('urgency', 0.5))
                    p2 = self.calculate_priority(v2, v2.get('loaded', False), v2.get('urgency', 0.5))
                    if p1 >= p2:
                        v_cmds[v_ids[j]] = 0.0
                    else:
                        v_cmds[v_ids[i]] = 0.0
        return v_cmds