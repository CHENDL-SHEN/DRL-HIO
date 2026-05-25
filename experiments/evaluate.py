class PerformanceEvaluator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.start_times = {}
        self.end_times = {}
        self.deadlocks = 0
        self.total_delays = 0.0
        self.task_count = 0
        
    def record_task_start(self, task_id, time):
        if task_id not in self.start_times:
            self.start_times[task_id] = time
            
    def record_task_end(self, task_id, time, deadline):
        self.end_times[task_id] = time
        self.task_count += 1
        if time > deadline:
            self.total_delays += (time - deadline)
            
    def record_deadlock(self):
        self.deadlocks += 1
        
    def get_metrics(self) -> dict:
        if self.task_count == 0: return {}
        
        makespan = max(self.end_times.values()) if self.end_times else 0
        avg_delay = self.total_delays / self.task_count
        
        return {
            'Make_span': makespan,          
            'Avg_Delay': avg_delay,         
            'Deadlocks': self.deadlocks,    
            'Throughput': self.task_count / (makespan + 1e-5) 
        }