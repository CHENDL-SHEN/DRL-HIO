import random
import json
import os
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
import bisect
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

@dataclass
class ProductionPlan:
    pono: int  
    start_ld: str  
    end_cc: str  
    refine_process: str  
    lf_station: Optional[str]  
    rh_station: Optional[str]  
    
    task_start_time: datetime  
    task_end_time: datetime  
    
    lf_start_time: Optional[datetime]  
    lf_end_time: Optional[datetime]  
    rh_start_time: Optional[datetime]  
    rh_end_time: Optional[datetime]  
    
    lf_duration: Optional[int]  
    rh_duration: Optional[int]  
    
    ld_to_lf_duration: Optional[int]  
    ld_to_rh_duration: Optional[int]  
    lf_to_rh_duration: Optional[int]  
    lf_to_cc_duration: Optional[int]  
    rh_to_cc_duration: Optional[int]  

def time_to_str(time_obj: datetime, include_date: bool = False) -> str:
    if include_date:
        return time_obj.strftime("%Y-%m-%d %H:%M:%S")
    return time_obj.strftime("%H:%M:%S")

def str_to_time(time_str: str, base_time: datetime = None) -> datetime:
    time_obj = datetime.strptime(time_str, "%H:%M:%S")
    if base_time is None:
        base_time = datetime(2026, 1, 1, 0, 0, 0)
    return time_obj.replace(year=base_time.year, month=base_time.month, day=base_time.day)


class TaskGenerator:
    LD_INTERVAL_MINUTES = 30      
    INITIAL_LD_BOOKING_OFFSET = 40  
    MIN_DURATION = 1              
    STATION_GAP_MINUTES = 5       
    
    TRANSPORT_ALPHA = 0.1         
    TRANSPORT_BETA = 0.05          
    DEFAULT_TRANSPORT_TIME = 10   
    TASK_INTERVAL_MIN = 10        
    TASK_INTERVAL_MAX = 20        
    
    def __init__(self, seed: int = None):
        self._init_resources()
        self._init_duration_config()
        self._init_transport_data()
        self._init_bookings()
        
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
            print(f"Warning: Random seed not set, results will not be reproducible. Seed: {seed}")
        else:
            print(f"Seed set successfully: {seed}")
        self._set_random_seed(seed)
    
    def _init_resources(self):
        self.start_lds = ["1LD", "2LD", "3LD"]
        self.end_ccs = ["1CC", "2CC", "3CC"]
        self.rh_stations = ["1RH", "2RH", "4RH"]
        self.lf_stations = ["1LF", "2LF", "4LF"]
        self.refine_processes = ["LF Refining", "RH Refining", "LF+RH Refining"]
    
    def _init_duration_config(self):
        self.refine_duration_config: Dict[str, Dict] = {
            "LF Refining": {"base": 60, "fluctuation": 20},
            "RH Refining": {"base": 30, "fluctuation": 15},
            "LF+RH Refining": {
                "LF": {"base": 60, "fluctuation": 20},
                "RH": {"base": 30, "fluctuation": 15}
            }
        }
    
    def _init_transport_data(self):
        self.transport_data = {
            "LD_LF": self._create_transport_dict([
                (("1LD", "1LF"), 5.0), (("1LD", "2LF"), 10.4), (("1LD", "4LF"), 7.0),
                (("2LD", "1LF"), 6.0), (("2LD", "2LF"), 11.4), (("2LD", "4LF"), 6.0),
                (("3LD", "1LF"), 7.0), (("3LD", "2LF"), 12.4), (("3LD", "4LF"), 5.0)
            ]),
            "LD_RH": self._create_transport_dict([
                (("1LD", "1RH"), 12.6), (("1LD", "2RH"), 8.0), (("1LD", "4RH"), 12.4),
                (("2LD", "1RH"), 11.6), (("2LD", "2RH"), 7.0), (("2LD", "4RH"), 13.4),
                (("3LD", "1RH"), 10.6), (("3LD", "2RH"), 6.0), (("3LD", "4RH"), 14.4)
            ]),
            "LF_RH": self._create_transport_dict([
                (("1LF", "1RH"), 12.6), (("1LF", "2RH"), 8.4), (("1LF", "4RH"), 10.4),
                (("2LF", "1RH"), 7.6), (("2LF", "2RH"), 13.6), (("2LF", "4RH"), 5.4),
                (("4LF", "1RH"), 5.6), (("4LF", "2RH"), 4.0), (("4LF", "4RH"), 14.4)
            ]),
            "LF_CC": self._create_transport_dict([
                (("1LF", "1CC"), 7.6), (("1LF", "2CC"), 8.6), (("1LF", "3CC"), 8.0),
                (("2LF", "1CC"), 13.4), (("2LF", "2CC"), 14.4), (("2LF", "3CC"), 2.6),
                (("4LF", "1CC"), 3.6), (("4LF", "2CC"), 4.6), (("4LF", "3CC"), 12.0)
            ]),
            "RH_CC": self._create_transport_dict([
                (("1RH", "1CC"), 6.4), (("1RH", "2CC"), 7.4), (("1RH", "3CC"), 7.4),
                (("2RH", "1CC"), 2.6), (("2RH", "2CC"), 3.6), (("2RH", "3CC"), 13.0),
                (("4RH", "1CC"), 15.0), (("4RH", "2CC"), 16.0), (("4RH", "3CC"), 2.6)
            ])
        }
    
    def _create_transport_dict(self, data_list: List[Tuple[Tuple[str, str], float]]) -> Dict:
        return {station_pair: {"round_trip_min_time": round_trip}
                for station_pair, round_trip in data_list}
    
    def _init_bookings(self):
        self.station_bookings: Dict[str, List[Tuple[datetime, datetime]]] = {}
        self.ld_bookings: Dict[str, datetime] = {}
    
    def _set_random_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_tasks(self, task_num: int, first_task_start: str = "00:00:00") -> List[ProductionPlan]:
        tasks = []
        last_task_start = str_to_time(first_task_start)
        
        self.ld_bookings = {
            station: str_to_time(first_task_start) - timedelta(minutes=self.INITIAL_LD_BOOKING_OFFSET)
            for station in self.start_lds
        }
        
        base_process_count = task_num // len(self.refine_processes)
        process_remainder = task_num % len(self.refine_processes)
        assigned_processes = []
        for process in self.refine_processes:
            assigned_processes.extend([process] * base_process_count)
        if process_remainder > 0:
            assigned_processes.extend(random.sample(self.refine_processes, process_remainder))
        random.shuffle(assigned_processes) 
        
        base_cc_count = task_num // len(self.end_ccs)
        cc_remainder = task_num % len(self.end_ccs)
        assigned_ccs = []
        for cc in self.end_ccs:
            assigned_ccs.extend([cc] * base_cc_count)
        if cc_remainder > 0:
            assigned_ccs.extend(random.sample(self.end_ccs, cc_remainder))
        random.shuffle(assigned_ccs) 
        
        for pono in range(task_num):
            task = self._create_single_task(
                pono, 
                first_task_start, 
                last_task_start, 
                assigned_processes[pono], 
                assigned_ccs[pono]
            )
            tasks.append(task)
            last_task_start = task.task_start_time
            
        return tasks
    
    def _create_single_task(self, pono: int, first_task_start: str, last_task_start: datetime, 
                            pre_assigned_process: str, pre_assigned_cc: str) -> ProductionPlan:
        start_ld, base_task_start = self._select_ld_and_start_time(pono, first_task_start, last_task_start)
        end_cc = pre_assigned_cc
        refine_process = pre_assigned_process
        
        lf_duration, rh_duration = self._calculate_process_durations(refine_process)
        
        station_info = self._calculate_optimal_station(start_ld, end_cc, refine_process, base_task_start, lf_duration, rh_duration)
        actual_task_start = station_info['actual_task_start']
        
        self.ld_bookings[start_ld] = actual_task_start
        
        lf_start, lf_end, rh_start, rh_end, task_end = self._calculate_time_axis(
            refine_process, actual_task_start, station_info, lf_duration, rh_duration
        )
        
        return ProductionPlan(
            pono=pono, start_ld=start_ld, end_cc=end_cc, refine_process=refine_process,
            lf_station=station_info.get('lf_station'), rh_station=station_info.get('rh_station'),
            task_start_time=actual_task_start, task_end_time=task_end,
            lf_start_time=lf_start, lf_end_time=lf_end, rh_start_time=rh_start, rh_end_time=rh_end,
            lf_duration=lf_duration, rh_duration=rh_duration,
            ld_to_lf_duration=station_info.get('ld_to_lf'), ld_to_rh_duration=station_info.get('ld_to_rh'),
            lf_to_rh_duration=station_info.get('lf_to_rh'), lf_to_cc_duration=station_info.get('lf_to_cc'),
            rh_to_cc_duration=station_info.get('rh_to_cc')
        )
    
    def _select_ld_and_start_time(self, pono: int, first_task_start: str, last_task_start: datetime) -> Tuple[str, datetime]:
        if pono == 0:
            return random.choice(self.start_lds), str_to_time(first_task_start)
        
        base_start = last_task_start + timedelta(minutes=random.randint(self.TASK_INTERVAL_MIN, self.TASK_INTERVAL_MAX))
        
        ld_available_times = {}
        for ld in self.start_lds:
            ld_ready_time = self.ld_bookings[ld] + timedelta(minutes=self.LD_INTERVAL_MINUTES)
            ld_available_times[ld] = max(base_start, ld_ready_time)
            
        earliest_possible_time = min(ld_available_times.values())
        best_lds = [ld for ld, t in ld_available_times.items() if t == earliest_possible_time]
        selected_ld = random.choice(best_lds)
        
        return selected_ld, earliest_possible_time

    def _calculate_process_durations(self, refine_process: str) -> Tuple[Optional[int], Optional[int]]:
        lf_duration, rh_duration = None, None
        if refine_process == "LF Refining":
            lf_duration = self._calculate_single_refine_duration("LF Refining")
        elif refine_process == "RH Refining":
            rh_duration = self._calculate_single_refine_duration("RH Refining")
        elif refine_process == "LF+RH Refining":
            lf_duration = self._calculate_single_refine_duration("LF Refining")
            rh_duration = self._calculate_single_refine_duration("RH Refining")
        return lf_duration, rh_duration
    
    def _calculate_single_refine_duration(self, process_type: str) -> int:
        config = self.refine_duration_config[process_type]
        return config["base"] + random.randint(-config["fluctuation"], config["fluctuation"])
    
    def _calculate_optimal_station(self, start_ld: str, end_cc: str, refine_process: str, 
                                   base_task_start: datetime, lf_duration: int, rh_duration: int) -> Dict:
        possible_combinations = []
        
        if refine_process == "LF Refining":
            possible_combinations = self._evaluate_lf_combinations(start_ld, end_cc, base_task_start, lf_duration)
        elif refine_process == "RH Refining":
            possible_combinations = self._evaluate_rh_combinations(start_ld, end_cc, base_task_start, rh_duration)
        elif refine_process == "LF+RH Refining":
            possible_combinations = self._evaluate_double_combinations(start_ld, end_cc, base_task_start, lf_duration, rh_duration)
        
        if not possible_combinations:
            raise ValueError(f"No valid combination found: Process={refine_process}, LD={start_ld}, CC={end_cc}")
        
        best = min(possible_combinations, key=lambda x: x['cost'])
        return best
    
    def _evaluate_lf_combinations(self, start_ld: str, end_cc: str, base_task_start: datetime, lf_duration: int) -> List[Dict]:
        combinations = []
        for lf_st in self.lf_stations:
            try:
                ld_to_lf = self._calculate_transport_duration("LD_LF", start_ld, lf_st)
                lf_to_cc = self._calculate_transport_duration("LF_CC", lf_st, end_cc)
                
                rel_lf_start = timedelta(minutes=ld_to_lf)
                rel_lf_end = rel_lf_start + timedelta(minutes=lf_duration)
                
                required_blocks = [(lf_st, rel_lf_start, rel_lf_end)]
                actual_task_start = self._find_no_wait_chain_start_time(base_task_start, required_blocks)
                
                task_end = actual_task_start + rel_lf_end + timedelta(minutes=lf_to_cc)
                combinations.append({
                    'lf_station': lf_st, 'rh_station': None,
                    'ld_to_lf': ld_to_lf, 'lf_to_cc': lf_to_cc,
                    'actual_task_start': actual_task_start,
                    'cost': (task_end - base_task_start).total_seconds() / 60
                })
            except ValueError:
                continue
        return combinations
    
    def _evaluate_rh_combinations(self, start_ld: str, end_cc: str, base_task_start: datetime, rh_duration: int) -> List[Dict]:
        combinations = []
        for rh_st in self.rh_stations:
            try:
                ld_to_rh = self._calculate_transport_duration("LD_RH", start_ld, rh_st)
                rh_to_cc = self._calculate_transport_duration("RH_CC", rh_st, end_cc)
                
                rel_rh_start = timedelta(minutes=ld_to_rh)
                rel_rh_end = rel_rh_start + timedelta(minutes=rh_duration)
                
                required_blocks = [(rh_st, rel_rh_start, rel_rh_end)]
                actual_task_start = self._find_no_wait_chain_start_time(base_task_start, required_blocks)
                
                task_end = actual_task_start + rel_rh_end + timedelta(minutes=rh_to_cc)
                combinations.append({
                    'lf_station': None, 'rh_station': rh_st,
                    'ld_to_rh': ld_to_rh, 'rh_to_cc': rh_to_cc,
                    'actual_task_start': actual_task_start,
                    'cost': (task_end - base_task_start).total_seconds() / 60
                })
            except ValueError:
                continue
        return combinations
    
    def _evaluate_double_combinations(self, start_ld: str, end_cc: str, base_task_start: datetime, lf_duration: int, rh_duration: int) -> List[Dict]:
        combinations = []
        for lf_st in self.lf_stations:
            for rh_st in self.rh_stations:
                try:
                    ld_to_lf = self._calculate_transport_duration("LD_LF", start_ld, lf_st)
                    lf_to_rh = self._calculate_transport_duration("LF_RH", lf_st, rh_st)
                    rh_to_cc = self._calculate_transport_duration("RH_CC", rh_st, end_cc)
                    
                    rel_lf_start = timedelta(minutes=ld_to_lf)
                    rel_lf_end = rel_lf_start + timedelta(minutes=lf_duration)
                    rel_rh_start = rel_lf_end + timedelta(minutes=lf_to_rh)
                    rel_rh_end = rel_rh_start + timedelta(minutes=rh_duration)
                    
                    required_blocks = [
                        (lf_st, rel_lf_start, rel_lf_end),
                        (rh_st, rel_rh_start, rel_rh_end)
                    ]
                    
                    actual_task_start = self._find_no_wait_chain_start_time(base_task_start, required_blocks)
                    task_end = actual_task_start + rel_rh_end + timedelta(minutes=rh_to_cc)
                    
                    combinations.append({
                        'lf_station': lf_st, 'rh_station': rh_st,
                        'ld_to_lf': ld_to_lf, 'lf_to_rh': lf_to_rh, 'rh_to_cc': rh_to_cc,
                        'actual_task_start': actual_task_start,
                        'cost': (task_end - base_task_start).total_seconds() / 60
                    })
                except ValueError:
                    continue
        return combinations

    def _calculate_time_axis(self, refine_process: str, actual_task_start: datetime, station_info: Dict,
                             lf_duration: Optional[int], rh_duration: Optional[int]) -> Tuple:
        lf_station, rh_station = station_info.get('lf_station'), station_info.get('rh_station')
        ld_to_lf, ld_to_rh = station_info.get('ld_to_lf'), station_info.get('ld_to_rh')
        lf_to_rh, lf_to_cc, rh_to_cc = station_info.get('lf_to_rh'), station_info.get('lf_to_cc'), station_info.get('rh_to_cc')
        
        lf_start, lf_end, rh_start, rh_end = None, None, None, None
        
        if refine_process == "LF Refining":
            lf_start = actual_task_start + timedelta(minutes=ld_to_lf)
            lf_end = lf_start + timedelta(minutes=lf_duration)
            task_end = lf_end + timedelta(minutes=lf_to_cc)
            self._book_station(lf_station, lf_start, lf_end)
            
        elif refine_process == "RH Refining":
            rh_start = actual_task_start + timedelta(minutes=ld_to_rh)
            rh_end = rh_start + timedelta(minutes=rh_duration)
            task_end = rh_end + timedelta(minutes=rh_to_cc)
            self._book_station(rh_station, rh_start, rh_end)
            
        elif refine_process == "LF+RH Refining":
            lf_start = actual_task_start + timedelta(minutes=ld_to_lf)
            lf_end = lf_start + timedelta(minutes=lf_duration)
            rh_start = lf_end + timedelta(minutes=lf_to_rh)
            rh_end = rh_start + timedelta(minutes=rh_duration)
            task_end = rh_end + timedelta(minutes=rh_to_cc)
            self._book_station(lf_station, lf_start, lf_end)
            self._book_station(rh_station, rh_start, rh_end)
        
        return lf_start, lf_end, rh_start, rh_end, task_end

    def _get_booking_key(self, station_id: str) -> str:
        station_num = ''.join(filter(str.isdigit, station_id))
        return f"REFINE_GROUP_{station_num}"

    def _find_no_wait_chain_start_time(self, base_start: datetime, required_blocks: List[Tuple[str, timedelta, timedelta]]) -> datetime:
        current_start = base_start
        
        while True:
            conflict_found = False
            for station_id, rel_start, rel_end in required_blocks:
                abs_start = current_start + rel_start
                abs_end = current_start + rel_end
                group_key = self._get_booking_key(station_id)
                
                if group_key in self.station_bookings:
                    for b_start, b_end in self.station_bookings[group_key]:
                        if not (abs_end <= b_start or abs_start >= b_end):
                            current_start = b_end + timedelta(minutes=self.STATION_GAP_MINUTES) - rel_start
                            conflict_found = True
                            break
                if conflict_found:
                    break
            
            if not conflict_found:
                return current_start

    def _book_station(self, station_id: str, start_time: datetime, end_time: datetime):
        group_key = self._get_booking_key(station_id)
        if group_key not in self.station_bookings:
            self.station_bookings[group_key] = []
        bisect.insort(self.station_bookings[group_key], (start_time, end_time))

    def _calculate_transport_duration(self, transport_type: str, start_station: str, end_station: str) -> int:
        if transport_type in self.transport_data and (start_station, end_station) in self.transport_data[transport_type]:
            data = self.transport_data[transport_type][(start_station, end_station)]
            return self._generate_actual_transport_time(data["round_trip_min_time"])
        raise ValueError(f"Transport data not found: Type={transport_type}, Start={start_station}, End={end_station}")
    
    def _generate_actual_transport_time(self, round_trip_min_time: float) -> int:
        mu = round_trip_min_time * (1 + self.TRANSPORT_ALPHA)
        sigma = mu * self.TRANSPORT_BETA
        t_rand = np.random.normal(mu, sigma)
        return max(math.ceil(t_rand), int(round_trip_min_time))
    
    def save_tasks_to_json(self, tasks: List[ProductionPlan], save_path: str = "./data/tasks.json") -> bool:
        try:
            task_dicts = []
            for task in tasks:
                task_dict = {
                    "pono": task.pono, "start_ld": task.start_ld, "end_cc": task.end_cc,
                    "refine_process": task.refine_process, "lf_station": task.lf_station, "rh_station": task.rh_station,
                    "time_info": {
                        "task_start": time_to_str(task.task_start_time), "task_end": time_to_str(task.task_end_time),
                        "lf_start": time_to_str(task.lf_start_time) if task.lf_start_time else None,
                        "lf_end": time_to_str(task.lf_end_time) if task.lf_end_time else None,
                        "rh_start": time_to_str(task.rh_start_time) if task.rh_start_time else None,
                        "rh_end": time_to_str(task.rh_end_time) if task.rh_end_time else None
                    },
                    "duration_info": {"lf_duration": task.lf_duration, "rh_duration": task.rh_duration},
                    "transport_info": {
                        "ld_to_lf": self._calc_time_diff(task.task_start_time, task.lf_start_time) if task.lf_start_time else None,
                        "ld_to_rh": self._calc_time_diff(task.task_start_time, task.rh_start_time) if task.rh_start_time and task.refine_process != "LF+RH Refining" else None,
                        "lf_to_rh": self._calc_time_diff(task.lf_end_time, task.rh_start_time) if task.lf_end_time and task.rh_start_time else None,
                        "lf_to_cc": self._calc_time_diff(task.lf_end_time, task.task_end_time) if task.lf_end_time and task.refine_process != "LF+RH Refining" else None,
                        "rh_to_cc": self._calc_time_diff(task.rh_end_time, task.task_end_time) if task.rh_end_time else None
                    }
                }
                task_dicts.append(task_dict)
            
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(task_dicts, f, ensure_ascii=False, indent=4)
            print(f"Task data saved successfully to: {os.path.abspath(save_path)}")
            return True
        except Exception as e:
            print(f"Error saving task data: {e}")
            return False
            
    def _calc_time_diff(self, start: Optional[datetime], end: Optional[datetime]) -> Optional[int]:
        if start and end:
            return int((end - start).total_seconds() / 60)
        return None
    
    def generate_gantt_chart(self, tasks: List[ProductionPlan], save_path: str = "./data/gantt_chart.png") -> bool:
        try:
            plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Arial', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            fig, (ax_station, ax_task) = plt.subplots(2, 1, figsize=(14, 14), sharex=True)
            
            def get_task_color(pono):
                import colorsys
                hue = (pono * 0.618033988749895) % 1.0
                saturation = 0.7 + (pono % 3) * 0.1
                value = 0.8 + (pono % 2) * 0.1
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            
            task_colors = {task.pono: get_task_color(task.pono) for task in tasks}
            process_colors = {'transport': '#FFFF99', 'lf_process': '#1f77b4', 'rh_process': '#2ca02c'}
            
            all_stations = {task.lf_station for task in tasks if task.lf_station}
            all_stations.update(task.rh_station for task in tasks if task.rh_station)
            station_order = sorted([s for s in all_stations if 'LF' in s]) + sorted([s for s in all_stations if 'RH' in s])
            station_y = {station: i for i, station in enumerate(station_order)}
            legend_added = set()
            
            for task in tasks:
                pono, color = task.pono, task_colors.get(task.pono, 'gray')
                if task.lf_station and task.lf_start_time and task.lf_end_time:
                    label = f'Task {pono}' if pono not in legend_added else ""
                    ax_station.barh(station_y[task.lf_station], task.lf_end_time - task.lf_start_time,
                                   left=task.lf_start_time, height=0.6, color=color, edgecolor='black', alpha=0.8, label=label)
                    ax_station.text(task.lf_start_time + (task.lf_end_time - task.lf_start_time) / 2, station_y[task.lf_station],
                                   f"Task {pono}", va='center', ha='center', fontsize=9)
                    legend_added.add(pono)
                if task.rh_station and task.rh_start_time and task.rh_end_time:
                    label = f'Task {pono}' if pono not in legend_added else ""
                    ax_station.barh(station_y[task.rh_station], task.rh_end_time - task.rh_start_time,
                                   left=task.rh_start_time, height=0.6, color=color, edgecolor='black', alpha=0.8, label=label)
                    ax_station.text(task.rh_start_time + (task.rh_end_time - task.rh_start_time) / 2, station_y[task.rh_station],
                                   f"Task {pono}", va='center', ha='center', fontsize=9)
                    legend_added.add(pono)
            
            ax_station.set_yticks([station_y[s] for s in station_order] if station_order else [])
            ax_station.set_yticklabels(station_order, fontsize=11)
            ax_station.grid(True, axis='x', alpha=0.5, linestyle='--')
            ax_station.set_title('Ladle Processing Scheduling Gantt Chart - Station View (With Resource Mutation Constraints)', fontsize=16, fontweight='bold')
            ax_station.set_ylabel('Station', fontsize=14)
            ax_station.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            
            task_y = {task.pono: i for i, task in enumerate(tasks)}
            for task in tasks:
                row = task_y[task.pono]
                color = task_colors.get(task.pono, 'gray')
                ax_task.barh(row, task.task_end_time - task.task_start_time, left=task.task_start_time,
                            height=0.8, color=color, edgecolor='black', alpha=0.3)
                
                if task.refine_process == "LF Refining":
                    ax_task.barh(row, task.lf_start_time - task.task_start_time, left=task.task_start_time,
                                height=0.5, color=process_colors['transport'], edgecolor='black', alpha=0.7)
                    ax_task.text(task.task_start_time + (task.lf_start_time - task.task_start_time) / 2, row,
                                task.start_ld, va='center', ha='center', fontsize=9, color='blue')
                    ax_task.barh(row, task.lf_end_time - task.lf_start_time, left=task.lf_start_time,
                                height=0.5, color=process_colors['lf_process'], edgecolor='black', alpha=1.0)
                    ax_task.barh(row, task.task_end_time - task.lf_end_time, left=task.lf_end_time,
                                height=0.5, color=process_colors['transport'], edgecolor='black', alpha=0.7)
                    ax_task.text(task.lf_end_time + (task.task_end_time - task.lf_end_time) / 2, row,
                                task.end_cc, va='center', ha='center', fontsize=9, color='red')
                
                elif task.refine_process == "RH Refining":
                    ax_task.barh(row, task.rh_start_time - task.task_start_time, left=task.task_start_time,
                                height=0.5, color=process_colors['transport'], edgecolor='black', alpha=0.7)
                    ax_task.text(task.task_start_time + (task.rh_start_time - task.task_start_time) / 2, row,
                                task.start_ld, va='center', ha='center', fontsize=9, color='blue')
                    ax_task.barh(row, task.rh_end_time - task.rh_start_time, left=task.rh_start_time,
                                height=0.5, color=process_colors['rh_process'], edgecolor='black', alpha=1.0)
                    ax_task.barh(row, task.task_end_time - task.rh_end_time, left=task.rh_end_time,
                                height=0.5, color=process_colors['transport'], edgecolor='black', alpha=0.7)
                    ax_task.text(task.rh_end_time + (task.task_end_time - task.rh_end_time) / 2, row,
                                task.end_cc, va='center', ha='center', fontsize=9, color='red')
                
                elif task.refine_process == "LF+RH Refining":
                    ax_task.barh(row, task.lf_start_time - task.task_start_time, left=task.task_start_time,
                                height=0.5, color=process_colors['transport'], edgecolor='black', alpha=0.7)
                    ax_task.text(task.task_start_time + (task.lf_start_time - task.task_start_time) / 2, row,
                                task.start_ld, va='center', ha='center', fontsize=9, color='blue')
                    ax_task.barh(row, task.lf_end_time - task.lf_start_time, left=task.lf_start_time,
                                height=0.5, color=process_colors['lf_process'], edgecolor='black', alpha=1.0)
                    ax_task.barh(row, task.rh_start_time - task.lf_end_time, left=task.lf_end_time,
                                height=0.5, color=process_colors['transport'], edgecolor='black', alpha=0.7)
                    ax_task.barh(row, task.rh_end_time - task.rh_start_time, left=task.rh_start_time,
                                height=0.5, color=process_colors['rh_process'], edgecolor='black', alpha=1.0)
                    ax_task.barh(row, task.task_end_time - task.rh_end_time, left=task.rh_end_time,
                                height=0.5, color=process_colors['transport'], edgecolor='black', alpha=0.7)
                    ax_task.text(task.rh_end_time + (task.task_end_time - task.rh_end_time) / 2, row,
                                task.end_cc, va='center', ha='center', fontsize=9, color='red')
                
                ax_task.text(task.task_start_time + (task.task_end_time - task.task_start_time) / 2, row,
                            task.refine_process, va='center', ha='center', fontweight='bold')
            
            ax_task.set_yticks([task_y[t.pono] for t in tasks])
            ax_task.set_yticklabels([f"Task {t.pono}" for t in tasks], fontsize=11)
            ax_task.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xticks(rotation=45, fontsize=10)
            ax_task.grid(True, axis='x', alpha=0.5, linestyle='--')
            ax_task.set_title('Ladle Processing Scheduling Gantt Chart - Task View (No-Wait Constraint)', fontsize=16, fontweight='bold')
            ax_task.set_xlabel('Time', fontsize=14)
            ax_task.set_ylabel('Task', fontsize=14)
            
            import matplotlib.patches as mpatches
            process_legend = [
                mpatches.Patch(facecolor=process_colors['transport'], edgecolor='black', linewidth=1, label='Transport'),
                mpatches.Patch(facecolor=process_colors['lf_process'], edgecolor='black', linewidth=1, label='LF Refining'),
                mpatches.Patch(facecolor=process_colors['rh_process'], edgecolor='black', linewidth=1, label='RH Refining')
            ]
            ax_task.legend(handles=process_legend, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            
            plt.tight_layout()
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gantt chart saved successfully to: {os.path.abspath(save_path)}")
            return True
        except Exception as e:
            print(f"Failed to generate Gantt chart: {e}")
            return False

    def check_task_time_consistency(self, tasks: List[ProductionPlan]) -> bool:
        all_valid = True
        for task in tasks:
            actual_total_time = int((task.task_end_time - task.task_start_time).total_seconds() / 60)
            expected_total_time = 0
            
            if task.refine_process == "LF Refining":
                expected_total_time = (task.ld_to_lf_duration or 0) + (task.lf_duration or 0) + (task.lf_to_cc_duration or 0)
            elif task.refine_process == "RH Refining":
                expected_total_time = (task.ld_to_rh_duration or 0) + (task.rh_duration or 0) + (task.rh_to_cc_duration or 0)
            elif task.refine_process == "LF+RH Refining":
                expected_total_time = (task.ld_to_lf_duration or 0) + (task.lf_duration or 0) + \
                                     (task.lf_to_rh_duration or 0) + (task.rh_duration or 0) + \
                                     (task.rh_to_cc_duration or 0)
            
            if abs(actual_total_time - expected_total_time) > 1:
                all_valid = False
                print(f"Task {task.pono} time calculation anomaly:")
                print(f"   Difference: {abs(actual_total_time - expected_total_time)} minutes")
        
        if all_valid:
            print(f"All {len(tasks)} tasks verified successfully. Consistency check passed under strictly closed no-wait constraints.")
        return all_valid


if __name__ == "__main__":
    task_num = 300
    first_task_start = "00:00:00"
    
    generator = TaskGenerator(seed=629978177)
    print(f"Generating {task_num} tasks...")
    task_list = generator.generate_tasks(task_num=task_num, first_task_start=first_task_start)
    
    generator.save_tasks_to_json(task_list)
    generator.check_task_time_consistency(task_list)
    generator.generate_gantt_chart(task_list)