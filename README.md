# DRL-HIO
# Abstract
The multi-span transport equipment in steelmaking and continuous casting (SCC) urgently faces scheduling bottlenecks due to the difficulty of balancing global task coordination efficiency and the local operational safety of heterogeneous equipment. This paper formulates this problem using a mixed-integer linear programming (MILP) model and proposes a Deep Reinforcement Learning-based Hierarchical Iterative Optimization algorithm (DRL-HIO) to address issues such as local deadlocks in SCC scheduling. The method is structured into three layers.At the global path planning and task decomposition layer, a physical layout model based on topological mapping is constructed, and a multi-constraint collaborative A* algorithm (MC-A*) is proposed to achieve global task decomposition and subtask encapsulation. At the subtask assignment and local collision avoidance layer, a dynamic task allocation method based on the Contract Net Protocol (CNP) and Multi-factor Utility Evaluation (MUE) is first introduced to enable dynamic matching between subtasks and the optimal Overhead Cranes (OCs); then, an Action Masking Safety Policy (AM-SP) is applied to impose hard anti-collision constraints on OCs and Ferry Cars (FCs), and a DRL-based collision prediction method is proposed to preview motion conflicts. At the conflict resolution and closed-loop verification layer, a discrete-time simulation-based conflict resolution method is employed. Through internal state information exchange and cross-layer anomaly feedback, DRL-HIO forms a complete iterative closed loop for dynamic rescheduling. Experimental results demonstrate that DRL-HIO outperforms traditional scheduling methods in terms of makespan, task delays, and robustness against disturbances.

# Overview
<img width="8044" height="4716" alt="2" src="https://github.com/user-attachments/assets/852d991a-ee7f-412d-8982-441a5da3e148" />

<br>

# Prerequisite
- Ensure you have Python 3.8+ installed along with the required dependencies: pip install torch numpy pandas networkx matplotlib seaborn
- Generate the benchmark scheduling tasks (under strict No-Wait and resource mutation constraints): python utils/task_generator.py
- Train the mid-level scheduling agent using the Conflict-Prioritized Experience Replay (CPER) mechanism: python train_dqn.py

# Running Evaluations
- To evaluate the proposed Proposed-DRL-HIO against Baseline-Rule (SDF) and Baseline-GA across different production scales (10, 30, and 100 global tasks): python run_experiments.py
- To test the dynamic anti-disturbance capability of the framework under a sudden 300-second unexpected equipment breakdown: python run_robustness.py
- Micro-Conflict Modality Micro-Simulations:   
     python plot_cma_rear_end.py   
     python plot_cma_trajectory.py   
     python plot_cma_idle_yielding.py

# Core Methodologies Implemented
- Constraint-Aware $A^*$ (scheduler/upper_planner.py): Computes real-time routing costs by factoring in dynamic edge-congestion coefficients ($\gamma_{load}$) alongside structural transfer penalties ($P_{trans}$).
- Action Masking Safety Shield (scheduler/middle_agent.py): Intercepts reinforcement learning policies via bounding-box predictive intersection sweeps to mask unsafe assignments ($-\infty$) before physical execution.
- CPER Buffer (training/replay_buffer.py): Dynamically balances state-action pairs using a risk factor ($\rho$) to sample high-conflict configurations, significantly speeding up policy convergence.
