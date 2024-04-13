# (Under Construction)
# iPrism: Characterize and Mitigate Risk by Quantifying Change in Escape Routes
*Paper accepted by DSN 2024*

## Problem Statement
In complex and dynamic real-world situations involving multiple actors, ensuring safety is a significant challenge. This complexity often leads to severe accidents. The current techniques for mitigating safety hazards are not effective because they do not guarantee accessible escape routes and do not specifically address actors contributing to hazards. As a result, these techniques may not provide timely responses. 

To overcome these limitations, we propose a new measure called the safety-threat indicator (STI). This metric helps identify crucial actors by incorporating defensive driving techniques through counterfactual reasoning. We utilize STI to analyze real-world datasets, revealing inherent biases towards safe scenarios. Additionally, we employ it to develop a hazard mitigation policy using reinforcement learning. 

Our approach demonstrates a substantial reduction in the accident rate for advanced autonomous vehicle agents in rare hazardous scenarios—up to a 77% improvement over current state-of-the-art methods. 


## Quantifying Risk



### Algorithm
```python
def compute_escape_routes(M, X, x_ego_t, constants, delta_t, k, N):
    """
    Compute escape routes using reach-tubes.

    :param M: The boundaries of the model
    :param X: The state of other entities from time t to t+k
    :param x_ego_t: The initial state of the ego
    :param constants: Control constants [a_min, a_max, phi_min, phi_max]
    :param delta_t: Time step
    :param k: Total number of time steps
    :param N: Number of samples
    :return: Reach tube from time t to t+k
    """
    a_min, a_max, phi_min, phi_max = constants
    init_cond_dict = {t: [x_ego_t]}

    # Simulation over time
    t = 0
    while t < k:
        next_cond_set = []
        for x_ego in init_cond_dict[t]:
            count = 0
            while count < N:
                # Uniformly sample control inputs within given bounds
                a = uniform(a_min, a_max)
                phi = uniform(phi_min, phi_max)

                # Compute the next state using the Bicycle Model
                x_ego_next = bicycle_model(x_ego, a, phi, delta_t)

                # Check collision and model boundaries
                if is_within_boundaries(x_ego_next, M) and not_collide(x_ego_next, X):
                    next_cond_set.append(x_ego_next)
                count += 1

        init_cond_dict[t + delta_t] = next_cond_set
        t += delta_t

    # Generate the bounded reach tube from the simulation results
    T = bounded_reach_tube(init_cond_dict)
    return T
```

### Results
**Table:** Comparative analysis of Lead-Time-for-Mitigating-Accident (LTFMA) in seconds across various risk metrics. PKL-All: trained on all scenarios. PKL-Holdout: trained on all scenarios except the *ghost cut-in* and the *lead cut-in* scenarios.

| **Metric**        | **Ghost Cut-In**   | **Lead Cut-In**    | **Lead Slowdown**      | **Rear-End**         | **All Scenarios** |
| ----------------- | ------------------ | ------------------ | ---------------------- | -------------------- | ----------------- |
| TTC               | 0.00 (0.00)        | 0.00 (0.00)        | 3.30 (0.89)            | 0.02 (0.17)          | 0.83              |
| Dist. CIPA        | 0.00 (0.00)        | 0.00 (0.00)        | **5.50 (0.89)**        | 0.02 (0.17)          | 1.38              |
| PKL-All           | 0.75 (0.30)        | 1.01 (0.76)        | 1.22 (0.62)            | 0.01 (0.12)          | 0.75              |
| PKL-Holdout       | 0.14 (0.21)        | 3.36 (4.18)        | 1.23 (0.69)            | 0.01 (0.12)          | 1.19              |
| **STI (ours)**    | **2.94 (0.33)**    | **8.37 (0.70)**    | 2.22 (0.23)            | **1.23 (0.11)**      | **3.69**          |

**Notes:**
- We used *LBC agent* as the ADS to control the ego actor to obtain these results.
- In the front accident scenario, the ego actor's ADS (*LBC agent*) avoided the accident, resulting in no LTFMA metric to report.
- *SD* stands for *standard deviation*.



## Mitigating Risk

### Implementation

### Results
**Table:** Comparative analysis of agents' accident prevention rates across scenarios.

**Explanation of Agent Comparison:**
- **LBC+controller w/ STI (LBC+system):** To show improvement over baseline agent.
- **LBC+controller w/o STI:** To show that score is important (ablation study).
- **LBC+TTC-based (ACA):** To show improvement w.r.t. ACA techniques.
- **RIP+controller w/ STI (RIP+system):** To show generalization with other agents.

| **Agent**                              | **Ghost cut-in**                |          |           |         | **Lead cut-in**                 |          |           |         | **Lead slowdown**               |          |           |         |
|----------------------------------------|---------------------------------|----------|-----------|---------|---------------------------------|----------|-----------|---------|---------------------------------|----------|-----------|---------|
|                                        | **CA↑ (%)**                      | **TCR↓ (%)** | **CA↑ (#)**| **TAS** | **CA↑ (%)**                      | **TCR↓ (%)** | **CA↑ (#)**| **TAS** | **CA↑ (%)**                      | **TCR↓ (%))** | **CA↑ (#)**| **TAS** |
| **LBC+controller w/ STI (LBC+system)** | 49%                             | 26.7%    | 252       | 519     | 98%                             | 0.3%     | 167       | 170     | 87%                             | 1.5%     | 103       | 118     |
| LBC+controller w/o STI                 | 1%                              | 51.6%    | 3         | 519     | 2%                              | 16.7%    | 3         | 170     | 86%                             | 1.6%     | 102       | 118     |
| LBC+TTC-based (ACA)                    | 0%                              | 51.9%    | 0         | 519     | 0%                              | 17.0%    | 0         | 170     | 92%                             | 1.0%     | 108       | 118     |
| **RIP+controller w/ STI (RIP+system)** | 86%                             | 6.5%     | 413       | 478     | 61%                             | 26.5%    | 406       | 671     | 71%                             | 12.9%    | 311       | 440     |

**Notes:**
- **CA#** stands for collision avoided (higher is better ↑); 
- **CA%** stands for the percentage of accident scenarios prevented by the mitigation strategy, calculated as `CA (%) = (CA (#) / TAS (#)) × 100`.
- **TCR** stands for total collision rate (lower is better ↓); **ACA** stands for automatic collision avoidance.
- 1000 scenario instances were executed for each scenario and for each baseline agent.
- **TAS** represents the total number of accident scenarios, capturing the accidents experienced by the agents.
- **TCR** is calculated as `TCR (%) = ((TAS(#) - CA(#)) / 1000) × 100`.



## Installation
### Dependencies
- [CARLA 0.9.10](https://carla.readthedocs.io/en/0.9.10/start_quickstart/)
  - Change `<PATH_TO_FILE>/iPrism/DiverseEnv/auto/bin/startsim.sh` to use the `CarlaUE4.sh` in the directory that `CARLA` is installed in
- [OATomobile](https://github.com/OATML/oatomobile)
- Please refer to `requirements.txt` for packages like PyTorch
- Filling in directories for `<PATH_TO_FILE>` at all places
- Setting `PYTHONPATH` as 
    `
    <PATH_TO_FILE>/iPrism/DiverseEnv/auto/sim/CARLA_0.9.10/PythonAPI:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/sim/CARLA_0.9.10:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/sim/CARLA_0.9.10/PythonAPI/carla:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/sim/CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge/scenario_runner:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/team_code:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge:<PATH_TO_FILE>/iPrism/DiverseEnv/carladataset/carla-sim:<PATH_TO_FILE>/iPrism/ReachML/reachml:<PATH_TO_FILE>/iPrism/ReachML/reachml/model:<PATH_TO_FILE>/iPrism/ReachML`
- Place `epoch24.ckpt` (separately uploaded) inside of `<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge/`


### System Requirements
- AMD Ryzen 7 5700x
- NVIDIA GeForce GTX 1080 Ti
- 64 GB Memory
- Ubuntu 18.04.5 LTS


### Inferencing with SMC agent
- Place a trained weight into `<PATH_TO_FILE>/iPrism/DiverseEnv/auto/ram_shared/dqn_sti_online` and rename to `inference_dicts.0.pkl`

    - The script will generate an error if no weights are placed in the above folder

- Comment out corresponding parts in `<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/leaderboard/mitigation_inference_driver.py` to select base agent

- Run `<PATH_TO_FILE>/iPrism/DiverseEnv/auto/mtps_campaign_manager/bin/mtps_campaign_manager.py` with the following arguments:
    ```
    --config
    <PATH_TO_FILE>/iPrism/DiverseEnv/auto/mtps_campaign_manager/bin/configs/<seed_scenario>.json
    --output_dir
    <folder_name>
    --mitigation_mode
    <mode>
    ```
    with options:
  - <seed_scenario>
    1. scen_front_accident.json
    2. scen_ghost_cutin.json
    3. scen_ghost_cutin_curved.json
    4. scen_lead_cutin.json
    5. scen_lead_slowdown.json
  - <mode> (with corresponding python scripts)
    1. `none` (LBC), choosing `mitigation_rl_dqn.py`
    2. `smart` (LBC+SMC), choosing `mitigation_rl_dqn.py`
    3. `rip` (RIP), choosing `mitigation_rl_rip_sti.py`
    4. `rip_smc` (RIP+SMC), choosing `mitigation_rl_rip_sti.py`
  
- Trained weights used in the paper are also provided separately besides the zip file 

- Sample generated traces using the trained weights are also provided in `/sample_data`

- It would take 24 hours to run all 1000 scenarios of a single seed scenario


### Training SMC agent
- Specify `routes` and `scenarios` to train the RL agent on
  - `ghost_cutin_curved` must run on `route_highway_curved.xml`
  - All other scenarios run on `route_highway.xml`
- Select parameters for the training scenario and place them at `<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge/scenario_runner/srunner/scenario_configs`
  - The parameters for agents used in the paper are also provided in `./chosen_training_scenarios` 
- Run `training_loop_standalone_process()` in `mitigation_rl_dqn.py` for using the LBC agent or `mitigation_rl_rip_sti.py` for the RIP agent
  - Select training parameters such as the number of episodes
- The trained weights are saved in `<PATH_TO_FILE>/iPrism/DiverseEnv/auto/ram_shared/dqn_sti_online` and can be selected for inference
- It would take about 48 hours to train an agent from scratch

### Notes
- STI is calculated in `<PATH_TO_FILE>/iPrism/DiverseEnv/carladataset/carla-sim/bev_planning_sim/generate_risk_traj_poly_single_timestamp_simfunction.py`
