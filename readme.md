# iPrism: Characterize and Mitigate Risk by Quantifying Change in Escape Routes
*Paper accepted by DSN 2024*. Introduction, visualization, and other relevant information can be found on the [project website](https://zihengjackchen.github.io/iprism-page/).

## Problem Statement
In complex and dynamic real-world situations involving multiple actors, ensuring safety is a significant challenge. This complexity often leads to severe accidents. We introduce a novel risk metric called the Safety-Threat Indicator (STI), which is inspired by the proactive strategies of experienced drivers to circumvent hazards by evaluating changes in available escape routes for the AV. STI outperforms the state-of-the-art heuristic and data-driven techniques by 2.7 ~ 4.9 times.

To effectively reduce the risks quantified by STI and prevent accidents, we also developed a reinforcement learning-based Safety-hazard Mitigation Controller (SMC). This controller learns optimal policies for risk reduction and accident avoidance. Our approach demonstrates a substantial reduction in the accident rate for advanced autonomous vehicle agents in rare hazardous scenarios—up to a 77% improvement over current state-of-the-art methods. 

### Examples
Please see the [demo](./STI-demo) folder for STI calculation on an example scenario and visit [project website](https://zihengjackchen.github.io/iprism-page/) for more information.

## Installation
This project uses [Git Large File Storage](https://zihengjackchen.github.io/iprism-page/). 

### Dependencies
- [CARLA 0.9.10](https://carla.readthedocs.io/en/0.9.10/start_quickstart/)
  - Change `<PATH_TO_FILE>/iPrism/DiverseEnv/auto/bin/startsim.sh` to use the `CarlaUE4.sh` in the directory that `CARLA` is installed in
- [OATomobile](https://github.com/OATML/oatomobile)
- Please refer to `requirements.txt` for packages like PyTorch
- Filling in directories for `<PATH_TO_FILE>` at all places
- Setting `PYTHONPATH` as 
    ```
    <PATH_TO_FILE>/iPrism/DiverseEnv/auto/sim/CARLA_0.9.10/PythonAPI:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/sim/CARLA_0.9.10:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/sim/CARLA_0.9.10/PythonAPI/carla:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/sim/CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge/scenario_runner:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge/leaderboard/team_code:<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge:<PATH_TO_FILE>/iPrism/DiverseEnv/carladataset/carla-sim:<PATH_TO_FILE>/iPrism/ReachML/reachml:<PATH_TO_FILE>/iPrism/ReachML/reachml/model:<PATH_TO_FILE>/iPrism/ReachML```
- Place `epoch24.ckpt` (separately uploaded) inside of `<PATH_TO_FILE>/iPrism/DiverseEnv/auto/agents/2020_CARLA_challenge/`


### System Requirements
- AMD Ryzen 7 5700x
- NVIDIA GeForce GTX 1080 Ti
- 64 GB Memory
- Ubuntu 18.04.5 LTS

### Generating STI in Post-Processing
Run `<PATH_TO_FILE>/iPrism/iPrism/DiverseEnv/carladataset/carla-sim/bev_planning/risk_driver_traj_parameter_sweeping.py` with arguments 
```
<folder_name>
dyn
PS
blocking
None
GT
```
The main logic can be found in `,PATH_TO_FILE>/iPrism/DiverseEnv/carladataset/carla-sim/bev_planning/generate_risk_traj_poly_single_timestamp.py`.
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


### Example Data
- `risk_detection`
  - Example data after generating STI in Post-Processing
- `risk_mitigation`
  - `LBC_w_STI`: Adding SMC to mitigate accident using STI as input
  - `LBC_wo_STI`: Adding SMC to mitigate accident using only cameras
  - `LBC_w_TTC`: Adding SMC to mitigate accident using TTC as input
  - `RIP`: Using RIP agent instead of LBC agent to handle OOD scenarios
  - `RIP_w_STI`: Using RIP agent as well as SMC to mitigate accident using STI as input
  - To see vanilla LBC result, please use the `sc_campaign` [here](https://github.com/zihengjackchen/OTA).


# Related Work
- [Learning by Cheating](https://github.com/bradyz/2020_CARLA_challenge)
- [RIP](https://rowanmcallister.github.io/publication/carnovel/)
- [OATomobile](https://github.com/OATML/oatomobile)