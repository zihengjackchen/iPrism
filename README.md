# iPrism: Characterize and Mitigate Risk by Quantifying Change in Escape Routes

#### Problem Statement
In complex and dynamic real-world situations involving multiple actors, ensuring safety is a significant challenge. This complexity often leads to severe accidents. The current techniques for mitigating safety hazards are not effective because they do not guarantee accessible escape routes and do not specifically address actors contributing to hazards. As a result, these techniques may not provide timely responses. 

To overcome these limitations, we propose a new measure called the safety-threat indicator (STI). This metric helps identify crucial actors by incorporating defensive driving techniques through counterfactual reasoning. We utilize STI to analyze real-world datasets, revealing inherent biases towards safe scenarios. Additionally, we employ it to develop a hazard mitigation policy using reinforcement learning. 

Our approach demonstrates a substantial reduction in the accident rate for advanced autonomous vehicle agents in rare hazardous scenarios—up to a 77% improvement over current state-of-the-art methods. 


#### Contributions
- Enhanced dependability of AVs in unfamiliar and accident-prone scenarios with innovative traffic risk assessment method
- Designed experiments to validate the proposed method using 30000+ trials in CARLA Simulator and real-world dataset
- Created parallelized data generation and testing pipelines and boosted efficiency by 200% using subprocess in Python
- Constructed 6000 unfamiliar scenarios from NHTSA pre-crash typologies and trained lightweight Reinforcement Learning
Agents in PyTorch to preemptively brake using the traffic risk as an indicator, reducing accidents by 72.7%

#### Result
*Related lightning talk accepted by VehicleSec 2024*  
*Paper accepted by DSN 2024*

