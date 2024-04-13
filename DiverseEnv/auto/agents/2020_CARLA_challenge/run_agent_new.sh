#!/bin/bash

# agent settings
export PORT=2000                                                    # change to port that CARLA is running on
export ROUTES=leaderboard/data/routes_testing/route_00.xml  # change to desired route
export TEAM_AGENT=image_agent.py                                    # no need to change
export TEAM_CONFIG=epoch24.ckpt                                      # change path to checkpoint
export HAS_DISPLAY=0


python3 leaderboard/leaderboard/leaderboard_evaluator.py \
--track=SENSORS \
--scenarios=leaderboard/data/fi_front_accidient.json \
--agent=image_agent.py \
--agent-config=epoch24.ckpt \
--routes=leaderboard/data/routes_fi/route_highway.xml \
--port=2000 \
--trafficManagerSeed=0 \


