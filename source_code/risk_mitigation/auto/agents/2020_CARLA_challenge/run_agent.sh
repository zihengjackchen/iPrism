#!/bin/bash
# source /auto/bin/docker_env.sh
# agent environment
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:./leaderboard
export PYTHONPATH=$PYTHONPATH:./leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:./scenario_runner
export PYTHONPATH=${PYTHONPATH}:./


# agent settings
export PORT=2000                                                    # change to port that CARLA is running on
export ROUTES=leaderboard/data/routes_training/route_42.xml  # change to desired route
export TEAM_AGENT=image_agent.py                                    # no need to change
export TEAM_CONFIG=epoch24.ckpt                                      # change path to checkpoint
export HAS_DISPLAY=0


if [ -d "$TEAM_CONFIG" ]; then
    CHECKPOINT_ENDPOINT="$TEAM_CONFIG/$(basename $ROUTES .xml).txt"
else
    CHECKPOINT_ENDPOINT="$(dirname $TEAM_CONFIG)/$(basename $ROUTES .xml).txt"
fi

echo python3 leaderboard/leaderboard/leaderboard_evaluator.py \
--track=SENSORS \
--scenarios=leaderboard/data/fi_front_accidient.json \
--agent=image_agent.py \
--agent-config=epoch24.ckpt \
--routes=$ROUTES \
--port=2000 \
--trafficManagerSeed=0 \
--dual_agent

#--enable_fi=True

#leaderboard/data/routes_fi/route_highway.xml \
echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."
