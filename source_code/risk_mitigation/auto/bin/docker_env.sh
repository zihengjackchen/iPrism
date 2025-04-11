export CARLA_ROOT=/auto/sim/carla-0.9.10/
export SCENARIO_RUNNER_ROOT=/auto/sim/scenario_runner
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI

#python3 -m pip install -r ${SCENARIO_RUNNER_ROOT}/requirements.txt
#python3 -m pip install -r ${CARLA_ROOT}/PythonAPI/carla/requirements.txt
#python3 -m pip install six
#python3 -m pip uninstall carla
#easy_install --user ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
