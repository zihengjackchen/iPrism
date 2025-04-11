#/usr/bin/bash

# MIT License

# Copyright (c) 2022 Shengkun Cui, Saurabh Jha

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


sudo apt update && sudo apt install python3-pip
sudo pip3 install numpy
sudo pip3 install -e /media/sheng/data4/projects/RiskGuard/argodataset/argoverse
echo "" >> /home/ubuntu/.bashrc
echo "export PYTHONPATH=$PYTHONPATH:/media/sheng/data4/projects/RiskGuard/argodataset/argoverse:/media/sheng/data4/projects/RiskGuard/planners/dependencies/rrt_star_planner/RRTStar:/media/sheng/data4/projects/RiskGuard/planners/dependencies/frenet_optimal_trajectory_planner/FrenetOptimalTrajectory:/media/sheng/data4/projects/RiskGuard/planners/dependencies/hybrid_astar_planner/HybridAStar:/media/sheng/data4/projects/RiskGuard/planners/dependencies && export PYLOT_HOME=/media/sheng/data4/projects/RiskGuard/planners" >> /home/ubuntu/.bashrc
source /home/ubuntu/.bashrc
sudo apt install ffmpeg libsm6 libxext6 -y
sudo apt install libqt5gui5 -y
cd /media/sheng/data4/projects/RiskGuard/argodataset/argoverse/bev_planning
