"""
MIT License

Copyright (c) 2022 Shengkun Cui, Saurabh Jha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

STATIC_FOT_HYPERPARAMETERS = {
    "MAX_SPEED": 100 / 3.6,
    "MAX_ACCEL": 4.0,
    "MAX_CURVATURE": 0.2,
    "MAX_ROAD_WIDTH": 8,  # over estimation
    "D_ROAD_W": 1,
    "N_S_SAMPLE": 2,
    "GOAL_THRESHOLD": 2.0,
    "GOAL_PROXIMITY": 1.0,
    "K_J": 0.01,
    "K_T": 0.1,
    "K_D": 1.0,
    "K_LAT": 1.0,
    "K_LON": 1.0
}