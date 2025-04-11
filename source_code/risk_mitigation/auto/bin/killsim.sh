#!/bin/bash
ps aux | grep CarlaUE4 | grep -v "grep" |  awk  '{print $2}' | xargs kill -9
