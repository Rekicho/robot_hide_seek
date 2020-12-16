#!/bin/bash

trap 'pkill hider; pkill seeker' INT

ros2 run robot_hide_seek hider &
ros2 run robot_hide_seek seeker &
ros2 run robot_hide_seek game_controller