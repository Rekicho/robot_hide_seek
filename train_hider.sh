#!/bin/bash

# trap './kill_all.sh' INT

# ros2 run robot_hide_seek hider --ros-args -p id:=1 &
# ros2 run robot_hide_seek seeker --ros-args -p id:=0 &
# ros2 run robot_hide_seek seeker --ros-args -p id:=1 &
# ros2 run robot_hide_seek game_controller --ros-args -p n_hiders:=2 -p n_seekers:=2 &
ros2 run robot_hide_seek train_hider