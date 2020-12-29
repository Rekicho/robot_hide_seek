#!/bin/sh

if [ "$#" -eq 0 ]; then
    echo "Running default world: 2x2"
    export HIDE_SEEK_WORLD="hide_seek_2x2.model"
elif [ "$#" -eq 1 ] || [ "$#" -ne 2 ]; then
    echo "Invalid number of arguments.\nUsage: $0 <hiders> <seekers>"
    exit 1
elif [ "$1" -eq 1 ] && [ "$2" -eq 2 ]; then
    echo "Running world: 1x3"
    export HIDE_SEEK_WORLD="hide_seek_1x2.model"
elif [ "$1" -eq 2 ] && [ "$2" -eq 2 ]; then
    echo "Running world: 2x2"
    export HIDE_SEEK_WORLD="hide_seek_2x2.model"
elif [ "$1" -eq 2 ] && [ "$2" -eq 1 ]; then
    echo "Running world: 2x1"
    export HIDE_SEEK_WORLD="hide_seek_2x1.model"
else
    echo "Invalid number of hiders & seekers.\nUsage: $0 <hiders> <seekers>"
    echo "List:\n $0 2 2 (default)"
    echo " $0 1 2"
    echo " $0 2 1"
fi

ros2 launch robot_hide_seek hide_seek.launch.py