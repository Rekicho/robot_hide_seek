Authors:
    André Esteves up201606673@fe.up.pt
    Bruno Sousa up201604145@fe.up.pt
    Francisco Filipe up201604601@fe.up.pt

    Robotics - MIEIC 5th Year - FEUP

Repository:
    https://github.com/Rekicho/robot_hide_seek

Directory Organization:
    /launch: launch files to launch Gazeboo simulation whit TurtleBot3 robots.
             Adapted from: 
             https://github.com/ROBOTIS-GIT/turtlebot3_simulations/tree/foxy-devel/turtlebot3_gazebo/launch

    /models: Adapted TurtleBot3 Burger models to add color according to robot role and
             allow each robot to subscribe and publish to different topics.
             Adapted from: 
             https://github.com/ROBOTIS-GIT/turtlebot3_simulations/tree/foxy-devel/turtlebot3_gazebo/models/turtlebot3_burger

    /resource: ROS2 Package resource.

    /results: Results obtained during Deep Q-Learning training.
              For each episode, the final reward (indicating win/loss)
              along with the average reward are stored.

    /robotic_hide_seek: Developed code.
              Includes:
                deepqlearn: 
                    Deep Q-Learn implementation.
                    Adapted from https://github.com/vmayoral/basic_reinforcement_learning

                deeptrain_hider:
                    Script to train hiders using Deep Q-Learn.
                    Adapted from https://github.com/vmayoral/basic_reinforcement_learning

                deeptrain_seeker:
                    Script to train seekers using Deep Q-Learn.
                    Adapted from https://github.com/vmayoral/basic_reinforcement_learning

                game_controller:
                    Game Controller node.

                gazebo_connection:
                    Script to pause/unpause/reset Gazebo simulation.
                    Adapted from https://bitbucket.org/theconstructcore/drone_training/src/master/

                hider_env:
                    Open AI Gym enviroment for hider training

                hider_train:
                    Hider that stores LIDAR sensors and does not send velocity commands.
                    Used for training.

                hider:
                    Hider node.

                qlearn:
                    Deep Q-Learn implementation.
                    Adapted from https://github.com/vmayoral/basic_reinforcement_learning

                seeker_env:
                    Open AI Gym enviroment for seeker training

                seeker_train:
                    Seeker that stores LIDAR sensors and does not send velocity commands.
                    Used for training.

                seeker:
                    Seeker node.

                train_hider:
                    Script to train hiders using Q-Learn.
                    Adapted from https://github.com/vmayoral/basic_reinforcement_learning

                train_seeker:
                    Script to train seekers using Q-Learn.
                    Adapted from https://github.com/vmayoral/basic_reinforcement_learning

                utils:
                    Constants and utility function both for the game and for training.

    /training_results: Saved state from robots training.
        /hider and /seeker: Deep Q-Learn Neural Network weights
        hiders.txt and seekers.txt: Q-Learn Q table.

    /worlds: Worlds implemented for the Gazebo Simulations.
             Implemented the following (n_hider, n_seeker) configuration:
                (1,1), (2,1), (2,2), (1,2)

    build.sh: Script to build Package

    deeptrain_hider.sh and deeptrain_seeker.sh: Train hider/seeker using Deep Q-Learn.
                                                Assumes run_sim.sh is running.

    kill_all.sh: Kill all nodes

    package.xml: ROS2 package declaration

    run_game.sh: Runs a 2 hider, 2 seeker game

    run_sim.sh: Runs simulation with a given number of hiders and seekers (default: 2 each)

    setup.cfg and setup.py: Setup ROS2 package.

    train_hider and train_seeker: Train hider/seeker using Deep Q-Learn.
                                  Assumes run_sim.sh is running.

Source code used:
    https://github.com/ROBOTIS-GIT/turtlebot3_simulations/tree/foxy-devel/
    https://github.com/vmayoral/basic_reinforcement_learning
    https://bitbucket.org/theconstructcore/drone_training/src/master/

Dependencies:
    Tested with Ubuntu 20.04
    ROS 2 Foxy Fitzroy
    Python 3
    Gazebo

    ROS 2 Foxy Packages:
        rclpy
        turtlebot3
        turtlebot3_msgs
        turtlebot3_simulations
        sensor_msgs
        geometry_msgs
        std_msgs
        rosgraph_msgs
        nav_msgs
        tf
        std_srvs
        

    Python 3 Packages:
        numpy
        keras
        tensorflow - GPU support isn't necessary but advised for faster Deep Q-Learning training (instructions: https://www.tensorflow.org/install/gpu )
        gym
        transformations

How to test:
    Install Python packages using pip

    Install ROS2 Foxy
    Create ~/ros2_ws folder
    Create src folder inside ros2_ws
    Install ROS2 packages
    Copy repository to ~/ros2_ws/src folder

    Add the following commands to ~/.bashrc:
        - source /opt/ros/foxy/setup.bash
        - source ~/ros2_ws/install/setup.bash
        - export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/ros2_ws/src/turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/models:~/ros2_ws/src/robot_hide_seek/models
        - export TURTLEBOT3_MODEL=burger
        - export ROS_DOMAIN_ID=30 #TURTLEBOT3

    Change directory to ~/ros2_ws/src/robot_hide_seek
    Execute:
        $ chmod +x *.sh

    To build the package, run:
        $ ./build.sh

    To run the simulation:
        $ ./run_sim.sh

        Simulation can be run at real-world speed by doing the following in worlds/*.model:
            Comment "<real_time_update_rate>0</real_time_update_rate>"
            Change "<max_step_size>0.01</max_step_size>" to "<max_step_size>0.001</max_step_size>"
            Uncomment "<real_time_factor>1</real_time_factor>"

        Otherwise, simulation will run at ~15x real-world speed (to train faster)

    To run game:
        $ ./run_game.sh

        Constant GAME_USES_TRAINING in utils.py defines whether the robots should use Deep Q-Learn training results (when True) or basic AI (when False)

    To train using Q-Learning:
        Training results are loaded when training starts.
        To train from scrath replace training_results/hiders.txt or training_results/seekers.txt content with '{}'.

        For hider:
            $ ./train_hider.sh

        For seeker:
            $ ./train_seeker.sh

        GAME_USES_TRAINING should be set to False

    To train using Deep Q-Learning:
        Training results are loaded when training starts.
        To train from scrath delete training_results/hider or training_results/seeker folders.

        For hider:
            $ ./deeptrain_hider.sh

        For seeker:
            $ ./deeptrain_seeker.sh

        GAME_USES_TRAINING should be set to False