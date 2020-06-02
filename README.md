# Robotic Grasping using Deep Reinforcement Learning

Reinforcement learning in a simulated environment for the control of Baxter robot manipulator.

`BaxterEnv.lua` interfaces with the Atari DQN to provide a custom environment conforming to the following [API](https://github.com/Kaixhin/rlenvs). Passes a resized 7x60x60 tensor (one RGB image from Baxter built-in camera, one RGB image from an external camera, and motor position information in the 4th channel) from the simulator into the DQN, and in return passes commands back to simulator.

A coloured sphere, cylinder or box is spawned at a random orientation at start and reset. The baxter robot attempts to navigate it's arm to pick up the object. Currently movement on the arm is limited to a rotation at the wrist and shoulder, as well as the ability to extend the reach while forcing the gripper to be facing downwards. 

An attempt to pickup the object results in termination, as unsuccessful attempts often throw the object out of reach. The success of the task is gauged by checking that the pose of the object is approxiamtely the pose of the end-effector at the end of the pickup action. A partial reward is given if the robot comes into contact with the object. At termination the environment is reset.

## Requirements
- ROS indigo
- Gazebo 7
- [baxter simulator installation](http://sdk.rethinkrobotics.com/wiki/Simulator_Installation)
- Torch7 (can use CUDA and cuDNN if available)
- torch-ros
- rgbd_launch

Also requires the following extra luarocks packages:
- luaposix 33.4.0
- luasocket
- moses
- logroll
- classic
- torchx
- rnn
- dpnn
- nninit
- tds

## Installation
Clone [Atari](https://github.com/Kaixhin/Atari), place `BaxterNet.lua` and `BaxterEnv.lua` in Atari folder.

Place `baxter-grasping-drl` package in ros workspace alongside baxter simulator installation.

While in ros workspace, rebuild by running
```
source devel/setup.bash
catkin_make
catkin_make install
```
## Use
Launch baxter_gazebo with 
```
./baxter.sh sim
roslaunch baxter_gazebo baxter_world.launch
```
Once loaded, in a new terminal run
```
source devel/setup.bash
rosrun baxter_grasping_drl robot_control.py
```

Navigate to the Atari directory and run:
`th main.lua -env BaxterEnv -modelBody BaxterNet -bootstraps 0 -PALpha 0 -steps 200000 -hiddenSize 512 -memSize 50000 -epsilonSteps 50000 -tau 10000 -learnStart 10000 -progFreq 1000 -valFreq 5000 -valSteps 2000 -checkpoint true -batchSize 16 -memSampleFreq 2 -gamma 0.90
`

## Citation
If you use this project in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry:
```
@article{joshi2020grasping,
  title={Robotic Grasping using Deep Reinforcement Learning},
  author={Joshi, Shirin and Kumra, Sulabh and Sahin, Ferat},
  journal={IEEE International Conference on Automation Science and Engineering, CASE 2020},
  year={2020}
}
```

## Acknowledgements
Based on baxter_dqn_ros (https://github.com/powertj/baxter_dqn.git)baxter_grasping_drl
