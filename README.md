# Deep Q Networks for learned Bi-Manual and n-Manual robot control and coordination
![Alt text](assets/trained_bi_manual.gif?raw=true "Title")

## Overview

Meet Shimon, a 4-armed Marimba-playing improvising robot at the Georgia Tech Center for Music Technology!
The aim of this project is to develop a learned representation of Shimon's physical constraints for use in real-time musical improvisation. We implement an extension of DeepMind's Atari-playing Deep Q Network that approximates a simple Left-Right Motor Cortex of an animal brain. After training, the network learns bi-manual coordination in a virtual environment and out-performs a human at the same task.

![Alt text](assets/Shimon.png?raw=true "Title")

## Bi-manual control and coordination
The modified Deep Q-Network is able to learn 
![Alt text](assets/trained_bi_manual2.gif?raw=true "Title")



## 4-arm control and coordination
The Network is yet unable to generalize and coordinate 4 arms. Some local optimums include:
"Grouping the arms together to form 2 arms"


"Grouping all arms together to form one large arm"



## Q-learning 


## Training

## Architecture

## Dependancies
To install the requirements used in this project, run the following command.
```
pip install -r requirements.txt
```
The main dependancies are:
* Python 3.5
* Keras 2.0.2
* Tensorflow 1.0.1
* Pygame 1.9.3

## Credits
* Deep Reinforcement Learning: Lamtharn (Hanoi) Hantrakul
* Shimon Hero Simulator: Zachary Kondak
* Advisor: Dr. Gil Weinberg
* Special thanks to Dr. Mason Bretan for insightful discussion and advice throughout the project. 
