#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf

NUM_ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1 # This controls how many frames to wait before deciding on an action. If F_P_A = 1, then Shimon
                      # chooses a new action every tick, which causes erratic movements with no exploration middle spaces
SAVE_MODEL = True  # If just troubleshooting, then turn SAVE_MODEL off to avoid cluttering the workspace with logs and models

### DEFINING THE TENSORFLOW MODEL ###

# Create some functions to speed up initializations of each layer type

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)  # Use random weight initialization
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # define network weights

    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600,512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, NUM_ACTIONS])
    b_fc2 = bias_variable([NUM_ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])  # input is state "s", which has dimensions 80x80x4

    # hidden layers
    # Hidden layer 1
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)  # Stride = 4
    h_pool1 = max_pool_2x2(h_conv1)

    # Hidden layer 2
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)  # Stride = 2

    # Hidden layer 3
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)  # Stride = 1
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])  # Need to flatten the convolutional volumes for input into fully connected layer

    # Fully connected layer 4
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # Final layer 5
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2  # The readout layer outputs the value assigned to each action in NUM_ACTIONS

    return s, readout

def trainNetwork(s, readout, sess):
    # define the cost function

    a = tf.placeholder("float", [None, NUM_ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)


