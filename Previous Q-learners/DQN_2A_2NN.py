# From 22/3/2017, I migrated the code to using conventions of tensorflow 1.0 and Keras 2.0

# This model is 2 arms controlled by 1 NN. The NN has 9 possible controls

from __future__ import print_function
import time
import os
import skimage as skimage
from skimage import transform, color, exposure
import random
import numpy as np
from collections import deque
import pickle
from keras.models import save_model
from keras.initializers import RandomNormal
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import shimon_hero as sh

SAVE_MODEL = True  # If just troubleshooting, then turn SAVE_MODEL off to avoid cluttering the workspace with logs and models
DEBUG = False  # Set to true for verbose printing

NUM_ACTIONS = 3  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 3200.  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 3  # This controls how many frames to wait before deciding on an action. If F_P_A = 1, then Shimon
# chooses a new action every tick, which causes erratic movements with no exploration middle spaces

# 1NN controlling 2 arms means you need to enumerate all the controls
action_dict = {0: -1, 1: 0, 2: 1}

img_rows, img_cols = 80, 80  # All images are downsampled to 80 x 80
img_channels = 4  # Stack 4 frames to infer movement

# Initialize instance of Shimon Hero game
game = sh.Game()  # Instantiate Shimon Hero game
timestr = time.strftime("%m-%d_%H-%M-%S")  # save the current time to name the model
model_prefix = "2A1NN_" + timestr  # The prefix used to identify the model and time training was created

# Save the shimon_hero paramters corresponding to the model
shimon_hero_param = game.get_settings()
if SAVE_MODEL:
    param_path = "./saved_models/" + model_prefix + "_param.p"
    with open(param_path, 'wb') as f:
        pickle.dump(shimon_hero_param, f, pickle.HIGHEST_PROTOCOL)


def buildmodel():
    # To make a model with shared weights, need to use Keras Functional API, not Sequential API

    input_img = Input(shape=(img_rows, img_cols, img_channels))

    # 1st Convolutional layer
    layer1_out = Conv2D(filters=32,
                        kernel_size=(8, 8),
                        strides=(4, 4),
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                        padding='same',
                        activation='relu')(input_img)

    # 2nd Convolutional layer
    layer2_out = Conv2D(filters=64,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                        padding='same',
                        activation='relu')(layer1_out)

    # 3rd Convolutional layer
    layer3_out = Conv2D(filters=64,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                        padding='same',
                        activation='relu')(layer2_out)

    # Flatten the CNN tensors into a long vector to feed into a Dense (Fully Connected Layer)
    # Up to this point, we have performed feature extraction on the game image. Now we need to split
    # it up to left and right arm instructions via FC layers
    CNN_out = Flatten()(layer3_out)

    FC1_out = Dense(512,
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    name='PostCNN_FC',
                    activation='relu')(CNN_out)

    # LEFT ARM
    L_FC_out = Dense(256,
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                      name='L_FC',
                      activation='relu')(FC1_out)

    L_output = Dense(NUM_ACTIONS,
                     name='L_output',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(L_FC_out)

    # RIGHT ARM
    R_FC_out = Dense(256,
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                      name='R_FC',
                      activation='relu')(FC1_out)

    R_output = Dense(NUM_ACTIONS,
                     name='R_output',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(R_FC_out)

    # Use the Adam optimizer for gradient descent
    adam = Adam(lr=1e-6)

    # Compile the model
    # model.compile(loss='mse', optimizer=adam)

    L_model = Model(inputs=input_img, outputs=L_output)
    L_model.compile(optimizer=adam,
                    loss='mse')

    R_model = Model(inputs=input_img, outputs=R_output)
    R_model.compile(optimizer=adam,
                    loss='mse')

    # Print a summary of the model
    print(L_model.summary())
    print(R_model.summary())
    return L_model, R_model


def trainNetwork(models):
    L_model, R_model = models

    # DeepMind papers uses what is called a "Replay Memory". Replay Memory stores a collection of states (or frames).
    # When the network is trained, a random sample is taken from Replay Memory. This is to retain i.i.d condition
    # since subsequent frames have highly correlated movement. Use deque() object from python library.
    RM = deque()

    # Get the first frame by doing nothing, for two arms this is [0, 0]
    image_data, r_t, game_over, game_score = game.next_action([0, 0])

    # Preprocess image --> Change to greyscale and downscale to 80x80
    x_t = skimage.color.rgb2gray(image_data)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
    # plt.matshow(x_t, cmap=plt.cm.gray)
    # plt.show()

    # To infer movement between frames, stack 4 frames together as one "sample" for training
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    # To feed data into CNN, it must be in the correct tensor format
    # In tensorflow, it must be in the form (1,80,80,4)
    s_t = s_t.reshape(1, s_t.shape[1], s_t.shape[2], s_t.shape[0])

    # Variables for parameter annealing
    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON

    # Create a log to store the training variables at each time step
    model_log_dir = "./saved_models/" + model_prefix + "_LOG.txt"
    if SAVE_MODEL:
        if not os.path.exists(model_log_dir):
            with open(model_log_dir, "w") as text_file:
                text_file.write("Model Log " + timestr + "\n")

    t = 0
    L_act_idx = 0
    R_act_idx = 0
    max_score = 0
    while (True):
        # Declare a few variables for the Bellman Equation
        loss = 0
        L_Q_t1 = 0
        R_Q_t1 = 0

        # To prevent the model from falling into a local minimum or "exploring only one side of the screen",
        # DeepMind chooses a random action from time to time to encourage exploration of other states (a hack)
        if t % FRAME_PER_ACTION == 0:  # This selects an action depending on FRAME_PER_ACTION (default = 1)
            if random.random() <= epsilon:
                L_act_idx = random.randrange(NUM_ACTIONS)
                R_act_idx = random.randrange(NUM_ACTIONS)
            else:
                L_Q = L_model.predict(s_t)  # Inputs the (1,80,80,4) stack of images and outputs a prediction of the best action
                L_act_idx = np.argmax(L_Q)  # The index of highest predicuted Q value in the last Dense layer is the action to take

                R_Q = L_model.predict(s_t)  # Inputs the (1,80,80,4) stack of images and outputs a prediction of the best action
                R_act_idx = np.argmax(R_Q)  # The index of highest predicuted Q value in the last Dense layer is the action to take

        # As time progresses, reduce epsilon gradually. This concept is exactly like "Simulated Annealing".
        # Epsilon starts out high (default=0.1) and tends towards FINAL_EPSILON (default=0.0001). So the
        # system has begins with high energy and is likely to jump around to new states and gradually "cools"
        # to a optimal minimum.
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and r_t


        image_data, r_t, game_over, game_score = game.next_action([action_dict[L_act_idx], action_dict[R_act_idx]])

        # preprocess the image
        x_t1 = skimage.color.rgb2gray(image_data)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # This is just one image frame (1,80,80,1)

        # From previous s_t stack, take the top 3 layers and stack these below x_t1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # Store this transition in RM (Replay Memory). The transition contains information on state, action_index, r_t, the next state, and wehther it is game over.
        RM.append((s_t, [L_act_idx, R_act_idx], r_t, s_t1, game_over))
        if len(RM) > REPLAY_MEMORY:
            RM.popleft()  # Get rid of the oldest item added to deque object

        # Up until this point, the code has just been the playing the game and storing the image stacks.
        # After a certain number of observations defined by OBSERVE (default = 3200), we have enough
        # data points in RM (Replay Memory) to begin drawing batches of training data
        if t > OBSERVE:
            # Sample a minibatch to train on
            mini_batch = random.sample(RM, BATCH)  # From selection RM, choose BATCH=32 number of times

            # Neural Networks are universal function approximators. Given an input stack of 4 images, we want the network
            # to output the correct the Q(s,a) for each possible given a state (the state is the stack of 4 images.)
            # This is like the y=f(x) problem where network is trying to learn the best f so that x maps onto y.
            # The "targets" are the outputs, which correspond to the Q(s,a) value for each possible action a.
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # This should (32,80,80,4)
            L_targets = np.zeros((BATCH, NUM_ACTIONS))  # This should be (32,3) and stores the r_t for each of the actions
            R_targets = np.zeros((BATCH, NUM_ACTIONS))

            # Use random indices to select minibatch from Replay Memory
            for i in range(0, len(mini_batch)):
                state_t = mini_batch[i][0]  # This is the state
                L_action_t = mini_batch[i][1][0]  # This is L_act_idx
                R_action_t = mini_batch[i][1][1]  # This is R_act_idx
                reward_t = mini_batch[i][2]  # This is the r_t at state t
                state_t1 = mini_batch[i][3]  # This is the next state t+1
                game_over_t = mini_batch[i][4]  # This is the boolean of whether game is over

                # Populate input tensor with samples. Remember inputs begins as np.zeros(32,80,80,4). We are now
                # filling it in line by line as i is incremented over minibatch
                inputs[i:i + 1] = state_t

                # Use model to predict the output action given state_t. This is like the LHS of Bellman EQN
                L_targets[i] = L_model.predict(state_t)
                R_targets[i] = R_model.predict(state_t)

                # Use model to predict the output action given state_t1. This is the Q(s',a') in the RHS of Bellman EQN
                L_Q_t1 = L_model.predict(state_t1)  # This is a 1x4 vector corresponding to probs of each possible action
                R_Q_t1 = R_model.predict(state_t1)  # D the same for the right arm

                # if game_over_t=True, then state equals reward_t
                if game_over_t:
                    L_targets[i, L_action_t] = reward_t
                    R_targets[i, R_action_t] = reward_t
                # if game is still playing, then action receives a discounted reward_t
                else:
                    L_targets[i, L_action_t] = reward_t + (GAMMA * np.max(L_Q_t1))
                    R_targets[i, R_action_t] = reward_t + (GAMMA * np.max(R_Q_t1))
                    # What just happened? After the step above, the action_index in targets (32x4) has the highest r_t
                    # We want the weight training process to bias towards the action_index with highest r_t obtained from Bellman EQN
                    # I think the example uses a very hacky method.

            # Compute the cumulative loss
            loss += L_model.train_on_batch(inputs, L_targets)
            loss += R_model.train_on_batch(inputs, R_targets)

        # Update variables for next pass
        s_t = s_t1
        t = t + 1

        # Keep track of the highest score and write to file if max_score changes
        if game_score > max_score:
            max_score = game_score
            logging_string = "TIME %8d | MAX_SCORE %3d" % (t, max_score)
            print(logging_string)
            if SAVE_MODEL:
                # If saving model, then only update the text file with information on the time step and the score
                with open(model_log_dir, 'a') as text_file:
                    text_file.write(logging_string + "\n")

        if DEBUG:
            # Track current annealed state
            if t <= OBSERVE:
                state = "Observing"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "Exploring"
            else:
                state = "Training"
            debugging_string = "TIME %8d | STATE %1s | EPS %3.10f | L_ACT %1d | R_ACT %1d | REW %5.1f | L_Q_MAX_t1 %8.4f | R_Q_MAX_t1 %8.4f | LOSS %8.4f | GAME_O %s | MAX_SCORE %3d" % (
                t, state, epsilon, L_act_idx, R_act_idx, r_t, np.max(L_Q_t1), np.max(R_Q_t1), loss, str(game_over), max_score)
            print(debugging_string)

        if SAVE_MODEL and t % 10000 == 0:
            # Save progress every 10,000 iterations
            print("Saving model at timestep: " + str(t))
            L_model_path = "./saved_models/" + "L_" + model_prefix + ".h5"
            R_model_path = "./saved_models/" + "R_" + model_prefix + ".h5"
            save_model(L_model, L_model_path, overwrite=True)  # saves weights, network topology and optimizer state (if any)
            save_model(R_model, R_model_path, overwrite=True)  # saves weights, network topology and optimizer state (if any)

    game.exit_game()


def main():
    models = buildmodel()
    trainNetwork(models)


if __name__ == "__main__":
    main()
