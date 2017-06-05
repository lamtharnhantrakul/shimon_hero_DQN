from __future__ import print_function

import numpy as np
import os
import pickle
import random
import skimage as skimage
import time
from collections import deque
from keras.initializers import RandomNormal
from keras.layers import Input
from keras.layers import LSTM, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.models import save_model
from keras.optimizers import Adam
from keras.utils import plot_model
from skimage import transform, color, exposure

from shimon_hero import shimon_hero_DQNLSTM as sh

SAVE_MODEL = False  # If just troubleshooting, then turn SAVE_MODEL off to avoid cluttering the workspace with logs and models
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
model_prefix = "2A1NNshared_" + timestr  # The prefix used to identify the model and time training was created

# Save the shimon_hero paramters corresponding to the model
shimon_hero_param = game.get_settings()


time_steps = 2  # We will allow the LSTM to look 2 steps in the past
num_notes = 36  # The note will be a one hot vector

if SAVE_MODEL:
    param_path = "./saved_models/" + model_prefix + "_param.p"
    with open(param_path, 'wb') as f:
        pickle.dump(shimon_hero_param, f, pickle.HIGHEST_PROTOCOL)


def buildmodel():
    # We are building a hybrid DQN and LSTM model. The main input is the image of the game. The auxilary input
    # are the one hot encodings of previous notes


    ### DEFINE MAIN INPUT IMAGE NETWORK
    ##
    #
    input_img = Input(shape=(img_rows, img_cols, img_channels),
                      name="main_input")

    # 1st Convolutional layer
    layer1_out = Conv2D(filters=32,
                        kernel_size=(8, 8),
                        strides=(4, 4),
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                        padding='same',
                        activation='relu')(input_img)
    #layer1_out = Dropout(0.5)(layer1_out)

    # 2nd Convolutional layer
    layer2_out = Conv2D(filters=64,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                        padding='same',
                        activation='relu')(layer1_out)
    #layer2_out = Dropout(0.5)(layer2_out)

    # 3rd Convolutional layer
    layer3_out = Conv2D(filters=64,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                        padding='same',
                        activation='relu')(layer2_out)
    #layer3_out = Dropout(0.5)(layer3_out)

    # Flatten the CNN tensors into a long vector to feed into a Dense (Fully Connected Layer)
    # Up to this point, we have performed feature extraction on the game image. Now we need to split
    # it up to left and right arm instructions via FC layers
    CNN_out = Flatten()(layer3_out)

    FC1_out = Dense(512,
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                    name='PostCNN_FC',
                    activation='relu')(CNN_out)

    ### DEFINE AUXILARY INPUT SEQUENCE
    ##

    note_input = Input(shape=(time_steps, num_notes), name='note_input')
    lstm_1_out = LSTM(128, return_sequences=True, name='lstm_hidden1')(note_input)
    lstm_1_out = Dropout(0.2)(lstm_1_out)
    lstm_1_hidden = TimeDistributed(Dense(64, activation='tanh'))(lstm_1_out)  # This will be concatenated with the fully connected layers after CNN
    lstm_1_hidden = Flatten()(lstm_1_hidden)

    lstm_note_out = LSTM(32)(lstm_1_out)
    lstm_note_out = Dropout(0.2)(lstm_note_out)
    lstm_note_output = Dense(num_notes, activation='softmax', name='lstm_note_output')(lstm_note_out)  # This one predicts the note class using softmax (Supervised signal)

    ### MERGE CNN and LSTM layers together
    ##
    CNN_LSTM_merge = concatenate([FC1_out, lstm_1_hidden])

    # LEFT ARM
    L_FC_out = Dense(256,
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                     name='L_FC',
                     activation='relu')(CNN_LSTM_merge)

    L_output = Dense(NUM_ACTIONS,
                     name='L_output',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(L_FC_out)

    # RIGHT ARM
    R_FC_out = Dense(256,
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                     name='R_FC',
                     activation='relu')(CNN_LSTM_merge)

    R_output = Dense(NUM_ACTIONS,
                     name='R_output',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(R_FC_out)

    # Use the Adam optimizer for gradient descent
    adam = Adam(lr=1.5e-6)

    model = Model(inputs=[input_img, note_input], outputs=[L_output, R_output, lstm_note_output])
    model.compile(optimizer=adam,
                  loss={'L_output': 'mse', 'R_output': 'mse', 'lstm_note_output': 'categorical_crossentropy'},
                  loss_weights={'L_output': 1.0, 'R_output': 1.0, 'lstm_note_output': 1.0})

    # Print a summary of the model
    print(model.summary())
    plot_model(model, to_file='DQN_LSTM_network.png')
    return model

def note2onehot(note_lookback):
    note_lookback_vec = np.zeros(num_notes,time_steps)
    for i, note in enumerate(note_lookback):
        note_lookback_vec[i, note] = 1
    return note_lookback_vec

def trainNetwork(model):
    # DeepMind papers uses what is called a "Replay Memory". Replay Memory stores a collection of states (or frames).
    # When the network is trained, a random sample is taken from Replay Memory. This is to retain i.i.d condition
    # since subsequent frames have highly correlated movement. Use deque() object from python library.
    RM = deque()

    # Get the first frame by doing nothing, for two arms this is [0, 0]
    image_data, r_t, game_over, game_score, note_vector = game.next_action([0, 0])

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
    note_lookback = [4, 2]
    next_note = 0
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
                note_lookback_vec = note2onehot(note_lookback)
                q_predictions = model.predict([s_t, note_lookback_vec])  # for 2 arms this looks like [array([[ 0.0046034 , -0.00290494, -0.00516033]], dtype=float32), array([[ 0.00205019, -0.01548742, -0.00640914]], dtype=float32)]
                #print (q_predictions[0].shape)
                L_act_idx = np.argmax(q_predictions[0])  # L_action_index = L_max_Q = np.argmax(q_predictions[0].flatten())
                R_act_idx = np.argmax(q_predictions[1])  # The index of highest predicuted Q value in the last Dense layer is the action to take

        # As time progresses, reduce epsilon gradually. This concept is exactly like "Simulated Annealing".
        # Epsilon starts out high (default=0.1) and tends towards FINAL_EPSILON (default=0.0001). So the
        # system has begins with high energy and is likely to jump around to new states and gradually "cools"
        # to a optimal minimum.
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and r_t


        image_data, r_t, game_over, game_score, next_note = game.next_action([action_dict[L_act_idx], action_dict[R_act_idx]])

        # preprocess the image
        x_t1 = skimage.color.rgb2gray(image_data)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # This is just one image frame (1,80,80,1)

        # From previous s_t stack, take the top 3 layers and stack these below x_t1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # Store this transition in RM (Replay Memory). The transition contains information on state, action_index, r_t, the next state, and wehther it is game over.
        RM.append((s_t, [L_act_idx, R_act_idx], r_t, s_t1, game_over, note_lookback, next_note))
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
            L_targets = np.zeros((BATCH, NUM_ACTIONS))  # This should be (32,3) and stores the r_t for each of the actions for L arm
            R_targets = np.zeros((BATCH, NUM_ACTIONS))  # This should be (32,3) and stores the r_t for each of the actions for L arm
            print (note_inputs.shape)

            # Use random indices to select minibatch from Replay Memory
            for i in range(0, len(mini_batch)):
                state_t = mini_batch[i][0]  # This is the state
                L_action_t = mini_batch[i][1][0]  # This is L_act_idx
                R_action_t = mini_batch[i][1][1]  # This is R_act_idx
                reward_t = mini_batch[i][2]  # This is the r_t at state t
                state_t1 = mini_batch[i][3]  # This is the next state t+1
                game_over_t = mini_batch[i][4]  # This is the boolean of whether game is over
                note_lookback_t = mini_batch[i][5]  # This is the note lookback
                next_note_t = mini_batch[i][6]  # This is the next note

                # Populate input tensor with samples. Remember inputs begins as np.zeros(32,80,80,4). We are now
                # filling it in line by line as i is incremented over minibatch
                inputs[i:i + 1] = state_t

                # Use model to predict the output action given state_t. This is like the LHS of Bellman EQN
                predictions_t = model.predict([state_t, note2onehot(note_lookback_t)])
                L_targets[i] = predictions_t[0]
                R_targets[i] = predictions_t[1]

                # Use model to predict the output action given state_t1. This is the Q(s',a') in the RHS of Bellman EQN
                predictions_t1 = model.predict(state_t1)
                L_Q_t1 = predictions_t1[0]  # This is a 1x3 vector corresponding to probs of each possible action
                R_Q_t1 = predictions_t1[1]  # D the same for the right arm

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
            # Keras behaviour: when you have multiple outputs, this is what the function model.train_on_batch returns
            # ['loss', 'L_output_loss', 'R_output_loss']
            # [1.5195604e-07, 8.6563546e-08, 6.5392506e-08]
            # Where "loss" is just the average of the L_output_loss and R_output_loss
            loss = model.train_on_batch({'main_input': inputs, 'aux_input': aux_input}, {'L_output': L_targets, 'R_output': R_targets})[0]  # Select the first element, which is the total loss from L and R arms

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
            model_path = "./saved_models/" + model_prefix + ".h5"
            save_model(model, model_path, overwrite=True)  # saves weights, network topology and optimizer state (if any)

    game.exit_game()


def main():
    model = buildmodel()
    trainNetwork(model)


if __name__ == "__main__":
    main()
