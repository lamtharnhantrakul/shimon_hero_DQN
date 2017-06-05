# LAST EDITED 2-19-2017 9:20 AM

#!/usr/bin/env python
from __future__ import print_function

import os
import random
import time
from collections import deque

import numpy as np
import skimage as skimage
from keras import initializations
from keras.initializations import normal
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from skimage import transform, color, exposure

import shimon_hero as sh

NUM_ACTIONS = 4 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1 # This controls how many frames to wait before deciding on an action. If F_P_A = 1, then Shimon
                      # chooses a new action every tick, which causes erratic movements with no exploration middle spaces



img_rows , img_cols = 80, 80  # All images are downsampled to 80 x 80
img_channels = 4  # Stack 4 frames to infer movement

# Initialize instance of Shimon Hero game
game = sh.Game()
timestr = time.strftime("%m-%d_%H-%M-%S")  # save the current time to name the model

# In the newest version of Keras (February 2017), you need to pass a kwarg called "dim_ordering" when initializing weights
def my_init(shape, name=None, dim_ordering=None):
    return initializations.normal(shape, scale=0.01, name=name)

def buildmodel():
    # Build the model using the same specifications as the DeepMind paper
    # Keras is a good library for protrotyping a quick model and will later
    # be modified/used in conjunction with pure tensorflow code
    # TODO: Need to figure out how to run on GPU

    print("Building CNN Model")
    model = Sequential()

    # 1st Convolutional layer
    model.add(Convolution2D(32, 8, 8,
                            subsample=(4, 4),
                            init=my_init,
                            border_mode='same',
                            input_shape=(img_rows, img_cols, img_channels)))
    model.add(Activation('relu'))

    # 2nd Convolutional layer
    model.add(Convolution2D(64, 4, 4,
                            subsample=(2, 2),
                            init=my_init,
                            border_mode='same'))
    model.add(Activation('relu'))

    # 3rd Convolutional layer
    model.add(Convolution2D(64, 3, 3,
                            subsample=(1, 1),
                            init=my_init,
                            border_mode='same'))
    model.add(Activation('relu'))

    # Flatten the CNN tensors into a long vector to feed into a Dense (Fully Connected Layer)
    model.add(Flatten())

    # Connect the flattened vector to a fully connected layer
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))

    # The number of outputs is equal to the number of valid actions e.g NUM_ACTIONS = 2 (up, down)
    model.add(Dense(NUM_ACTIONS, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    # Use the Adam optimizer for gradient descent
    adam = Adam(lr=1e-6)

    # Compile the model
    model.compile(loss='mse', optimizer=adam)

    print("CNN Model Complete")
    return model

def trainNetwork(model, args):
    a_t = np.zeros(NUM_ACTIONS)  # Preallocate action vector at time t
    action_index = 0

    # DeepMind papers uses what is called a "Replay Memory". Replay Memory stores a collection of states (or frames).
    # When the network is trained, a random sample is taken from Replay Memory. This is to retain i.i.d condition
    # since subsequent frames have highly correlated movement.
    RM = deque()

    # Get the first state by doing nothing. For 2 arms this is [1,0,0,0]
    do_nothing = np.zeros(NUM_ACTIONS)
    do_nothing[0] = 1
    image_data, r_t, game_over = game.next_action(do_nothing)

    # Preprocess image --> Change to greyscale and downscale to 80x80
    x_t = skimage.color.rgb2gray(image_data)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
    assert (x_t.shape == (80,80))
    #plt.matshow(x_t, cmap=plt.cm.gray)
    #plt.show()


    # To infer movement between frames, stack 4 frames together as one "sample" for training
    s_t = np.stack((x_t,x_t,x_t,x_t), axis=0)

    # To feed data into CNN, it must be in the correct tensor format
    # In tensorflow, it must be in the form (1,80,80,4)
    s_t = s_t.reshape(1, s_t.shape[1], s_t.shape[2], s_t.shape[0])
    assert (s_t.shape == (1,80,80,4))

    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON

    t = 0

    # Create a log to store the training variables at each time step
    model_log_dir = "./saved_models/model_" + timestr + "_LOG.txt"
    if not os.path.exists(model_log_dir):
        with open(model_log_dir, "w") as text_file:
            text_file.write("Model Log \n")

    #for i in range(1):
    with open(model_log_dir, 'a') as text_file:
        while (True):
            # Declare a few variables for the Bellman Equation
            loss = 0
            Q_sa = 0

            #print ("time: ", t)
            # To prevent the model from falling into a local minimum or "exploring only one side of the screen",
            # DeepMind chooses a random action from time to time to encourage exploration of other states (a hack)
            if t % FRAME_PER_ACTION == 0:  # This selects an action depending on FRAME_PER_ACTION (default = 1)
                if random.random() <= epsilon:
                    #print("XXXXXX --- { RANDOM ACTION } --- XXXXXXXXX")
                    a_t = np.zeros(NUM_ACTIONS)  # Need to reset a_t, otherwise loop may add 1 to [0001] giving [0101]
                    action_index = random.randrange(NUM_ACTIONS)
                    a_t[action_index] = 1  # set action command with correct one-hot encoding
                else:
                    #print("--- { NEW ACTION } ---")
                    a_t = np.zeros(NUM_ACTIONS)  # Need to reset a_t, otherwise loop may add 1 to [0001] giving [0101]
                    q = model.predict(s_t)  # Inputs the (1,80,80,4) stack of images and outputs a prediction of the best action
                    max_Q = np.argmax(q)  # The index of highest probability in the last Dense layer is the action to take
                    action_index = max_Q.copy()  # The state with max_Q is now the action_index
                    a_t[action_index] = 1  # set action command with correct one-hot encoding

            #print ("action_index: ", action_index)


            # As time progresses, reduce epsilon gradually. This concept is exactly like "Simulated Annealing".
            # Epsilon starts out high (default=0.1) and tends towards FINAL_EPSILON (default=0.0001). So the
            # system has begins with high energy and is likely to jump around to new states and gradually "cools"
            # to a optimal minimum.
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observed next state and r_t
            image_data, r_t, game_over = game.next_action(a_t)

            # preprocess the image
            x_t1 = skimage.color.rgb2gray(image_data)
            x_t1 = skimage.transform.resize(x_t1, (80, 80))
            x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # This is just one image frame (1,80,80,1)
            assert (x_t1.shape == (1,80,80,1))

            # From previous s_t stack, take the top 3 layers and stack these below x_t1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
            assert (s_t1.shape == (1,80,80,4))

            # Store this transition in RM (Replay Memory). The transition contains information on state, action_index, r_t, the next state, and wehther it is game over.
            RM.append((s_t, action_index, r_t, s_t1, game_over))
            if len(RM) > REPLAY_MEMORY:
                RM.popleft()  # Get rid of the oldest item added to deque object

            # Up until this point, the code has just been the playing the game and storing the image stacks.
            # After a certain number of observations defined by OBSERVE (default = 3200), we have enough
            # data points in RM (Replay Memory) to begin drawing batches of training data
            if t > OBSERVE:
                # Sample a minibatch to train on
                mini_batch = random.sample(RM, BATCH)  # From selection RM, choose BATCH=32 number of times

                inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # This should (32,80,80,4)
                assert (inputs.shape == (32,80,80,4))
                targets = np.zeros((BATCH,NUM_ACTIONS))  # This should be (32,4) and stores the r_t for each of the actions
                assert (targets.shape == (32,4))

                # Use random indices to select minibatch from Replay Memory
                for i in range(0,len(mini_batch)):
                    state_t = mini_batch[i][0]  # This is the state
                    action_t = mini_batch[i][1]  # This is the action index
                    reward_t = mini_batch[i][2]  # This is the r_t at state t
                    state_t1 = mini_batch[i][3]  # This is the next state t+1
                    game_over = mini_batch[i][4]  # This is the boolean of whether game is over

                    # Populate input tensor with samples. Remember inputs begins as np.zeros(32,80,80,4). We are now
                    # filling it in line by line as i is incremented over minibatch
                    inputs[i:i+1] = state_t

                    # Use model to predict the output action given state_t. This is like the RHS of Bellman EQN/
                    targets[i] = model.predict(state_t)
                    # Use model to predict the output action given state_t1. This is the Q(s,a) LHS value of Bellman EQN
                    Q_sa = model.predict(state_t1)

                    # if game_over=True, then state equals r_t
                    if game_over:
                        targets[i, action_t] = reward_t
                    # if game is still playing, then action receives a discounted r_t
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                    # What just happened? After the step above, the action_index in targets (32x4) has the highest r_t
                    # We want the weight training process to bias towards the action_index with highest r_t obtained from Bellman EQN
                    # I think the example uses a very hacky method.

                # Compute the cumulative loss
                loss += model.train_on_batch(inputs, targets)

            # Update variables for next pass
            s_t = s_t1
            t = t + 1

            # Save progress every 10,000 iterations
            if t % 1000 == 0:
                print("Saving model so far at timestep: " + str(t))
                model_name = "model_" + timestr + ".h5"
                dir_path = "./saved_models/" + model_name
                model.save_weights(dir_path, overwrite=True)
                #with open("modelB.json", "w") as outfile:
                    #json.dump(model.to_json(), outfile)

            # Print info to track training
            if t <= OBSERVE:
                state = "Observing"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "Exploring"
            else:
                state = "Training"

            """
            print("TIME", t,
                  "/ STATE", state,
                  "/ EPS", '%.10f' % epsilon,
                  "/ ACT", action_index,
                  "/ REW", '%1d' % r_t,
                  "/ Q_MAX", '%2.3f'% np.max(Q_sa),
                  "/ LOSS", '%2.3f' % loss,
                  "/ GAME_O ", game_over)
            """

            data_string = "TIME %8d | STATE %1s | EPS %3.10f | ACT %1d | REW %5d | Q_MAX %3.3f | LOSS %2.3f | GAME_O %s " % (t, state, epsilon, action_index, r_t, np.max(Q_sa), loss, str(game_over))
            text_file.write(data_string + "\n")

        print("Episode finished!")
        print("************************")

        game.exit_game()


def playGame(args):
    model = buildmodel()
    trainNetwork(model, args)


def main():
    # parser = argparse.ArgumentParser(description='Shimon Hero mode')
    # parser.add_argument('-m','--mode', help='Train / Run', required=True)
    # args = vars(parser.parse_args())
    # args = "Train"
    # playGame(args)
    args = "Train"
    playGame(args)


if __name__ == "__main__":
    main()
