# From 22/3/2017, I migrated the code to using conventions of tensorflow 1.0 and Keras 2.0

from __future__ import print_function
import os
import skimage as skimage
from skimage import transform, color, exposure
import numpy as np
from collections import deque
import pickle
from keras.models import load_model
import shimon_hero as sh

NUM_ACTIONS = 3  # number of valid actions
EXPLORE = 3000000.  # frames over which to anneal epsilon0
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
FRAME_PER_ACTION = 3  # This controls how many frames to wait before deciding on an action. If F_P_A = 1, then Shimon
# chooses a new action every tick, which causes erratic movements with no exploration middle spaces

action_dict = {0: -1, 1: 0, 2: 1}  # Shimon hero will interpret [-1] as one arm left, [1] as one arm right. Or [-1 -1] as two arm left and left.
# So QNN to controls is [1 0 0] = [-1], [0 1 0] = [0], [0 0 1] = [1]

def loadModelPrefix():
    ### Query user for the desired model
    dir = '../saved_models'

    files = [fname for fname in os.listdir(dir)
             if fname.endswith('.h5')]

    print("Hello, I found %s models in the directory '%s':" % (str(len(files)), dir))
    for i, file in enumerate(files):
        print("[%d]" % i, file)
    model_idx = raw_input("Select index [i] of desired model: ")
    model_name = files[int(model_idx)]
    model_dir = dir + "/" + model_name

    model_prefix = model_name.split(".")[0]
    game_param_dir = dir + "/" + model_prefix + "_param.p"
    return model_dir, game_param_dir

def loadModel(model_dir):
    print("Loading weights from: " + model_dir)
    model = load_model(model_dir)
    print("Model loaded successfully")
    return model

def loadGame(param_dir):
    gs = dict()
    with open(param_dir, 'rb') as f:
        shimon_hero_params = pickle.load(f)
    for paramater in shimon_hero_params:
        gs[paramater] = shimon_hero_params[paramater]
    game = sh.Game(gs)
    return game

def playGame(model, game):
    a_t = np.zeros(NUM_ACTIONS)  # Preallocate action vector at time t
    action_index = 0

    RM = deque()

    image_data, r_t, game_over, game_score = game.next_action([0])

    # Preprocess image --> Change to greyscale and downscale to 80x80
    x_t = skimage.color.rgb2gray(image_data)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    # To infer movement between frames, stack 4 frames together as one "sample" for training
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    # To feed data into CNN, it must be in the correct tensor format
    # In tensorflow, it must be in the form (1,80,80,4)
    s_t = s_t.reshape(1, s_t.shape[1], s_t.shape[2], s_t.shape[0])

    # Variables for parameter annealing
    OBSERVE = 999999999  # We keep observe, never train
    epsilon = FINAL_EPSILON

    # for i in range(1):
    t = 0
    action_index = 0
    max_score = 0
    while (True):
        # Declare a few variables for the Bellman Equation
        loss = 0
        Q_sa_t1 = 0

        if t % FRAME_PER_ACTION == 0:  # This selects an action depending on FRAME_PER_ACTION (default = 1)
            # print("--- { NEW ACTION } ---")
            q = model.predict(s_t)  # Inputs the (1,80,80,4) stack of images and outputs a prediction of the best action
            max_Q = np.argmax(q)  # The index of highest probability in the last Dense layer is the action to take
            action_index = max_Q.copy()  # The state with max_Q is now the action_index
            # a_t[action_index] = 1  # set action command with correct one-hot encoding

        # run the selected action and observed next state and r_t
        image_data, r_t, game_over, game_score = game.next_action([action_dict[action_index]])

        # preprocess the image
        x_t1 = skimage.color.rgb2gray(image_data)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # This is just one image frame (1,80,80,1)

        # From previous s_t stack, take the top 3 layers and stack these below x_t1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # Store this transition in RM (Replay Memory). The transition contains information on state, action_index, r_t, the next state, and wehther it is game over.
        RM.append((s_t, action_index, r_t, s_t1, game_over))
        if len(RM) > REPLAY_MEMORY:
            RM.popleft()  # Get rid of the oldest item added to deque object

        # Update variables for next pass
        s_t = s_t1
        t = t + 1

        debugging_string = "TIME %8d | ACT %3d | MAX_SCORE %3d" % (t, action_dict[action_index],max_score)
        print(debugging_string)

    game.exit_game()

def main():
    model_dir, param_dir = loadModelPrefix()
    model = loadModel(model_dir)
    game = loadGame(param_dir)
    playGame(model, game)

if __name__ == "__main__":
    main()
