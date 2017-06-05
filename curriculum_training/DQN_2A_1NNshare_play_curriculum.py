# From 22/3/2017, I migrated the code to using conventions of tensorflow 1.0 and Keras 2.0

from __future__ import print_function

import numpy as np
import os
import pickle
import skimage as skimage
from collections import deque
from keras.models import load_model
from skimage import transform, color, exposure

from shimon_hero import shimon_hero_curriculum as sh

NUM_ACTIONS = 3  # number of valid actions
EXPLORE = 3000000.  # frames over which to anneal epsilon0
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
FRAME_PER_ACTION = 3  # This controls how many frames to wait before deciding on an action. If F_P_A = 1, then Shimon
# chooses a new action every tick, which causes erratic movements with no exploration middle spaces

USE_CURRICULUM = True
CURRICULUM_DIRECTORY = './midi/curriculum2'
levels = []
levelsRandParams = [(30, False, (56, 64), 0.02, []), (30, False, (56, 64), 0.02, [2,4,6,8,10]), (30, False, (56, 64), 0.02, [3,6,9,12,15,18]), (30, False, (52, 68), 0.02, [5,10,15]),
                    (30, False, (52, 68), 0.02, [5,10,15]), (30, False, (52, 68), 0.02, [5,6,11,12,17,18])]

if USE_CURRICULUM:
    if os.path.isdir(CURRICULUM_DIRECTORY):
        levels = [name for name in os.listdir(CURRICULUM_DIRECTORY)
         if os.path.isdir(os.path.join(CURRICULUM_DIRECTORY, name))]
print(levels)

action_dict = {0: -1, 1: 0, 2: 1}  # Shimon hero will interpret [-1] as one arm left, [1] as one arm right. Or [-1 -1] as two arm left and left.
# So QNN to controls is [1 0 0] = [-1], [0 1 0] = [0], [0 0 1] = [1]

def loadModelPrefix():
    ### Query user for the desired model
    dir = './saved_models'

    files = [fname for fname in os.listdir(dir)
             if fname.endswith('.h5')]

    print("Hello, I found %s models in the directory '%s':" % (str(len(files)), dir))
    for i, file in enumerate(files):
        print("[%d]" % i, file)
    model_idx = input("Select index [i] of desired model: ")
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
    RM = deque()  # Here you add an image, then pop it immediatly when playing

    image_data, r_t, game_over, game_score = game.next_action([0, 0])

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

    t = 0
    L_act_idx = 0
    R_act_idx = 0
    max_score = 0
    for index, level in enumerate(levels):
        game = sh.Game({'USE_MIDI': True, 'MIDI_FILES_DIRECTORY': os.path.join(CURRICULUM_DIRECTORY, level),
                        'THRESHOLD_TIME': levelsRandParams[index][0],
                        'RANDOM_NOTES_IN_BETWEEN': levelsRandParams[index][1],
                        'RANDOM_NOTES_RANGE': levelsRandParams[index][2],
                        'RANDOM_NOTES_PROB': levelsRandParams[index][3],
                        'RANDOM_NOTES_NUMS': levelsRandParams[index][4]})
        tot_notes = game.tot_notes
        print(game.get_settings())
        max_score = 0
        while(max_score < game.tot_notes):
            if t % FRAME_PER_ACTION == 0:  # This selects an action depending on FRAME_PER_ACTION (default = 1)
                q_predictions = model.predict(
                    s_t)  # for 2 arms this looks like [array([[ 0.0046034 , -0.00290494, -0.00516033]], dtype=float32), array([[ 0.00205019, -0.01548742, -0.00640914]], dtype=float32)]
                # print (q_predictions[0].shape)
                L_act_idx = np.argmax(q_predictions[0])  # L_action_index = L_max_Q = np.argmax(q_predictions[0].flatten())
                R_act_idx = np.argmax(q_predictions[1])  # The index of highest predicuted Q value in the last Dense layer is the action to take

            # run the selected action and observed next state and r_t
            image_data, r_t, game_over, game_score = game.next_action([action_dict[L_act_idx], action_dict[R_act_idx]])
            tot_notes = game.tot_notes

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

            if game_score > max_score:
                max_score = game_score

            # Update variables for next pass
            s_t = s_t1
            t = t + 1

            debugging_string = "TIME %8d | L_ACT %3d | R_ACT %3d | MAX_SCORE %3d | LEVEL %2d | TOT_NOTES %3d" % (t, action_dict[L_act_idx],action_dict[R_act_idx],max_score,index+1,tot_notes)
            print(debugging_string)

    game.exit_game()

def main():
    model_dir, param_dir = loadModelPrefix()
    model = loadModel(model_dir)  # load model weights
    #game = loadGame(param_dir)  # load corresponding settings in shimon when model was trained
    game = sh.Game()
    playGame(model, game)

if __name__ == "__main__":
    main()
