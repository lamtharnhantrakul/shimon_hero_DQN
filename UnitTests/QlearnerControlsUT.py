from shimon_hero import shimon_hero as sh

# Initialize instance of Shimon Hero game
game = sh.Game(human_playing=False)
NUM_ACTIONS  = 4
FRAME_PER_ACTION = 1
ACTION_HOLD = 6
import numpy as np
import random
a_t = np.zeros(NUM_ACTIONS)
a_t[0] = 1

t = 0
while True:

    '''
    if t % ACTION_HOLD == 0:  # We need to give a chance for the agent to "hold" the z key over more than 1 frame
        print "--- NEW ACTION ---"
        a_t = np.zeros(NUM_ACTIONS)
        action_index = random.randrange(NUM_ACTIONS)
        a_t[action_index] = 1  # set action command with correct one-hot encoding
    '''

    if t % FRAME_PER_ACTION == 0:  # We need to give a chance for the agent to "hold" the z key over more than 1 frame
        #print "--- NEW ACTION ---"
        a_t = np.zeros(NUM_ACTIONS)
        action_index = random.randrange(NUM_ACTIONS)
        a_t[action_index] = 1  # set action command with correct one-hot encoding


    #print a_t
    image_data, r_t, game_over = game.next_action(a_t)
    print ("r_t: ", r_t)
    t += 1