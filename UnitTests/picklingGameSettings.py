### THIS IS TEMPORARY! CAN DELETE ONCE ZACH UPDATES THE REAL SHIMON_HERO
import pickle
import pygame
import sched
import time

from shimon_hero import shimon_hero as sh

s = sched.scheduler(time.time, time.sleep)
def print_time(last_time): pass # print (time.time() - last_time)

fr = 40. # frame rate
s_per_frame = (1./fr)

gs = dict()
with open('../saved_models/model_03-15_22-24-56_param.p', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    shimon_hero_params = pickle.load(f)

for paramater in shimon_hero_params:
    gs[paramater] = shimon_hero_params[paramater]

game = sh.Game(gs)
print(game.get_settings())

game_over = False
key_action = [0, 0, 0, 0]
last_time = time.time()
while not game_over:
    # type: (object) -> object
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            #print(event.key)
            if event.key == 49:  # 1 key down
                key_action[0] = -1
            elif event.key == 50:  # 2 key down
                key_action[0] = 1
            elif event.key == 113:  # q key down
                key_action[1] = -1
            elif event.key == 119:  # w key down
                key_action[1] = 1
            elif event.key == 97:  # a key down
                key_action[2] = -1
            elif event.key == 115:  # s key down
                key_action[2] = 1
            elif event.key == 122:  # z key down
                key_action[3] = -1
            elif event.key == 120:  # x key down
                key_action[3] = 1
        if event.type == pygame.KEYUP:
            if event.key == 49:  # 1 key down
                key_action[0] = 0
            elif event.key == 50:  # 2 key down
                key_action[0] = 0
            elif event.key == 113:  # q key down
                key_action[1] = 0
            elif event.key == 119:  # w key down
                key_action[1] = 0
            elif event.key == 97:  # a key down
                key_action[2] = 0
            elif event.key == 115:  # s key down
                key_action[2] = 0
            elif event.key == 122:  # z key down
                key_action[3] = 0
            elif event.key == 120:  # x key down
                key_action[3] = 0

    # arm_action = [1, 0, 0, 0] # do nothing
    # if(key_action[0] == 1 and key_action[1] == 0):
    #     arm_action = [0, 1, 0, 0]
    # elif (key_action[0] == 0 and key_action[1] == 1):
    #     arm_action = [0, 0, 1, 0]
    # elif (key_action[0] == 1 and key_action[1] == 1):
    #     arm_action = [0, 0, 0, 1]

    #print(key_action)
    image_data, reward, game_over, game_score = game.next_action(key_action)
    print "reward: ", reward, "game_over: ", game_over, "game_score: ", game_score
    #s.enter(0, 1, game.next_action, ([key_action]))
    #s.enter(s_per_frame, 1, print_time, ([last_time]))
    #s.run()
    #last_time = time.time()
game.exit_game()

# img = smp.toimage(image_data.T)
# img.show()                   v   z# viesw pixel array ofhuman_playing=True