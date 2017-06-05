### THIS IS TEMPORARY! CAN DELETE ONCE ZACH UPDATES THE REAL SHIMON_HERO

import pygame
import time
import math
import numpy as np
import socket
from copy import copy
from mido import MidiFile
import os
import time
import random


class GameSettings(object):
    def __init__(self):

        # SET DIMENSIONS AND STUFF
        self.SCREEN_WIDTH = 96  # pretty much scales everything else
        self.TOTAL_NUM_NOTES = 24
        self.NOTE_WIDTH = self.SCREEN_WIDTH / self.TOTAL_NUM_NOTES
        self.NOTE_HEIGHT = self.NOTE_WIDTH * 4.

        self.SHIMON_RANGE = 1385.                       # for sending to shimon
        self.WIDTH_PROPORTION = 55. / self.SHIMON_RANGE #
        self.SEND_TO_SHIMON = False

        self.ARM_WIDTH = self.NOTE_WIDTH
        # math.ceil(((-1.0) * self.WIDTH_PROPORTION * self.SCREEN_WIDTH) / (((-1.0) * self.WIDTH_PROPORTION) - 1.0))
        self.ARM_HEIGHT = self.ARM_WIDTH * 4.
        self.SCREEN_HEIGHT = int(self.SCREEN_WIDTH + self.ARM_HEIGHT)

        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT = 5005

        #self.sock = socket.socket(socket.AF_INET,  # Internet
        #socket.SOCK_DGRAM)  # UDP

        # SET ARMS STUFF
        self.NUMBER_OF_ARMS = 2
        self.ARM_SPEED = 2
        #self.ARM_SPEED = 15  # specified in pixels/frame - represents the maximum speed for the arms
        self.ACCELERATION = -1  # if acceleration is < 0 use instant velocity - specified in pixels/frame/frame
        #self.ACCELERATION = 0.1
        # ** with the current setup, ARM_SPEED must be a multiple of ACCELERATION
        self.PIXELS_TO_STOP = 2 * (self.ARM_SPEED / self.ACCELERATION)  # max number of pixels needed to stop
        self.NOTE_SPEED = 1

        # set control mode, can be either list_enumerated [1, 0, 0...], [0, 1, 0]
        # this mode only exists because it conveniently works well with the deep mind stuff.
        # the other mode is direct which takes a list which has the direction of each arm:
        # 1 being right, -1 being left and 0 being stay still
        self.CONTROL_MODE = "direct"

        # REWARD STUFF
        self.REWARD_CATCHING_NOTE = 1.
        self.PENALTY_MISSING_NOTE = -1.
        self.PLAYER_DIES_PENALTY = 0.
        self.ARM_COLLISION_PENALTY = 0. # punishment for arms colliding
        self.AMOUNT_TO_MISS = self.ARM_HEIGHT  # this is how many pixels a note can go past the arms before it is considered missed
        self.POINT_THRESHOLD = -10
        self.MOVE_PENALTY = -0.1

        # GAME SETTINGS
        self.COLLISION_NOTE_RESET = False  # if true, arms collideing causes death, else arms colliding causes arms to stay still
        self.PROB_NOTE_SPAWNED = 0.02
        self.DISPLAY_SCORE = False
        self.SOUND_WHEN_GENERATED = False  # if true, notes will play sound when generated as well as hit

        # DIRECTORY SETTINGS
        self.USE_MIDI = False
        self.MIDI_FILES_DIRECTORY = './midi'  # if empty, defaul is random

        self.ARM_STARTS = [0] * self.NUMBER_OF_ARMS  # Creates a list [0,0]

        # Start the arms in random positions
        self.ARM_STARTS[0] = random.uniform(0,0.5) * self.SCREEN_WIDTH
        self.ARM_STARTS[1] = random.uniform(0.6,1) * self.SCREEN_WIDTH

        self.GAME_TITLE = 'shimon_hero_hanoi'
        self.TPS = 100

        # Loose definition of a colours used in the game
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (219, 218, 191)
        self.DARK_GREY = (50, 50, 50)
        self.NOTE = (255, 215, 64)
        self.GREY = (255, 255, 255)


gs = GameSettings()  # global singleton containing all game settings


class Block(pygame.sprite.Sprite):
    def __init__(self, width, height):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([width, height])
        self.rect = self.image.get_rect()


class Arm(Block):
    def __init__(
            self,
            index,
            arm_list,
            colour=gs.GREY,
            width=gs.ARM_WIDTH,
            start=0.0):

        self.start = start
        self.mult = 1.0
        self.position = start
        Block.__init__(self, width, gs.ARM_HEIGHT)
        self.image.fill(colour)
        self.arm_list = arm_list
        self.index = index
        self.rect.x = self.position
        self.rect.y = gs.SCREEN_HEIGHT - gs.ARM_HEIGHT
        self.score = 0
        self.speed = [+gs.ARM_SPEED, 0]  # how fast the arm can go - should really be called max speed
        self.current_speed = 0.0  # represents how fast the arm is going (only used when working with acceleration)
        self.last_speed = 0.0
        self.direction = 0.0  # direction of intended movement (only used when working with acceleration)

    def move(self):
        if gs.ACCELERATION <= 0.0: # simple case: instant velocity
            right = gs.SCREEN_WIDTH - gs.ARM_WIDTH
            left = 0.0
            self.position = max(min((self.position + self.speed[0], right)), left)
            self.rect.x = self.position
            self.rect.y += self.speed[1]
        else:  # use acceleration

            wall = [1,1]
            if self.index == 0 : # leftmost arm
                wall[0] = 0.5
                left = 0.0
                if self.index == gs.NUMBER_OF_ARMS - 1: # only one arm
                    right = gs.SCREEN_WIDTH - gs.ARM_WIDTH
                else: # more than 1 arm
                    right = self.arm_list[self.index + 1].position - gs.ARM_WIDTH
            elif self.index == gs.NUMBER_OF_ARMS - 1: # rightmost arm
                wall[1] = 0.5
                left = self.arm_list[self.index - 1].position + gs.ARM_WIDTH
                right = gs.SCREEN_WIDTH - gs.ARM_WIDTH
            else:
                left = self.arm_list[self.index - 1].position + gs.ARM_WIDTH
                right = self.arm_list[self.index + 1].position - gs.ARM_WIDTH

            output = False
            if((self.position - left) <= (gs.PIXELS_TO_STOP * wall[0])) and self.direction < 0.0:
                self.mult = math.sqrt(max(0.0, self.position - left) / (gs.PIXELS_TO_STOP * wall[0]))
                output = True
            elif((right - self.position) <= (gs.PIXELS_TO_STOP * wall[1])) and self.direction > 0.0:
                self.mult = math.sqrt(max(0.0, right - self.position) / (gs.PIXELS_TO_STOP * wall[1]))
                output = True
            else:
                self.mult = 1.0

            if self.direction != self.speed[0]: # arm changed direction
                self.current_speed = self.last_speed
                output = True
            if self.current_speed == 0.0:
                output = True # we want to output here because this is when we can actually update the position of arms

            if self.current_speed > self.speed[0]:
                self.current_speed = max(self.speed[0], self.current_speed - gs.ACCELERATION)
            elif self.current_speed < self.speed[0]:
                self.current_speed = min(self.speed[0], self.current_speed + gs.ACCELERATION)
            self.position = min(right, max(left, self.position + (self.mult * self.current_speed)))
            self.last_speed = (self.mult * self.current_speed)
            self.rect.x = self.position
            self.rect.y += self.speed[1]
            if output: # arm changed direction
                pass
                # print("this is where we output stuff to shimon")
            self.direction = self.speed[0]

    def update(self):
        self.move()


class Penalty(Block):
    def __init__(self, color = gs.DARK_GREY):
        height = gs.ARM_HEIGHT
        Block.__init__(self, gs.SCREEN_WIDTH, height )
        self.image.fill(color)
        self.rect.x = 0
        self.rect.y = gs.SCREEN_HEIGHT - gs.ARM_HEIGHT + gs.AMOUNT_TO_MISS

    def update(self):
        pass


class Note(Block):
    def __init__(
            self,
            colour=gs.NOTE,
            speed=gs.NOTE_SPEED,
            note=None):
        Block.__init__(self, gs.NOTE_WIDTH, gs.NOTE_HEIGHT)
        self.image.fill(colour)
        offset = 0
        mult = (gs.SCREEN_WIDTH) / gs.TOTAL_NUM_NOTES

        if note != None:
            middle = gs.TOTAL_NUM_NOTES / 2
            if gs.TOTAL_NUM_NOTES % 2 == 1:
                middle = gs.TOTAL_NUM_NOTES / 2 + 1

            self.note = (note - 60) + middle
            while self.note >= gs.TOTAL_NUM_NOTES:
                self.note -= 12
            while self.note < 0:
                self.note += 12
        else:
            self.note = int(np.random.randint(0, gs.TOTAL_NUM_NOTES))

        self.sound_file = str(self.note) + '.wav'
        self.rect.x = (self.note * mult) + offset
        self.rect.y = 0
        self.speed = speed

    def update(self):
        self.rect.y += self.speed

    def note_missed(self):
        return self.rect.y >= (gs.SCREEN_HEIGHT - gs.ARM_HEIGHT - gs.NOTE_HEIGHT - gs.AMOUNT_TO_MISS)

class Game(object):
    def __init__(self, *initial_data, **kwargs):

        np.random.seed(int(time.time())) # random seed

        for dictionary in initial_data: # allows for parameters to be loaded in as a dictionary
            for key in dictionary:
                setattr(gs, key, dictionary[key])
        for key in kwargs:
            setattr(gs, key, kwargs[key])

        # Every time the game is started, the __init__ method of class is called.
        # So we reset the arms, but have to change the singelton gs attribute.
        gs.ARM_STARTS[0] = random.uniform(0, 0.5) * gs.SCREEN_WIDTH
        gs.ARM_STARTS[1] = random.uniform(0.6, 1) * gs.SCREEN_WIDTH

        self.start = time.time()

        # Launch PyGame
        pygame.init()
        pygame.mixer.init()
        pygame.display.set_caption(gs.GAME_TITLE)
        self.screen = pygame.display.set_mode([gs.SCREEN_WIDTH, gs.SCREEN_HEIGHT])

        # Initialize a few useful variables
        self.font = pygame.font.SysFont("calibri", 20)
        self.reward = 0
        self.is_terminal = False
        self.count = +1
        self.clock = pygame.time.Clock()
        self.score = ""
        self.note_count = 0
        self.step_count = 0
        self.midi_notes = []

        # For loading MIDI
        if gs.USE_MIDI:
            if os.path.isdir(gs.MIDI_FILES_DIRECTORY):
                for file in os.listdir(gs.MIDI_FILES_DIRECTORY):
                    if file.endswith('.midi') or file.endswith('.mid'):
                        print ("reading midi file: ", file)
                        midiFile = MidiFile(gs.MIDI_FILES_DIRECTORY + '/' + str(file))
                        for i, track in enumerate(midiFile.tracks):
                            for message in track:
                                if message.type == "note_on":
                                    #print("note: " + str(message.note) + " time: " + str(message.time))
                                    self.midi_notes.append((message.note, message.time/gs.NOTE_SPEED))

        self.penalty_zone = Penalty()
        self.penalty_list = pygame.sprite.Group()
        self.penalty_list.add(self.penalty_zone)
        self.all_items_list = pygame.sprite.Group()
        self.arm_sprite_list = pygame.sprite.Group()
        self.arm_list = []
        self.note_list = pygame.sprite.Group()
        self.arm_actions = []
        for i in range(gs.NUMBER_OF_ARMS):
            arm = Arm(i, self.arm_list, start=gs.ARM_STARTS[i])
            self.arm_sprite_list.add(arm)
            self.all_items_list.add(arm)
            self.arm_actions.append(1) # 1 means move in default direction
            self.arm_list.append(arm)
        self.last_time = time.time()

    def get_settings(self):
        return gs.__dict__

    def get_collisions(self, arms, directions):
        collisions = (np.zeros(gs.NUMBER_OF_ARMS)).tolist()
        for i in range(len(arms)):
            current_arm = arms[i]
            for j in range((i + 1), len(arms)):
                other_arm = arms[j]
                other_next_x = other_arm.rect.x + (gs.ARM_SPEED * directions[j])
                current_next_x = current_arm.rect.x + (gs.ARM_SPEED * directions[i])
                if((other_next_x - current_next_x) < gs.ARM_WIDTH):
                    #print(i, other_arm.rect.x, current_arm.rect.x)
                    collisions[i] = 1
                    collisions[j] = 1
        return collisions

    def arm_collision(self, arms):
        for i in range(len(arms)):
            current_arm = arms[i]
            for j in range((i + 1), len(arms)):
                other_arm = arms[j]
                if((other_arm.rect.x - current_arm.rect.x) < gs.ARM_WIDTH):
                    #print(i, other_arm.rect.x, current_arm.rect.x)
                    return True
        return False

    def next_action(self, input_actions):

        ### ALL RELATED TO SENDING MESSAGES TO SHIMON
        #   To improve readability, Hanoi has simply commented it out to reduce confusion
        '''
        if (gs.SEND_TO_SHIMON):
            this_time = time.time()
            if self.last_time != 0:
                arms_x = ''
                dt = this_time - self.last_time
                for i in range(len(self.arm_list)):
                    arms_x += ' ' + str(self.arm_list[i].rect.x / float(gs.SCREEN_WIDTH - gs.ARM_WIDTH))
                speed = (gs.ARM_SPEED/float(gs.SCREEN_WIDTH - gs.ARM_WIDTH)) * float(gs.SHIMON_RANGE) / dt
                aG = 9.80665
                accel = (((gs.ACCELERATION/float(gs.SCREEN_WIDTH - gs.ARM_WIDTH))*float(gs.SHIMON_RANGE)/dt)*aG)/1000.0
                arms_x = arms_x + ' ' + str(accel) + ' ' + str(speed)
                gs.sock.sendto(arms_x, (gs.UDP_IP, gs.UDP_PORT))
            self.last_time = this_time
        '''

        ### The next time next_action() is called after a crash, the game resets.
        if self.is_terminal:
            self.__init__()  # If the player dies, then restart the game. HOWEVER, I need the self.is_terminal=True
            # and self.reward = -ve to be passed on to the q_learning algorithm. So reset the game when next_action
            # is called after the game that was lost.
            # Problem is that self.gave_over = True is not transferred directly to q learner immediatly.


        ### We are using "direct" control mode of shimon arms in all cases
        for i in range(len(self.arm_actions)):
            self.arm_actions[i] = input_actions[i]

        ### This piece of code adds a penalty for colliding arms, in the current version of the DQN this is not needed
        #   However a future version could easily penalize an arm if it decides to move around too much
        '''
        for i in range(len(self.arm_actions)):
            if i != gs.NUMBER_OF_ARMS - 1: # not right arm
                this_arm_x = self.arm_list[i].rect.x
                right_arm_x = self.arm_list[i+1].rect.x
                space = right_arm_x - this_arm_x
                if space <= gs.ARM_WIDTH: #arms are touching
                    this_arm_dir = self.arm_actions[i]
                    right_arm_dir = self.arm_actions[i + 1]

                    # If arms touch, then game over
                    #self.reward = -1
                    #if (gs.RESET):
                        #self.is_terminal = True

                    if this_arm_dir != right_arm_dir: # arms moving in different directions

                        if this_arm_dir == 1: # this arm collision
                            self.arm_list[i+1].score += gs.ARM_COLLISION_PENALTY #punish other arm

                            #self.reward = -1  # If Shimon misses a note, give negative reward
                            #if (gs.RESET):
                                #self.is_terminal = True  # If arms touch, then game over

                        if right_arm_dir == -1: # right arm collision
                            self.arm_list[i].score += gs.ARM_COLLISION_PENALTY
        '''

        # if collision is not set to terminal, then don't allow arms to move if they are collided
        if gs.ACCELERATION <= 0.0 and not gs.COLLISION_NOTE_RESET:
            collisions = self.get_collisions(self.arm_list, self.arm_actions)
            collision = False
            total_movement = 0
            collision_start = 0
            for i in range(len(self.arm_actions)):
                if(collision):
                    collision = collisions[i] == 1
                    if collision: # continued collision
                        total_movement += self.arm_actions[i]
                    else: # no more collision
                        dir = 0
                        if total_movement > 0:
                            dir = 1
                        elif total_movement < 0:
                            dir = -1
                        move = dir * gs.ARM_SPEED
                        if(self.arm_list[collision_start].rect.x + move<= 0):
                            dir = max(0, dir)
                        print(self.arm_list[collision_start].rect.x + move)
                        for j in range(collision_start, i):
                            self.arm_actions[j] = dir
                else:
                    collision = collisions[i] == 1
                    if collision: # new collision
                        collision_start = i
                        total_movement = self.arm_actions[i]
            if collision:
                dir = 0
                if total_movement > 0:
                    dir = 1
                elif total_movement < 0:
                    dir = -1
                move = dir * gs.ARM_SPEED
                if (self.arm_list[len(self.arm_list) - 1].rect.x + move >= gs.SCREEN_WIDTH - gs.ARM_WIDTH - 1):
                    dir = min(0, dir)
                    # print(self.arm_list[len(self.arm_list) - 1].rect.x + move)
                move = dir * gs.ARM_SPEED
                if (self.arm_list[collision_start].rect.x + move <= 0):
                    dir = max(0, dir)
                for j in range(collision_start, len(self.arm_actions)):
                    self.arm_actions[j] = dir

        # Update all items' positions and the game_count
        for i in range(len(self.arm_list)):
            self.arm_list[i].speed = [gs.ARM_SPEED * self.arm_actions[i], 0]
        self.all_items_list.update()
        self.count += 1

        # Create the background, basic frame, and score line
        self.screen.fill(gs.BLACK)
        score_total = 0
        for arm in self.arm_list:
            score_total += arm.score

        # Display actual score in the game environment, normally turned off as it interferes with the CNN
        if gs.DISPLAY_SCORE:
            self.score = self.font.render(
                str(int(score_total)), True, gs.WHITE)
            self.screen.blit(
                self.score,
                (6, 0))

        # Generate notes based on MIDI file input or randomly
        if self.note_count < len(self.midi_notes) - 1:
            while self.midi_notes[self.note_count][1] == self.step_count:
                note = Note(note = self.midi_notes[self.note_count][0])
                if gs.SOUND_WHEN_GENERATED:
                    sound = pygame.mixer.Sound('piano_notes/' + str(note.sound_file))
                    sound.play()
                self.note_list.add(note)
                self.all_items_list.add(note)
                self.note_count += 1
                self.step_count = 0
            self.step_count += 1
            if self.note_count >= len(self.midi_notes) - 1:
                print("Now generating random notes")
        else:
            if (np.random.uniform() < gs.PROB_NOTE_SPAWNED):
                note = Note()
                if gs.SOUND_WHEN_GENERATED:
                    sound = pygame.mixer.Sound('piano_notes/' + str(note.sound_file))
                    sound.play()
                self.note_list.add(note)
                self.all_items_list.add(note)

        ### Reward for every frame is 0 by default
        self.reward = 0

        ### If the agent moves, it incurs a small cost, if both arms move, then it incurs (-0.1) + (-0.1) cost
        for move in input_actions:
            if move != 0:
                self.reward += gs.MOVE_PENALTY

        ### Detect for correctly hit notes
        for arm in self.arm_list:
            note_hit = pygame.sprite.spritecollide(arm, self.note_list, True)
            # Play corresponding note sounds when note is hit
            for note in note_hit:  # Play audio for confirmation
                sound = pygame.mixer.Sound('piano_notes/' + str(note.sound_file))
                sound.play()
            if note_hit:
                arm.score += len(note_hit) * gs.REWARD_CATCHING_NOTE
                self.reward = 1  # reward at a frame is 1 when a note is hit

        ### Detect missed notes, Zach implemented this by defining a collision between a penalty zone
        #   and a falling note
        penalty_hits = pygame.sprite.spritecollide(self.penalty_zone, self.note_list, True)
        if penalty_hits:
            arm.score += gs.PENALTY_MISSING_NOTE
            self.reward = -1  # If Shimon misses a note, give negative reward
            if gs.COLLISION_NOTE_RESET:  # If Shimon misses a note, restart the game
                self.is_terminal = True

        # If the arm hits an obstacle or the screen's border, it's game over
        # The scores are updated and the game is reset
        if gs.COLLISION_NOTE_RESET:
            did_arms_collide = self.arm_collision(self.arm_list)
            if did_arms_collide:
                #print("ARMS COLLIDED")
                self.last_score = self.arm_list[0].score
                self.all_items_list.empty()
                self.arm_sprite_list.empty()
                self.note_list.empty()
                #self.reward += gs.PLAYER_DIES_PENALTY
                self.reward = -1  #
                self.is_terminal = True

        # Print all objects in the screen
        self.penalty_list.draw(self.screen)
        self.all_items_list.draw(self.screen)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        # self.clock.tick(gs.TPS)

        return image_data, self.reward, self.is_terminal, score_total

    def exit_game(self):
        # Save settings, reset the graph, and close the session
        pygame.display.quit()
        pygame.quit()
