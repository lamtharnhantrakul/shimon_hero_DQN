import pygame
import time
import numpy as np
from copy import copy

FRAME_WIDTH = 3
FRAME_FROM_BORDER = 3

# Hyperparameters
SCREEN_WIDTH = 96 + (2 * (FRAME_WIDTH + FRAME_FROM_BORDER))
MARGIN_FOR_FONT = 36
SCREEN_HEIGHT = int(SCREEN_WIDTH / 2) * 2 + MARGIN_FOR_FONT
ARM_WIDTH = 5
NUMBER_OF_ARMS = 2
ARM_SPEED = 1
NOTE_SPEED = 1
REWARD_CATCHING_NOTE = 1.
PLAYER_DIES_PENALTY = -1000.
PROB_NOTE_SPAWNED = 0.05

ARM_DIRECTION = (np.zeros(NUMBER_OF_ARMS)).tolist()
ARM_STARTS = (np.zeros(NUMBER_OF_ARMS)).tolist()
left_arms = 0
right_arms = 0
for i in range(NUMBER_OF_ARMS): # initialize arm stuff
    if i % 2 == 0: # arm is on left
        arm_start = FRAME_FROM_BORDER + FRAME_WIDTH + (left_arms * ARM_WIDTH)
        arm_direction = 1 # arm will move right by default
        ARM_STARTS[left_arms] = arm_start
        ARM_DIRECTION[left_arms] = arm_direction
        left_arms += 1
    else: # arm is on right
        arm_start = SCREEN_WIDTH - (1 + FRAME_FROM_BORDER + FRAME_WIDTH + ((1 + right_arms) * ARM_WIDTH))
        arm_direction = -1  # arm will move left by default
        ARM_STARTS[NUMBER_OF_ARMS - 1 - right_arms] = arm_start
        ARM_DIRECTION[NUMBER_OF_ARMS - 1 - right_arms] = arm_direction
        right_arms += 1

GAME_TITLE = 'shimon_hero'
TPS = 100


# Loose definition of a colours used in the game
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (219, 218, 191)
NOTE = (255, 215, 64)
GREY = (112, 138, 127)


class Block(pygame.sprite.Sprite):
    def __init__(self, width, height):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([width, height])
        self.rect = self.image.get_rect()


class Arm(Block):
    def __init__(
            self,
            colour=RED,
            width=ARM_WIDTH,
            height=18,
            start=FRAME_FROM_BORDER + FRAME_WIDTH):

        self.start = start
        Block.__init__(self, width, height)
        self.image.fill(colour)
        self.rect.x = self.start
        self.rect.y = SCREEN_HEIGHT - height - 18
        self.score = 0
        self.speed = [+ARM_SPEED, 0]

    def move(self):
        right = SCREEN_WIDTH - 1 - FRAME_FROM_BORDER - FRAME_WIDTH - ARM_WIDTH
        left = FRAME_FROM_BORDER + FRAME_WIDTH
        self.rect.x = max(min((self.rect.x + self.speed[0], right)), left)
        self.rect.y += self.speed[1]

    def update(self):
        self.move()


class Note(Block):
    def __init__(
            self,
            colour=NOTE,
            width=(SCREEN_WIDTH - (2 * (FRAME_FROM_BORDER + FRAME_WIDTH))) / 48,
            height=12,
            speed=NOTE_SPEED):
        Block.__init__(self, width, height)
        self.image.fill(colour)
        offset = ((FRAME_FROM_BORDER + FRAME_WIDTH) / 2)
        mult = (SCREEN_WIDTH - FRAME_FROM_BORDER - FRAME_WIDTH) / 48
        self.rect.x = (int(np.random.randint(0, 48)) * mult) + offset
        self.rect.y = FRAME_FROM_BORDER + FRAME_WIDTH
        self.speed = speed

    def update(self):
        self.rect.y += self.speed


class Game():

    def __init__(self, human_playing=False):
        np.random.seed(int(time.time()))

        self.human_playing = human_playing
        self.start = time.time()

        # Launch PyGame
        pygame.init()
        pygame.display.set_caption(GAME_TITLE)
        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

        # Initialize a few useful variables
        self.font = pygame.font.SysFont("calibri", 20)
        self.reward = 0
        self.is_terminal = False
        self.count = +1
        self.clock = pygame.time.Clock()
        self.score = ""

        self.all_items_list = pygame.sprite.Group()
        self.arm_sprite_list = pygame.sprite.Group()
        self.arm_list = []
        self.note_list = pygame.sprite.Group()
        self.arm_actions = []
        for i in range(NUMBER_OF_ARMS):
            arm = Arm(start=ARM_STARTS[i])
            self.arm_sprite_list.add(arm)
            self.all_items_list.add(arm)
            self.arm_actions.append(1) # 1 means move in default direction
            self.arm_list.append(arm)

    def arm_collision(self, arms):
        for i in range(len(arms)):
            current_arm = arms[i]
            for j in range((i + 1), len(arms)):
                other_arm = arms[j]
                if((other_arm.rect.x - current_arm.rect.x) < ARM_WIDTH):
                    #print(i, other_arm.rect.x, current_arm.rect.x)
                    return True
        return False

    def next_action(self, input_actions):

        if self.is_terminal:
            self.__init__()  # If the player dies, then restart the game. HOWEVER, I need the self.is_terminal=True
            # and self.reward = -ve to be passed on to the q_learning algorithm. So reset the game when next_action
            # is called after the game that was lost.
            # Problem is that self.gave_over = True is not transferred directly to q learner immediatly.

        # type: (object) -> object
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_terminal = True

            # If human is playing (for troubleshooting)
            if self.human_playing:
                if event.type == pygame.KEYDOWN:
                    if event.key == 122:  # z key down
                        self.arm_actions[0] = -1
                    elif event.key == 120:  # x key down
                        self.arm_actions[1] = -1
                    elif event.key == 99:  # c key down
                        self.arm_actions[2] = -1
                    elif event.key == 118:  # v key down
                        self.arm_actions[3] = -1
                if event.type == pygame.KEYUP:
                    if event.key == 122:  # z key up
                        self.arm_actions[0] = 1
                    elif event.key == 120:  # x key up
                        self.arm_actions[1] = 1
                    elif event.key == 99:  # c key up
                        self.arm_actions[2] = 1
                    elif event.key == 118:  # v key up
                        self.arm_actions[3] = 1

        # If Q-learner is playing - For 2 arms there are 4 combos (00 z0 0x zx)
        # input_actions --> a one-hot vector corresponding to action e.g [1,0,0,0] or [0,0,1,0]

        #print input_actions
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        if input_actions[0] == 1:  # do nothing (00) i.e all keys up
            self.arm_actions[0] = 1
            self.arm_actions[1] = 1

        elif input_actions[1] == 1:  # press z key (z0)
            self.arm_actions[0] = -1
            self.arm_actions[1] = 1

        elif input_actions[2] == 1:  # press x key (0x)
            self.arm_actions[0] = 1
            self.arm_actions[1] = -1

        elif input_actions[3] == 1:  # press x and z key (zx)
            self.arm_actions[0] = -1
            self.arm_actions[1] = -1

        # Update all items' positions and the game_count
        for i in range(len(self.arm_list)):
            self.arm_list[i].speed = [ARM_SPEED * self.arm_actions[i] * ARM_DIRECTION[i], 0]
        self.all_items_list.update()
        self.count += 1

        # Create the background, basic frame, and score line
        self.screen.fill(BLACK)
        self.frame = pygame.draw.rect(self.screen, WHITE, pygame.Rect(
            (FRAME_FROM_BORDER, FRAME_FROM_BORDER),
            (SCREEN_WIDTH - 2 * FRAME_FROM_BORDER,
                SCREEN_HEIGHT - 2 * FRAME_FROM_BORDER)),
            FRAME_WIDTH)
        score_total = 0
        for arm in self.arm_list:
            score_total += arm.score

        '''
        self.score = self.font.render(
            '      Current score : ' + str(int(score_total)), True, WHITE)
        self.screen.blit(
            self.score,
            (FRAME_FROM_BORDER + FRAME_WIDTH + 6,
                FRAME_FROM_BORDER + FRAME_WIDTH))
        self.reward = copy(score_total)  # Need to return the overall score
        '''

        # Generate obstacles and noteen coins randomly
        if (np.random.uniform() < PROB_NOTE_SPAWNED):
            note = Note()
            self.note_list.add(note)
            self.all_items_list.add(note)

        note_hits =  []
        for arm in self.arm_list:
            note_hits += pygame.sprite.spritecollide(arm, self.note_list, True)

        # If note was caught by the arm, then reward is distributed
        if note_hits:
            reward = REWARD_CATCHING_NOTE * len(note_hits)
            for arm in self.arm_list:
                arm.score += reward

        self.reward = copy(score_total)

        # If the arm hits an obstacle or the screen's border, it's game over
        # The scores are updated and the game is reset
        did_arms_collide = self.arm_collision(self.arm_list)
        if (did_arms_collide):
            print("ARMS COLLIDED")
            self.last_score = self.arm_list[0].score

            self.all_items_list.empty()
            self.arm_sprite_list.empty()
            self.note_list.empty()
            self.reward += PLAYER_DIES_PENALTY
            self.is_terminal = True



        # Print all objects in the screen
        self.all_items_list.draw(self.screen)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        # self.clock.tick(TPS)



        return image_data, self.reward, self.is_terminal

    def exit_game(self):
        # Save settings, reset the graph, and close the session
        pygame.display.quit()
        pygame.quit()
