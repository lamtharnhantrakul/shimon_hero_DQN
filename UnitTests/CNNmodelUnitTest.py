from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

#GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
NUM_ACTIONS = 4 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1


img_rows , img_cols = 80, 80  # All images are downsampled to 80 x 80
img_channels = 4  # Stack 4 frames to infer movement

# In the newest version of Keras (February 2017), you need to pass a kwarg called "dim_ordering"
def my_init(shape, name=None, dim_ordering=None):
    return initializations.normal(shape, scale=0.01, name=name)

# Build the model using the same specifications as the DeepMind paper
print ("Building CNN Model")
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