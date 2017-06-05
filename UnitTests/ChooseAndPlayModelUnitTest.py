import os
import sys

model_dir = '../saved_models'
'''
files = [os.path.join(model_dir, fname)
        for fname in os.listdir(model_dir)
        if fname.endswith('.h5')]
'''
files = [fname for fname in os.listdir(model_dir)
        if fname.endswith('.h5')]

print "Hello, I found %s models in the directory '%s':" % (str(len(files)), model_dir)
for i, file in enumerate(files):
    print i, file

model_idx = raw_input("Select index [i] of desired model: ")

model = files[int(model_idx)]

print "Model %s selected" % (str(model))