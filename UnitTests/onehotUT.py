import numpy as np

num_notes = 36
time_steps = 2

def note2onehot(note_lookback):
    note_lookback_vec = np.zeros((num_notes,time_steps))
    for i, note in enumerate(note_lookback):
        note_lookback_vec[note, i] = 1
    print note_lookback_vec.shape
    return note_lookback_vec

print (note2onehot([2,4]))