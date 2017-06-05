import os



if not os.path.exists('./saved_models/model_log.txt'):
    with open('../saved_models/model_log.txt', "w") as text_file:
        text_file.write("Model Log \n")

state = "observing"
epsilon = 0.12391283
action_index = 3
r_t = 2
q_max = 3.54
loss = 0.3829
game_over = False

with open('../saved_models/model_log.txt', 'a') as text_file:
    for t in range(100):
        data_string = "TIME %8d | STATE %1s | EPS %3.10f" % (t, state, epsilon)
        text_file.write(data_string + "\n")