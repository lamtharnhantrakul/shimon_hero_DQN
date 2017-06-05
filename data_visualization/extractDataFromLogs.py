import re
from tqdm import tqdm
fname = "./model_03-16_12-54-19_LOG.txt"

for line in tqdm(open(fname)):

    start = line.find('TIME')
    end = line.find("|", start)
    time_stamp = line[start + len('TIME'):end]
    time_stamp.replace(" ", "")
    with open("./timesteps.txt", "a") as text_file:
        print(time_stamp, file=text_file)


    start = line.find('Q_MAX_t1')
    end = line.find("|", start)
    Q_max = line[start + len('Q_MAX_t1') + 3:end]

    with open("./Q_max.txt", "a") as text_file:
        print(Q_max, file=text_file)



    start = line.find('MAX_SCORE')
    end = line.find("|", start)
    max_score = line[start + len('MAX_SCORE'):end]
    #print(max_score)

    with open("./max_score.txt", "a") as text_file:
        print(max_score, file=text_file)

