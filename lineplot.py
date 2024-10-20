# Runs the code of the specialist optimisation ten times (if neeeded) and creates a graph of the mean fitness and best fitness during optimisation.

import os, sys
from matplotlib import pyplot as plt
import numpy as np

# reads and interprets results.
def find_stats(experiment_name):
    file_aux = open(experiment_name + '/results.txt','r')
    # reading text
    text = []
    for line in file_aux:
        words = line.split(" ")
        text.append(words)
    file_aux.close()

    # extracting meaningful data: the generation number, the best and the mean. 
    gen, best, mean, extracting = [], [], [], False
    for i in range(len(text)):
        text[i] = list(filter(None, text[i]))
        if extracting:
            gen.append(int(text[i][0]))
            best.append(float(text[i][1]))
            # need to strip to remove \n from last value in line.
            mean.append(float(text[i][2].strip()))
        elif text[i][0] == 'gen':
            extracting = True         
    return gen, best, mean

# determines mean and standard deviation over all runs
def collate(experiment_name):
    bests, means=[],[]
    for i in range(10):
        gen,best,mean = find_stats(experiment_name + str(i))
        bests.append(np.array(best))
        means.append(np.array(mean))
    meanmean=[np.mean(k) for k in zip(*means)]
    meanstd=[np.std(k) for k in zip(*means)]
    bestmean=[np.mean(k) for k in zip(*bests)]
    beststd=[np.std(k) for k in zip(*bests)]
    return gen, meanmean, meanstd, bestmean, beststd

def add_graph(plt,experiment_name):
    # I don't know what happens if there are a different number of generations between experiment 1 and experiment 2
    gen,meanmean,meanstd,bestmean,beststd = collate(experiment_name)
    plt.plot(gen,meanmean)
    plt.fill_between(gen,[y - error for y, error in zip(meanmean, meanstd)],[y + error for y, error in zip(meanmean, meanstd)],alpha=0.5)
    plt.plot(gen,bestmean)
    plt.fill_between(gen,[y - error for y, error in zip(bestmean, beststd)],[y + error for y, error in zip(bestmean, beststd)],alpha=0.5)

try:
    commandline = sys.argv[2]
except:
    print("usage: 'python lineplot.py series1 series2 enemy_id' a series consists of ten directories. If the directories are called test2[0-9], call it test2")
    exit()
experiment_name1, experiment_name2, enemy_number = sys.argv[1], sys.argv[2], sys.argv[3]

add_graph(plt,experiment_name1)
add_graph(plt,experiment_name2)
plt.ylabel('fitness')
plt.xlabel('generations')
plt.title('fitness curve for enemy '+enemy_number)
plt.legend(['Mean EA1','Best EA1','Mean EA2', 'Best EA2'])
plt.ylim(-10, 100)
plt.grid()
plt.show()
