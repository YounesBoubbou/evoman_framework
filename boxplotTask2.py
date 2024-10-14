import os, sys
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

try:
    commandline = sys.argv[4]
except:
    print("usage: 'python boxplot.py EA11 EA12 EA21 EA22'")
    exit()

def read_best_runs(experiment_name):
    # find the best individual per run resulting from run10.py
    NNs = []
    for i in range(10):
        # read weights of best individual
        aux_file = open(experiment_name + str(i) + '/best.txt','r')
        NN = np.array(aux_file.read().splitlines()).astype(float)
        NNs.append(NN)
        aux_file.close()
    return NNs

def test_run(env,NN):
    f,p,e,t = env.play(pcont=NN)
    return p-e

# Determine the individual gain for each best individual among four series of ten runs.
def testing(experiment_names): 
    gains = []
    # for each series of ten runs
    for exp_name in experiment_names:
        gains_exp = []
        NNs = read_best_runs(exp_name)
        # for each best individual among the ten runs
        for NN in NNs:
            gain = test_run(env,NN)
            gains_exp.append(gain)
        gains.append(gains_exp)
    return gains


# standard environment
n_hidden_neurons = 10
env = Environment(enemies=[1,2,3,4,5,6,7,8],
                  multiplemode='yes',
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# determining gains
experiment_names1 = [sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4]]
gains1 = testing(experiment_names1)

# plot gains
plt.boxplot(gains1)
plt.xticks([1,2,3,4], ['EA1 E1','EA2 E1','EA1 E2','EA2 E2'])
plt.title('Mean individual gain of best performing individual')
plt.ylabel('Individual gain')
plt.xlabel('Experiment name')

# statistical test
p1 = round(ttest_ind(gains1[0], gains1[1]).pvalue, 6)
p2 = round(ttest_ind(gains1[2], gains1[3]).pvalue, 6)

# plotting p-values
y, h = max(map(max, gains1)) + 2, 2,
plt.plot([1, 1, 2, 2], [y, y+h, y+h, y], lw=1.5, c='k')
plt.text((1+2)*.5, y+h, p1, ha='center', va='bottom', color='k')
plt.plot([3, 3, 4, 4], [y, y+h, y+h, y], lw=1.5, c='k')
plt.text((3+4)*.5, y+h, p2, ha='center', va='bottom', color='k')

plt.show()

