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
    print("usage: 'python boxplotTask2.py EA11 EA21 EA12 EA22'")
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
    return p, e

# Determine the individual gain for each best individual among four series of ten runs.
def testing(experiment_names): 
    gains, best_gain, best_NN = [], -100, 0
    # for each series of ten runs
    for exp_name in experiment_names:
        gains_exp = []
        NNs = read_best_runs(exp_name)

        # for each best individual among the ten runs check the gain
        for NN in NNs:
            p,e = test_run(env,NN)
            gain = p-e
            if gain > best_gain:
                best_gain = gain
                best_NN = NN
            gains_exp.append(gain)
        gains.append(gains_exp)
    return gains, best_NN


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
gains1, NN = testing(experiment_names1)

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

# making table of gains for best NN
scores = [['player_life','enemy_life']]
env.update_parameter('multiplemode','no')
for i in range(1,9):
    env.update_parameter('enemies',[i])
    p,e = test_run(env,NN)
    scores.append([str(p),str(e)])

if not os.path.exists('boxplot'):
    os.makedirs('boxplot')
np.savetxt('boxplot/best.txt',NN)
aux_file = open('boxplot/table.txt','w')
for score_enemy in scores:
    text = str('\n '+score_enemy[0])+', '+str(score_enemy[1])
    print(score_enemy)
    aux_file.write(text)
aux_file.close()