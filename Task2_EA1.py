###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Pieter Goosens        			                                  #
#     				                                  #
###############################################################################

# imports framework
import sys
from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import os

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = sys.argv[1]

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

if int(sys.argv[2]) == 1:
    enemies_list = [2,5,6,7,8]
elif int(sys.argv[2]) == 2:
    enemies_list = [6,7,8] 


# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  multiplemode="yes",
                  enemies=enemies_list, 
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Genetic Algorithm    ###

ini = time.time()  # sets time marker

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

dom_u = 1
dom_l = -1
npop = 100
gens = 30
p_mutation = 0.39
alpha = 0.51

#### Functions for the fitness
# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))

# limits
def limits(x):
    if x > dom_u:
        return dom_u
    elif x < dom_l:
        return dom_l
    else:
        return x

## Function for mutation
def mutation(offspring, gen):
    sigma_gen = 1 - 0.9 * (gen / gens)
    for i in range(0, len(offspring)):
        if np.random.uniform(0, 1) <= p_mutation:
            offspring[i] = offspring[i] + np.random.normal(0, sigma_gen)
    offspring = np.array(list(map(lambda y: limits(y), offspring)))
    return offspring

# Blend crossover for generating offspring
def blend_crossover(p1, p2, gen):
    o1, o2 = [], []
    for i in range(0, len(p1)):
        d_i = np.abs(p1[i] - p2[i])
        lower_bound = min(p1[i], p2[i]) - alpha * d_i
        upper_bound = max(p1[i], p2[i]) + alpha * d_i
        o1_gene_i = np.random.uniform(lower_bound, upper_bound)
        o2_gene_i = np.random.uniform(lower_bound, upper_bound)
        o1.append(o1_gene_i)
        o2.append(o2_gene_i)

    # Directly apply mutation to offspring
    o1 = mutation(o1, gen)
    o2 = mutation(o2, gen)
    return o1, o2

# Deterministic crowding for survivor selection
def deter_crowding(p1, p2, o1, o2, fit_p1, fit_p2, fit_o1, fit_o2):
    d1_o1 = np.linalg.norm(np.array(p1) - np.array(o1))
    d1_o2 = np.linalg.norm(np.array(p1) - np.array(o2))
    d2_o1 = np.linalg.norm(np.array(p2) - np.array(o1))
    d2_o2 = np.linalg.norm(np.array(p2) - np.array(o2))
    if d1_o1 + d2_o2 < d1_o2 + d2_o1:
        # Parent 1 competes with Offspring 1, Parent 2 competes with Offspring 2
        if fit_o1 > fit_p1:
            p1 = o1  # Replace Parent 1 with Offspring 1
            fit_p1 = fit_o1
        if fit_o2 > fit_p2:
            p2 = o2  # Replace Parent 2 with Offspring 2
            fit_p2 = fit_o2
    else:
        # Parent 1 competes with Offspring 2, Parent 2 competes with Offspring 1
        if fit_o2 > fit_p1:
            p1 = o2  # Replace Parent 1 with Offspring 2
            fit_p1 = fit_o2
        if fit_o1 > fit_p2:
            p2 = o1  # Replace Parent 2 with Offspring 1
            fit_p2 = fit_o1

    return p1, p2, fit_p1, fit_p2

## Recombination with deterministic crowding
def recombination(pop, fit_pop, gen):
    total_offspring = np.zeros((0, n_vars))
    total_fit = []
    shuffled_indices = np.random.choice(npop, size=npop, replace=False)

    for i in range(0, npop, 2):
        p1 = pop[shuffled_indices[i]]
        fit_p1 = fit_pop[shuffled_indices[i]]

        p2 = pop[shuffled_indices[i+1]]
        fit_p2 = fit_pop[shuffled_indices[i+1]]

        o1, o2 = blend_crossover(p1, p2, gen)  # Create two offspring using blend crossover and mutation
        fit_o1 = simulation(env, np.array(o1))
        fit_o2 = simulation(env, np.array(o2))

        # Apply deterministic crowding to select survivors
        new_parent1, new_parent2, fit_new_parent1, fit_new_parent2 = deter_crowding(p1, p2, o1, o2, fit_p1, fit_p2, fit_o1, fit_o2)

        total_offspring = np.vstack((total_offspring, new_parent1))
        total_offspring = np.vstack((total_offspring, new_parent2))

        total_fit.append(fit_new_parent1)
        total_fit.append(fit_new_parent2)

    return total_offspring, total_fit

# Initialize population
pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
fit_pop = evaluate(pop)
best = np.argmax(fit_pop)
mean = np.mean(fit_pop)
std = np.std(fit_pop)
ini_g = 0
solutions = [pop, fit_pop]
env.update_solutions(solutions)

# Save results for first generation
with open(f"{experiment_name}/results.txt", 'a') as file_aux:
    file_aux.write('\n\ngen best mean std')
    print(f'\n GENERATION {ini_g} {round(fit_pop[best], 6)} {round(mean, 6)} {round(std, 6)}')
    file_aux.write(f'\n{ini_g} {round(fit_pop[best], 6)} {round(mean, 6)} {round(std, 6)}')

# Evolution
for i in range(1, gens):
    pop, fit_pop = recombination(pop, fit_pop, i)  # recombination
    
    best = np.argmax(fit_pop)
    std = np.std(fit_pop)
    mean = np.mean(fit_pop)

    # Save results
    with open(f"{experiment_name}/results.txt", 'a') as file_aux:
        print(f'\n GENERATION {i} {round(fit_pop[best], 6)} {round(mean, 6)} {round(std, 6)}')
        file_aux.write(f'\n{i} {round(fit_pop[best], 6)} {round(mean, 6)} {round(std, 6)}')

    # Save generation number
    with open(f"{experiment_name}/gen.txt", 'w') as file_aux:
        file_aux.write(str(i))

    # Save file with the best solution
    np.savetxt(f"{experiment_name}/best.txt", pop[best])

    # Save simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()

# Print execution time for experiment
fim = time.time()
print(f'\nExecution time: {round((fim-ini)/60)} minutes \n')
print(f'\nExecution time: {round(fim-ini)} seconds \n')

# Save control (simulation has ended) file for bash loop file
with open(f"{experiment_name}/neuroended", 'w') as file_aux:
    file_aux.write('')

env.state_to_log()  # checks environment state
