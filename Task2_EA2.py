###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Younes Boubbou			                                  #
#     				                                  #
###############################################################################

import logging
import sys
import os
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
import random
from multiprocessing import Pool

# Setup EvoMan environment
experiment_name = sys.argv[1]

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

if int(sys.argv[2]) == 1:
    enemies_list = [2,5,6,7,8]
elif int(sys.argv[2]) == 2:
    enemies_list = [6,7,8] 

# Define the number of islands and migration interval
num_islands = 4
migration_interval = 10
alpha = 0.51
dom_u = 1
dom_l = -1

# Initialize the environment
env = Environment(experiment_name=experiment_name,
                  multiplemode="yes",
                  enemies=enemies_list,
                  playermode="ai",
                  player_controller=player_controller(10),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# Choose this for not using visuals and thus making experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Logging setup
def setup_logging(log_filename):
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

def evaluate(x):
    return simulation(env, x)

def initialize_population(pop_size, n_vars):
    return np.random.uniform(-1, 1, (pop_size, n_vars))

def tournament_selection(population, fitness, k=3):
    selected = []
    for _ in range(2*len(population)):
        candidates = random.sample(range(len(population)), k)
        selected.append(population[max(candidates, key=lambda i: fitness[i])])
    return np.array(selected)

def crossover(parent1, parent2):
    o1 = []
    for i in range(0, len(parent1)):
        d_i = np.abs(parent1[i] - parent2[i])
        lower_bound = min(parent1[i], parent2[i]) - alpha * d_i
        upper_bound = max(parent1[i], parent2[i]) + alpha * d_i
        o1_gene_i = np.random.uniform(lower_bound, upper_bound)
        o1.append(o1_gene_i)
    return o1

# limits
def limits(x):
    if x > dom_u:
        return dom_u
    elif x < dom_l:
        return dom_l
    else:
        return x

def mutation(offspring, gen, n_gens, mutation_rate=0.1):
    sigma_gen = 1 - 0.9 * (gen / n_gens)
    for i in range(0, len(offspring)):
        if np.random.uniform(0, 1) <= mutation_rate:
            offspring[i] = offspring[i] + np.random.normal(0, sigma_gen)
    offspring = np.array(list(map(lambda y: limits(y), offspring)))
    return offspring

def island_ea(island_id, pop_size=25, n_vars=265, n_gens=30, elitism=True, elitism_rate=0.04):
    population = initialize_population(pop_size, n_vars)
    best_fitness = float('-inf')
    best_solution = None
    mean_fitnesses = []
    
    num_elite = max(1, int(elitism_rate * pop_size))  # Number of elite individuals to preserve

    for gen in range(n_gens):
        fitness = [evaluate(ind) for ind in population]
        
        # Find the best individuals in the current population
        best_indices = np.argsort(fitness)[-num_elite:]  # Indices of the best individuals
        elites = population[best_indices]  # Select elites
        
        # Update best solution
        current_best = max(fitness)
        if current_best > best_fitness:
            best_fitness = current_best
            best_solution = population[np.argmax(fitness)]
        
        # Selection
        parents = tournament_selection(population, fitness)
        
        # Create new population
        new_population = []
        for i in range(0, 2 * pop_size - num_elite, 2):  # Adjust for elitism
            parent1, parent2 = parents[i], parents[i + 1]
            child1 = mutation(crossover(parent1, parent2), gen, n_gens)
            new_population.extend([child1])
        
        # Add the elite individuals to the new population
        new_population.extend(elites.tolist())
        population = np.array(new_population)
        
        mean_fit = np.mean(fitness)
        mean_fitnesses.append(mean_fit)

    return best_fitness, np.mean(mean_fitnesses), best_solution  # Return best solution too

def main():
    setup_logging(f"{experiment_name}/experiment_log.txt")
    
    n_gens = 30
    
    # Open the results file to save the generations data
    results_file_path = f"{experiment_name}/results.txt"
    best_overall_solution = None
    best_overall_fitness = float('-inf')
    
    with open(results_file_path, "w") as results_file:
        # Write header for the results file
        results_file.write(f"{'gen':<5}{'best':>12}{'mean':>12}\n")
        
        # Create the process pool once and reuse it across all generations
        with Pool(num_islands) as p:
            for gen in range(n_gens):
                # Collect fitness data from all islands
                results = p.starmap(island_ea, [(i, 25, 265, 1) for i in range(num_islands)])
                
                # Get global best fitness and mean fitness for this generation
                global_best_fitness = max([result[0] for result in results])
                global_mean_fitness = np.mean([result[1] for result in results])
                
                # Update best overall solution if found
                for result in results:
                    if result[0] > best_overall_fitness:
                        best_overall_fitness = result[0]
                        best_overall_solution = result[2]
                
                # Write the statistics for this generation to the results file
                results_file.write(f"{gen:<5}{global_best_fitness:>12.6f}{global_mean_fitness:>12.6f}\n")
    
    # Save the final best solution to a file at the end
    best_solution_path = f"{experiment_name}/best_overall_solution.txt"
    np.savetxt(best_solution_path, best_overall_solution)
    logging.info(f"Best overall solution fitness: {best_overall_fitness}")

if __name__ == "__main__":
    main()
