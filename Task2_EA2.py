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
from multiprocessing import Pool, Manager

# Setup EvoMan environment
experiment_name = 'improved_island_ea'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Define the number of islands and migration interval
num_islands = 4
migration_interval = 10

# Initialize the environment
env = Environment(experiment_name=experiment_name,
                  multiplemode="yes",
                  enemies=[2, 4, 6],
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
    for _ in range(len(population)):
        candidates = random.sample(range(len(population)), k)
        selected.append(population[max(candidates, key=lambda i: fitness[i])])
    return np.array(selected)

def crossover(parent1, parent2):
    child = np.copy(parent1)
    mask = np.random.rand(len(parent1)) < 0.5
    child[mask] = parent2[mask]
    return child

def mutation(individual, mutation_rate=0.1):
    mask = np.random.rand(len(individual)) < mutation_rate
    individual[mask] += np.random.normal(0, 0.1, np.sum(mask))
    np.clip(individual, -1, 1, out=individual)
    return individual

def island_ea(island_id, shared_migrants, pop_size=50, n_vars=265, n_gens=100):
    population = initialize_population(pop_size, n_vars)
    best_fitness = float('-inf')
    best_solution = None
    
    for gen in range(n_gens):
        fitness = [evaluate(ind) for ind in population]
        
        # Update best solution
        current_best = max(fitness)
        if current_best > best_fitness:
            best_fitness = current_best
            best_solution = population[np.argmax(fitness)]
        
        # Selection
        parents = tournament_selection(population, fitness)
        
        # Create new population
        new_population = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1 = mutation(crossover(parent1, parent2))
            child2 = mutation(crossover(parent2, parent1))
            new_population.extend([child1, child2])
        
        population = np.array(new_population)
        
        # Migration
        if gen % migration_interval == 0 and gen > 0:
            # Select migrants
            num_migrants = pop_size // 10  # 10% of population
            migrants_indices = np.argsort(fitness)[-num_migrants:]
            migrants = population[migrants_indices]
            
            # Send migrants
            shared_migrants[island_id] = migrants.tolist()  # Use .tolist() to avoid issues with numpy arrays
            
            # Receive migrants
            received_migrants = []
            for i in range(num_islands):
                if i != island_id and i in shared_migrants:
                    received_migrants.extend(shared_migrants[i])
            
            # Integrate migrants
            if received_migrants:
                num_to_replace = min(len(received_migrants), pop_size // 5)  # Replace up to 20% of population
                replace_indices = np.argsort(fitness)[:num_to_replace]
                population[replace_indices] = np.array(received_migrants[:num_to_replace])
        
        mean_fit = np.mean(fitness)
        print(f"Island {island_id}, Generation {gen}: Best Fitness = {best_fitness}, Mean Fitness = {mean_fit}")
    
    return best_solution, best_fitness

def main():
    setup_logging(f"{experiment_name}/experiment_log.txt")
    
    manager = Manager()
    shared_migrants = manager.dict({i: [] for i in range(num_islands)})
    
    with Pool(num_islands) as p:
        results = p.starmap(island_ea, [(i, shared_migrants) for i in range(num_islands)])
    
    best_solution, best_fitness = max(results, key=lambda x: x[1])
    
    logging.info(f"Best overall solution fitness: {best_fitness}")
    np.savetxt(f"{experiment_name}/best_solution.txt", best_solution)

if __name__ == "__main__":
    main()