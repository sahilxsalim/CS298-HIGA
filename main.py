import random
import logging
import time
from concurrent.futures import ProcessPoolExecutor
import sys
import os
from utils import read_processing_times, calculate_makespan, neh_heuristic


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def palmer_heuristic(num_jobs, processing_times):
    """
    Implements Palmer's Slope Index heuristic for PFSP makespan minimization.

    Args:
        num_jobs: The number of jobs.
        processing_times: List of lists with job processing times per machine.

    Returns:
        A list representing the job sequence permutation.
    """
    num_machines = len(processing_times[0])

    if num_machines <= 1:
        return list(range(num_jobs))

    job_indices = []
    # calculate slope index for each job
    for i in range(num_jobs):
        slope_index = 0
        for j in range(num_machines):
            weight = num_machines - 1 - 2 * j
            slope_index += weight * processing_times[i][j]
        job_indices.append((i, slope_index))

    # sort jobs in descending order of their slope index
    job_indices.sort(key=lambda x: x[1], reverse=True)

    final_sequence = [job[0] for job in job_indices]

    return final_sequence

def generate_initial_permutation(num_jobs, processing_times, strategy):
    if strategy == 'random':
        return random.sample(list(range(num_jobs)), num_jobs)
    elif strategy == 'SPT':
        jobs = [(i, sum(processing_times[i])) for i in range(num_jobs)]
        jobs.sort(key=lambda x: x[1])
        return [job[0] for job in jobs]
    elif strategy == 'LPT':
        jobs = [(i, sum(processing_times[i])) for i in range(num_jobs)]
        jobs.sort(key=lambda x: x[1], reverse=True)
        return [job[0] for job in jobs]
    elif strategy == 'NEH':
        return neh_heuristic(processing_times)
    elif strategy == 'Palmer':
        return palmer_heuristic(num_jobs, processing_times)
    else:
        return random.sample(list(range(num_jobs)), num_jobs)

# genetic operators
def tournament_selection(population, tournament_size):
    """select an individual using tournament selection."""
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: x[1])[0]


def rank_based_selection(island):
    """Rank-based selection: select based on fitness rank distribution."""
    # sort individuals by fitness
    sorted_island = sorted(island, key=lambda x: x[1])

    rank_weights = [1/(i+1) for i in range(len(island))]
    total_weight = sum(rank_weights)
    rank_probs = [w/total_weight for w in rank_weights]

    # select based on rank probabilities
    selected_idx = random.choices(
        range(len(island)), weights=rank_probs, k=1)[0]
    return sorted_island[selected_idx][0].copy()


def uniform_selection(island):
    """Uniform selection: select an individual randomly."""
    selected = random.choice(island)
    return selected[0].copy()


def roulette_wheel_selection(island):
    """Roulette wheel selection: select individuals with probability proportional to fitness.
    Since this is a minimization problem, fitnesses are transformed using 1/fitness."""

    transformed_fitness = [1/individual[1] for individual in island]
    
    # calculate selection probabilities
    total_fitness = sum(transformed_fitness)
    
    selection_probs = [fit/total_fitness for fit in transformed_fitness]
    
    # select individual based on these probabilities
    selected_idx = random.choices(range(len(island)), weights=selection_probs, k=1)[0]
    return island[selected_idx][0].copy()


def elitist_selection(island):
    """Elitist selection: select the best individuals based purely on fitness.
    For a minimization problem, these are the individuals with lowest fitness values.
    Returns the two best solutions from the island."""
    # sort individuals by fitness
    sorted_island = sorted(island, key=lambda x: x[1])
    
    # return the top two individuals
    best_solution = sorted_island[0][0].copy()
    second_best_solution = sorted_island[1][0].copy()
    
    return best_solution, second_best_solution


def pmx_crossover(parent1, parent2, crossover_prob):
    """Partially Mapped Crossover (PMX) for permutation problems."""
    if random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    size = len(parent1)
    cxpoint1 = random.randint(0, size - 2)
    cxpoint2 = random.randint(cxpoint1 + 1, size - 1)

    offspring1 = [-1] * size
    offspring2 = [-1] * size

    # Copy crossover segment
    offspring1[cxpoint1:cxpoint2 + 1] = parent1[cxpoint1:cxpoint2 + 1]
    offspring2[cxpoint1:cxpoint2 + 1] = parent2[cxpoint1:cxpoint2 + 1]

    # Define inverse mappings
    inverse_mapping1 = {parent1[i]: parent2[i]
                        for i in range(cxpoint1, cxpoint2 + 1)}
    inverse_mapping2 = {parent2[i]: parent1[i]
                        for i in range(cxpoint1, cxpoint2 + 1)}

    # Fill remaining positions
    for i in list(range(0, cxpoint1)) + list(range(cxpoint2 + 1, size)):
        # Offspring 1
        item1 = parent2[i]
        while item1 in offspring1[cxpoint1:cxpoint2 + 1]:
            item1 = inverse_mapping1.get(item1, item1)
        offspring1[i] = item1

        # Offspring 2
        item2 = parent1[i]
        while item2 in offspring2[cxpoint1:cxpoint2 + 1]:
            item2 = inverse_mapping2.get(item2, item2)
        offspring2[i] = item2

    return offspring1, offspring2


def ox_crossover(parent1, parent2, crossover_prob):
    """Order Crossover (OX) for permutation problems (fixed to match standard OX)."""
    if random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    offspring1 = [-1] * size
    offspring2 = [-1] * size

    # Copy the segment from parent1 to offspring1 and parent2 to offspring2
    offspring1[cxpoint1:cxpoint2+1] = parent1[cxpoint1:cxpoint2+1]
    offspring2[cxpoint1:cxpoint2+1] = parent2[cxpoint1:cxpoint2+1]

    # Fill the rest from the other parent starting from the end of the segment
    def fill_ox(offspring, donor, cxpoint1, cxpoint2):
        size = len(offspring)
        pos = (cxpoint2 + 1) % size
        donor_pos = (cxpoint2 + 1) % size
        while -1 in offspring:
            if donor[donor_pos] not in offspring:
                offspring[pos] = donor[donor_pos]
                pos = (pos + 1) % size
            donor_pos = (donor_pos + 1) % size
        return offspring

    offspring1 = fill_ox(offspring1, parent2, cxpoint1, cxpoint2)
    offspring2 = fill_ox(offspring2, parent1, cxpoint1, cxpoint2)
    return offspring1, offspring2


def lox_crossover(parent1, parent2, crossover_prob):
    """Linear Order Crossover (LOX) for permutation problems."""
    if random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    # Copy the segment from parent1 to offspring1, and parent2 to offspring2
    segment1 = parent1[cxpoint1:cxpoint2+1]
    segment2 = parent2[cxpoint1:cxpoint2+1]

    # For offspring1: take segment1 fill remaining positions in order from parent2
    remaining1 = [gene for gene in parent2 if gene not in segment1]
    offspring1 = remaining1[:cxpoint1] + segment1 + remaining1[cxpoint1:]

    # For offspring2: take segment2 fill remaining positions in order from parent1
    remaining2 = [gene for gene in parent1 if gene not in segment2]
    offspring2 = remaining2[:cxpoint1] + segment2 + remaining2[cxpoint1:]

    return offspring1, offspring2


def swap_mutation(solution, mutation_prob):
    """Swap mutation: exchange two positions in the solution."""
    if random.random() > mutation_prob:
        return solution.copy()
    sol = solution.copy()
    i, j = random.sample(range(len(sol)), 2)
    sol[i], sol[j] = sol[j], sol[i]
    return sol


def inversion_mutation(solution, mutation_prob):
    """Inversion mutation: reverse a segment of the solution."""
    if random.random() > mutation_prob:
        return solution.copy()
    sol = solution.copy()
    i, j = sorted(random.sample(range(len(sol)), 2))
    sol[i:j+1] = sol[i:j+1][::-1]
    return sol


def insertion_mutation(solution, mutation_prob):
    """Insertion mutation: remove a job and insert it at a random position."""
    if random.random() > mutation_prob:
        return solution.copy()
    sol = solution.copy()
    i, j = random.sample(range(len(sol)), 2)
    if i > j:
        i, j = j, i
    job = sol.pop(j)
    sol.insert(i, job)
    return sol

# Island Evolution

def evolve_one_island(island, island_idx, processing_times, population_size, tournament_size, island_params):
    """Evolve one island for one generation.
       Returns the new island population.
    """
    new_island = []
    # Elitism: preserve the best individual
    sorted_island = sorted(island, key=lambda x: x[1])
    best_solution, best_fitness = sorted_island[0]

    new_island.append((best_solution, best_fitness))

    while len(new_island) < population_size:
        if island_idx % 4 == 0:
            parent1, parent2 = elitist_selection(island)
        elif island_idx % 4 == 1:
            parent1 = roulette_wheel_selection(island)
            parent2 = roulette_wheel_selection(island)
        elif island_idx % 4 == 2:
            parent1 = tournament_selection(island, tournament_size)
            parent2 = tournament_selection(island, tournament_size)
        else:
            parent1 = rank_based_selection(island)
            parent2 = rank_based_selection(island)

        if island_idx % 4 == 0:
            offspring1, offspring2 = ox_crossover(
                parent1, parent2, island_params['crossover_prob'])
        elif island_idx % 4 == 1:
            offspring1, offspring2 = lox_crossover(
                parent1, parent2, island_params['crossover_prob'])
        else:
            offspring1, offspring2 = pmx_crossover(
                parent1, parent2, island_params['crossover_prob'])
            

        if island_idx % 4 == 0:
            offspring1 = swap_mutation(offspring1, island_params['mutation_prob'])
            offspring2 = swap_mutation(offspring2, island_params['mutation_prob'])
        elif island_idx % 4 == 1:
            offspring1 = inversion_mutation(
                offspring1, island_params['mutation_prob'])
            offspring2 = inversion_mutation(
                offspring2, island_params['mutation_prob'])
        else:
            offspring1 = insertion_mutation(
                offspring1, island_params['mutation_prob'])
            offspring2 = insertion_mutation(
                offspring2, island_params['mutation_prob'])

        fitness1 = calculate_makespan(offspring1, processing_times)
        fitness2 = calculate_makespan(offspring2, processing_times)

        new_island.append((offspring1, fitness1))
        if len(new_island) < population_size:
            new_island.append((offspring2, fitness2))
    return new_island


def migrate_islands(islands, stagnation_counts, migration_size, migration_direction='source_to_target'):
    """Performs migration between islands based on stagnation counts.
    
    Parameters:
    - islands: List of island populations
    - stagnation_counts: List of stagnation counts per island
    - migration_size: Number of individuals to migrate
    - migration_direction: Either 'source_to_target', 'target_to_source', or 'ring'
    """
    island_indices = sorted(range(len(islands)),
                        key=lambda i: stagnation_counts[i])
    num_islands = len(islands)


    if migration_direction == 'ring':

        migrants_dict = {}
        for i in range(num_islands):
            source_idx = i
            sorted_source = sorted(islands[source_idx], key=lambda x: x[1])
            migrants_dict[source_idx] = sorted_source[:migration_size]


        for i in range(num_islands):
            source_idx = i
            target_idx = (i + 1) % num_islands
            # print(source_idx, target_idx)
            # drop worst in target and add migrants from source
            sorted_target = sorted(islands[target_idx], key=lambda x: x[1], reverse=True)
            sorted_source = sorted(islands[source_idx], key=lambda x: x[1])
            migrants = migrants_dict[source_idx]
            islands[target_idx] = sorted_target[migration_size:] + migrants

        return islands
    
    for k in range(num_islands // 2):
        # Less stagnant island (source)
        source_idx = island_indices[k]
        # More stagnant island (target)
        target_idx = island_indices[num_islands - 1 - k]

        if migration_direction == 'target_to_source': # Swap direction (migrate from more stagnant to less stagnant)
            source_idx, target_idx = target_idx, source_idx


        # Select best individuals from source island
        sorted_source = sorted(islands[source_idx], key=lambda x: x[1])
        migrants = sorted_source[:migration_size]

        # Replace worst individuals in target island
        sorted_target = sorted(
            islands[target_idx], key=lambda x: x[1], reverse=True)
        islands[target_idx] = sorted_target[migration_size:] + migrants

    return islands

# Main Island GA Function


def run_island_ga(processing_times,
                  num_islands=8,
                  population_size=150,
                  tournament_size=5,
                  crossover_prob=0.8,
                  mutation_prob=0.2,
                  migration_interval=10,
                  migration_size=5,
                  max_generations=float('inf'),
                  max_runtime=None,
                  verbose=False,
                  adaptation_threshold=15,
                  stagnation_threshold=30,
                  migration_direction='target_to_source'):
    """
    Run the island genetic algorithm with stagnation-based migration.

    Parameters:
    - processing_times: List of lists with job processing times per machine
    - num_islands: Number of islands
    - population_size: Size of each island's population
    - tournament_size: Size of tournament for selection
    - crossover_prob: Probability of crossover
    - mutation_prob: Probability of mutation
    - migration_interval: Generations between migrations
    - migration_size: Number of individuals to migrate
    - max_generations: Maximum number of generations
    - max_runtime: Maximum runtime in seconds
    - verbose: Enable detailed logging
    - adaptation_threshold: Generations without improvement before adapting parameters
    - stagnation_threshold: Generations without improvement before restarting islands
    - migration_direction: Either 'source_to_target', 'target_to_source', or 'ring'
    """
    # Logging configuration
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    num_jobs = len(processing_times)
    strategies = ['NEH', 'Palmer', 'SPT', 'LPT']

    neh_solution = neh_heuristic(processing_times)
    neh_fitness = calculate_makespan(neh_solution, processing_times)
    
    islands = []
    for i in range(num_islands):
        island = [(neh_solution, neh_fitness)]
        strategy = strategies[i % len(strategies)]
        init_perm = generate_initial_permutation(
            num_jobs, processing_times, strategy)
        init_fit = calculate_makespan(init_perm, processing_times)
        island.append((init_perm, init_fit))

        while len(island) < population_size:
            perm = generate_initial_permutation(
                num_jobs, processing_times, 'random')
            fit = calculate_makespan(perm, processing_times)
            island.append((perm, fit))
        islands.append(island)

    best_solution = neh_solution
    best_fitness = neh_fitness

    # initialize island-specific parameters
    island_params = []
    for _ in range(num_islands):
        params = {
            'crossover_prob': crossover_prob,
            'mutation_prob': mutation_prob,
            'adaptation_threshold': adaptation_threshold,
            'stagnation_threshold': stagnation_threshold
        }
        island_params.append(params)

    start_time = time.time()
    generation = 0
    prev_best = best_fitness
    no_improvement_count = 0
    makespan_history = []

    # initialize stagnation tracking
    previous_best_fitness = [
        min(fitness for _, fitness in island) for island in islands]
    stagnation_counts = [0] * num_islands

    with ProcessPoolExecutor() as executor:
        while generation < max_generations and (max_runtime is None or (time.time() - start_time) < max_runtime):
            # evolve islands in parallel
            futures = [executor.submit(evolve_one_island, island, i, processing_times,
                                       population_size, tournament_size, island_params[i])
                       for i, island in enumerate(islands)]
            islands = [f.result() for f in futures]
            generation += 1

            # update stagnation counts and adapt island-specific parameters
            for i in range(num_islands):
                current_best = min(fitness for _, fitness in islands[i])
                if current_best < previous_best_fitness[i]:
                    previous_best_fitness[i] = current_best
                    stagnation_counts[i] = 0
                else:
                    stagnation_counts[i] += 1
                
                # adapt island-specific parameters based on stagnation
                if stagnation_counts[i] > island_params[i]['adaptation_threshold']:
                    island_params[i]['mutation_prob'] = min(0.4, island_params[i]['mutation_prob'] * 1.02)
                    island_params[i]['crossover_prob'] = max(0.6, island_params[i]['crossover_prob'] * 0.98)
                    logging.info(f"Island {i} adapted: mutation_prob={island_params[i]['mutation_prob']:.3f}, crossover_prob={island_params[i]['crossover_prob']:.3f}")

            # update global best
            for island in islands:
                for sol, fit in island:
                    if fit < best_fitness:
                        best_solution = sol.copy()
                        best_fitness = fit

            makespan_history.append(best_fitness)

            if best_fitness < prev_best:
                no_improvement_count = 0
                prev_best = best_fitness
            else:
                no_improvement_count += 1

            # check stagnation and restart islands if needed
            for i, island in enumerate(islands):
                if stagnation_counts[i] > island_params[i]['stagnation_threshold']:
                    logging.info(f"Restarting island {i} due to stagnation. Stagnation count: {stagnation_counts[i]}")
                    sorted_island = sorted(island, key=lambda x: x[1])
                    new_island = [sorted_island[0]]
                    while len(new_island) < population_size:
                        permutation = generate_initial_permutation(
                            num_jobs, processing_times, 'random')
                        fit = calculate_makespan(
                            permutation, processing_times)
                        new_island.append((permutation, fit))
                    islands[i] = new_island
                    stagnation_counts[i] = 0
                    # reset parameters for restarted island
                    island_params[i]['mutation_prob'] = mutation_prob
                    island_params[i]['crossover_prob'] = crossover_prob

            # migration with diversity measure check
            if generation % migration_interval == 0:
                islands = migrate_islands(islands, stagnation_counts, migration_size, migration_direction)

            if verbose:
                logging.info(
                    f"Generation {generation}: Best makespan = {best_fitness}")

    total_runtime = time.time() - start_time
    return best_solution, best_fitness, makespan_history, total_runtime


def run_miga_instance(processing_times, random_seed):
    """
    Runs the MIGA algorithm for a given instance and returns the best makespan.
    Uses default parameters as defined in the original main block.
    """
    random.seed(random_seed)
    
    best_solution, best_fitness, _, execution_time = run_island_ga(
        processing_times,
        num_islands=4,
        population_size=100,
        tournament_size=5,
        crossover_prob=0.8,
        mutation_prob=0.1,
        migration_interval=50,
        migration_size=5,
        max_generations=300, 
        verbose=False,  
        adaptation_threshold=15,
        stagnation_threshold=30,
        migration_direction='target_to_source'
    )
    return best_fitness, execution_time


if __name__ == "__main__":
    processing_times = read_processing_times('testcase.txt')

    start_time = time.time()
    random.seed(42)
    best_solution, best_fitness, makespan_history, execution_time = run_island_ga(
        processing_times,
        num_islands=4,
        population_size=100,
        tournament_size=3,
        crossover_prob=0.8,
        mutation_prob=0.1,
        migration_interval=10,
        migration_size=2,
        max_generations=300,
        adaptation_threshold=15,
        stagnation_threshold=30,
        migration_direction='target_to_source')
    
    print(f"Target to Source migration best makespan: {best_fitness}")
    print(f"Target to Source migration execution time: {execution_time:.2f} seconds")   