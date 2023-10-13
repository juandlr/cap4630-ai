"""
Project 2 - Traveling Salesman with genetic algorithm
Uses one point crossover and roulette selection of parents with a new population generational approach
Sep 20, 2023
Author: Juan Lopez
Z23635255
"""
import math

import matplotlib.pyplot as plt
import random


def create_population(size, cities_array):
    """
    Creates a population of individuals for a genetic algorithm.

    Args:
        size (int): The number of individuals in the population.
        cities_array (array): the original order.
    Returns:
        list: A list of different order of x,y coordinates, different path.
    """
    population_array = []

    for _ in range(size):
        # Create a copy of the shuffled array
        individual = cities_array.copy()
        # Shuffle the original array in place
        random.shuffle(individual)

        population_array.append(individual)

    return population_array


def calculate_fitness(population):
    fitness_scores = []
    for individual in population:
        individual_distance = calculate_individual_distance(individual)
        fitness_scores.append(individual_distance)

    return fitness_scores


def calculate_individual_distance(individual):
    """
    Calculates the total distance of a single individual
    """
    total_distance = 0
    for i in range(len(individual) - 1):
        distance = calculate_distance(individual[i], individual[i + 1])
        total_distance += distance

    return total_distance


def calculate_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def roulette_selection(population, fitness_scores):
    """
    Selects an individual from the population using roulette wheel selection.

    Args:
        population (list): List of individuals in the population.
        fitness_scores (list): List of fitness scores corresponding to each individual.

    Returns:
        object: The selected individual.
    """
    # Calculate the total inverted fitness score of the population
    total_fitness = sum(1.0 / score for score in fitness_scores)

    # Generate a random value between 0 and the total fitness score
    random_value = random.uniform(0, total_fitness)

    # Initialize variables for tracking the selected individual
    cumulative_fitness = 0
    selected_individual = None

    # Perform roulette wheel selection with inverted fitness scores
    for i in range(len(population)):
        cumulative_fitness += 1.0 / fitness_scores[i]
        if cumulative_fitness >= random_value:
            selected_individual = population[i]
            break

    return selected_individual


def one_point_crossover(parent1, parent2):
    """
    Perform one-point crossover on two parent individuals for the Traveling Salesman Problem.

    Parameters:
    - parent1: The first parent individual (a list of cities).
    - parent2: The second parent individual (a list of cities).

    Returns:
    - offspring1: The first offspring individual (a list of cities).
    - offspring2: The second offspring individual (a list of cities).
    """
    # Choose a random crossover point
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)

    # Create the first offspring by taking the first part of parent1 and filling the rest from parent2
    offspring1 = parent1[:crossover_point]
    for city in parent2:
        if city not in offspring1:
            offspring1.append(city)

    # Create the second offspring by taking the first part of parent2 and filling the rest from parent1
    offspring2 = parent2[:crossover_point]
    for city in parent1:
        if city not in offspring2:
            offspring2.append(city)

    return offspring1, offspring2


def mutate(individual, mutation_rate):
    """
    Mutates a tour by swapping two random cities with a certain mutation rate.

    Args:
        individual (list): A list representing the order of cities (points).
        mutation_rate (float): The probability of mutation for each pair of cities.

    Returns:
        list: A mutated tour.
    """
    # Check if mutation should occur based on the mutation rate
    offspring = list(individual)
    if random.random() < mutation_rate:
        # Choose two distinct random indices for swapping
        idx1, idx2 = random.sample(range(len(offspring)), 2)

        # Swap the cities at the selected indices
        offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]

    return offspring


def create_new_generation(current_population, population_size):
    """
    Generate a new population using one-point crossover, replaces the current population completely using a generational approach.

    Parameters:
    - current_population: The current population as a list of individuals.
    - population_size: The desired size of the new population.

    Returns:
    - new_population: The new population as a list of individuals.
    """
    new_population = []

    while len(new_population) < population_size:
        # generate offspring by using one point crossover
        # parent1 = roulette_selection(current_population, fitness)
        # parent2 = roulette_selection(current_population, fitness)
        # offspring1, offspring2 = one_point_crossover(parent1, parent2)
        # new_population.append(offspring1)
        # new_population.append(offspring2)

        # generate offspring by using mutation
        parent1 = roulette_selection(current_population, fitness)
        parent2 = roulette_selection(current_population, fitness)
        offspring1 = mutate(parent1, 0.1)
        offspring2 = mutate(parent2, 0.1)
        new_population.append(offspring1)
        new_population.append(offspring2)

    # If the new population size is larger than the desired size, truncate it
    if len(new_population) > population_size:
        new_population = new_population[:population_size]

    return new_population


def draw_plot(cities):
    # Create a scatter plot to visualize the random points
    plt.figure(figsize=(6, 6))
    # Extract x and y coordinates from the list of points using list comprehension
    x_coordinates = [point[0] for point in cities]
    y_coordinates = [point[1] for point in cities]
    plt.scatter(x_coordinates, y_coordinates, c='blue', marker='o', s=10)
    distance = calculate_individual_distance(cities)

    for i in range(len(cities) - 1):
        plt.plot([x_coordinates[i], x_coordinates[i + 1]], [y_coordinates[i], y_coordinates[i + 1]], c='red')

    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Randomly Generated Points ' + str(distance))


num_cities = 25
population_size = 100
cities = [] #[[143, 141], [82, 112], [11, 22], [81, 36], [149, 84], [23, 68], [75, 35], [184, 133], [45, 154], [78, 97]]
new_population = []
record_distance = math.inf
fitness = []
best_individual = cities
generations = 500

# create cities
for _ in range(num_cities):
    x = random.randint(0, 200)
    y = random.randint(0, 200)
    cities.append([x, y])

print(cities)

# create initial population
original_population = create_population(population_size, cities)
current_population = original_population[:]
fitness = calculate_fitness(current_population)
for index, value in enumerate(fitness):
    if value < record_distance:
        record_distance = value
        best_individual = current_population[index]


# run evolution for n generations
for i in range(generations):
    current_population = create_new_generation(current_population, population_size)
    fitness = calculate_fitness(current_population)
    for index, value in enumerate(fitness):
        if value < record_distance:
            record_distance = value
            best_individual = current_population[index]

print(record_distance)
print(best_individual)
print(calculate_individual_distance(best_individual))

draw_plot(cities)
draw_plot(best_individual)
plt.show()
