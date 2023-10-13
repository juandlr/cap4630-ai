"""
Project 3 -
ACO TSP
Oct 7, 2023
Author: Juan Lopez
Z23635255
"""
import math
import matplotlib.pyplot as plt
import random
import numpy as np


class Ant:
    def __init__(self):
        self.tour = []  # List to store the ant's memory
        self.is_best = False
        start_position = random.randint(0, attraction_count - 1)
        self.visit_attraction(start_position)

    def visit_attraction(self, attraction):
        """
        Visit a specific attraction and update the ant's memory and action.
        """
        self.tour.append(attraction)

    def visit_random_attraction(self, attractions):
        """
        Visit a random attraction from a list of attractions and update the ant's memory and action.
        """
        attraction = random.choice(attractions)
        self.tour.append(attraction)

    def visit_probabilistic_attraction(self, pheromone_trails, attraction_count, distance_matrix, alpha, beta):
        """
        Visit an attraction probabilistically based on pheromone levels and attractiveness.
        """
        current_attraction = self.tour[-1]
        all_attractions = list(range(0, attraction_count))
        possible_attractions = list(set(all_attractions) - set(self.tour))

        possible_indexes = []
        possible_probabilities = []
        total_probabilities = 0

        for attraction in possible_attractions:
            possible_indexes.append(attraction)
            pheromones_on_path = math.pow(pheromone_trails[current_attraction][attraction], alpha)
            heuristic_for_path = math.pow(1 / distance_matrix[current_attraction][attraction], beta)
            probability = pheromones_on_path * heuristic_for_path
            possible_probabilities.append(probability)
            total_probabilities += probability

        possible_probabilities = [probability / total_probabilities for probability in possible_probabilities]

        return [possible_indexes, possible_probabilities]

    def get_distance_traveled(self):
        """
        Calculate and return the distance traveled by the ant based on its memory.
        """
        total_distance = 0
        for i in range(len(self.tour) - 1):  # Loop through ant's memory
            from_attraction = self.tour[i]
            to_attraction = self.tour[i + 1]
            total_distance += distance_matrix[from_attraction][to_attraction]
        return total_distance


def roulette_wheel_selection(possible_indexes, possible_probabilities):
    """
    Perform roulette wheel selection to choose an attraction based on fitness values.
    """
    total_probability = sum(possible_probabilities)
    roulette_spin = random.uniform(0, total_probability)
    cumulative_probability = 0.0
    selected_index = None

    for index, probability in zip(possible_indexes, possible_probabilities):
        cumulative_probability += probability
        if cumulative_probability >= roulette_spin:
            selected_index = index
            break

    return selected_index


def setup_ants(attraction_count, number_of_ants_factor):
    """
    Set up the population of ants.
    @param attraction_count:
    @param number_of_ants_factor:
    @return:
    """
    ant_colony = []

    # Number of ants
    num_ants = int(attraction_count * number_of_ants_factor)

    # Initialize ants
    for ant_id in range(num_ants):
        # Initialize an ant with a random starting position (attraction)
        start_position = random.randint(0, attraction_count - 1)
        ant = Ant()
        ant_colony.append(ant)

    return ant_colony


def update_pheromones(evaporation_rate, pheromone_trails, attraction_count, ant_colony):
    """
        Update pheromone levels on the pheromone trails matrix based on ant tours and evaporation.
        Args:
            evaporation_rate (float): The rate at which pheromone evaporates (0.0 to 1.0).
            pheromone_trails (list): A 2D list representing the pheromone levels on edges.
            attraction_count (int): The number of attractions or cities.
            ant_colony (list): A list of ants, each with a tour.

        Returns:
            None
    """
    for x in range(attraction_count):
        for y in range(attraction_count):
            pheromone_trails[x][y] *= evaporation_rate  # Evaporate pheromone

    for ant in ant_colony:
        ant_distance_traveled = ant.get_distance_traveled()
        for i in range(len(ant.tour) - 1):
            current_attraction = ant.tour[i]
            next_attraction = ant.tour[i + 1]
            pheromone_trails[current_attraction][next_attraction] += 1.0 / ant_distance_traveled


def draw_plot(ant, cities):
    # Create a scatter plot to visualize the random points
    plt.figure(figsize=(6, 6))
    # Extract x and y coordinates from the list of points using list comprehension
    x_coordinates = [point[0] for point in cities]
    y_coordinates = [point[1] for point in cities]
    plt.scatter(x_coordinates, y_coordinates, c='blue', marker='o', s=10)
    distance = ant.get_distance_traveled()

    for i in range(len(cities) - 1):
        plt.plot([x_coordinates[i], x_coordinates[i + 1]], [y_coordinates[i], y_coordinates[i + 1]], c='red')

    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Randomly Generated Points ' + str(distance))


def distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum((point1 - point2) ** 2))


def generate_distance_matrix(points):
    num_points = len(points)
    city_distances = [[0] * num_points for _ in range(num_points)]

    for i in range(num_points):
        for j in range(num_points):
            city_distances[i][j] = distance(points[i], points[j])

    return city_distances


def init_pheromone_trails(size):
    """
    creates the initial pheromone_trails matrix
    @param size:
    @return:
    """
    # Create a 10x10 2D list filled with 1s
    matrix = [[1] * size for _ in range(size)]
    return matrix


# pheromone_trails = [
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1]
# ]
# city_distances = [
#     [0, 8, 7, 4, 6, 4],
#     [8, 0, 5, 7, 11, 5],
#     [7, 5, 0, 9, 6, 7],
#     [4, 7, 9, 0, 5, 6],
#     [6, 11, 6, 5, 0, 3],
#     [4, 5, 7, 6, 3, 0]
# ]
attraction_count = 10  # Number of attractions
alpha = 1.0  # Pheromone influence
beta = 2.0  # Heuristic influence
number_of_ants_factor = 0.5  # Fraction of attractions to use as the number of ants
evaporation_rate = 0.5
best_distance = math.inf
best_ant = None
total_iterations = 100
cities = [[143, 141], [82, 112], [11, 22], [81, 36], [149, 84], [23, 68], [75, 35], [184, 133], [45, 154], [78, 97]]
distance_matrix = generate_distance_matrix(cities)
pheromone_trails = init_pheromone_trails(attraction_count)

# initiates the program
for i in range(total_iterations):
    ant_colony = setup_ants(attraction_count, number_of_ants_factor)
    for ant in ant_colony:
        while len(ant.tour) != attraction_count:
            attraction_info = ant.visit_probabilistic_attraction(pheromone_trails, attraction_count, distance_matrix, alpha, beta)
            attraction_index = roulette_wheel_selection(attraction_info[0], attraction_info[1])
            ant.visit_attraction(attraction_index)

        ant_distance = ant.get_distance_traveled()
        print(ant_distance)
        if ant_distance < best_distance:
            best_distance = ant.get_distance_traveled()
            print(best_distance)
            best_ant = ant

best_ant.is_best = True
update_pheromones(0.5, pheromone_trails, attraction_count, ant_colony)
print(best_distance)
print(best_ant)
print(best_ant.tour)
rearranged_cities = [cities[i] for i in best_ant.tour]

# Print the rearranged list
for element in rearranged_cities:
    print(element)


sample_ant = Ant()
ant.tour = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
draw_plot(sample_ant, cities)
draw_plot(best_ant, rearranged_cities)
plt.show()
