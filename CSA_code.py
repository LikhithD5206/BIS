import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Function to generate a random path (cuckoo nest)
def generate_random_path(cities):
    path = list(cities)
    random.shuffle(path)
    return path

# Function to calculate total distance for a given path
def calculate_total_distance(path):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += euclidean(path[i], path[i + 1])
    total_distance += euclidean(path[-1], path[0])  # Return to start
    return total_distance

# Function to generate Levy flight
def levy_flight(dim, beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    step = np.random.normal(0, sigma, size=dim) * np.random.normal(0, 1, size=dim)
    return step

# Function to apply Cuckoo Search Algorithm for TSP
def cuckoo_search(cities, n_nests=20, max_iter=500, p_a=0.25, alpha=0.1):
    # Initialize nests (paths)
    nests = [generate_random_path(cities) for _ in range(n_nests)]
    fitness = [calculate_total_distance(nest) for nest in nests]
    
    best_nest = nests[np.argmin(fitness)]
    best_fitness = min(fitness)

    # Iterative process
    for _ in range(max_iter):
        # Generate new solutions (nests) by Levy flight
        new_nests = []
        for nest in nests:
            step = levy_flight(len(cities))
            new_nest = np.copy(nest)
            for i in range(len(nest)):
                new_nest[i] = nest[i] + step[i]  # Update the path by step
            new_nests.append(new_nest)

        # Evaluate fitness for the new nests
        new_fitness = [calculate_total_distance(new_nest) for new_nest in new_nests]

        # Compare and select better nests
        for i in range(n_nests):
            if new_fitness[i] < fitness[i]:
                nests[i] = new_nests[i]
                fitness[i] = new_fitness[i]

        # Randomly replace some nests (exploration)
        for i in range(int(p_a * n_nests)):
            nests[i] = generate_random_path(cities)
            fitness[i] = calculate_total_distance(nests[i])

        # Find the best nest
        min_fitness_idx = np.argmin(fitness)
        if fitness[min_fitness_idx] < best_fitness:
            best_nest = nests[min_fitness_idx]
            best_fitness = fitness[min_fitness_idx]
    
    return best_nest, best_fitness

# Generate random cities (coordinates in 2D)
num_cities = 10
cities = np.random.rand(num_cities, 2)  # Random 2D points

# Run the Cuckoo Search Algorithm
best_path, best_distance = cuckoo_search(cities)

# Plotting the result
fig, ax = plt.subplots()
ax.scatter(cities[:, 0], cities[:, 1], c='red', marker='o')

# Plot the best path
for i in range(len(best_path)-1):
    ax.plot([best_path[i][0], best_path[i+1][0]], [best_path[i][1], best_path[i+1][1]], 'b-')

ax.plot([best_path[-1][0], best_path[0][0]], [best_path[-1][1], best_path[0][1]], 'b-')  # Connect back to start
ax.set_title(f"Best Path with Total Distance: {best_distance:.2f}")
plt.show()

print(f"Best Path: {best_path}")
print(f"Best Total Distance: {best_distance:.2f}")
