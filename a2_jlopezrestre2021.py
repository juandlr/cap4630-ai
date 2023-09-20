import matplotlib.pyplot as plt
import random

# Number of random points to generate
num_points = 50

# Initialize an empty list to store (x, y) coordinate pairs
points = []

# Generate random (x, y) coordinate pairs and save them in the list
for _ in range(num_points):
    x = random.randint(0, 200)
    y = random.randint(0, 200)
    points.append([x, y])

# Create a scatter plot to visualize the random points
plt.figure(figsize=(6, 6))
# Extract x and y coordinates from the list of points using list comprehension
x_coordinates = [point[0] for point in points]
y_coordinates = [point[1] for point in points]
plt.scatter(x_coordinates, y_coordinates, c='blue', marker='o', s=10)
plt.xlim(0, 200)
plt.ylim(0, 200)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Randomly Generated Points')
plt.show()
