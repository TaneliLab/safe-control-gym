import numpy as np
import matplotlib.pyplot as plt
# Corrected example data where the line intersects the plane
# Plane: point and normal
P0 = np.array([1, 2, 3])  # Point on the plane
N = np.array([1, -2, 1])  # Normal vector of the plane

# Line: two points
L1 = np.array([-3, -2, 0])  # First point on the line
L2 = np.array([1, 2, 5])  # Second point on the line, ensuring it's not parallel to the plane

# Compute direction vector of the line (L2 - L1)
d = L2 - L1

# Calculate intersection
t = np.dot(N, P0 - L1) / np.dot(N, d)
intersection = L1 + t * d

# Create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the line
line_x = [L1[0], L2[0]]
line_y = [L1[1], L2[1]]
line_z = [L1[2], L2[2]]
ax.plot(line_x, line_y, line_z, label='Line', color='blue')

# Extend the line for visualization purposes
extended_line_x = [L1[0], intersection[0]]
extended_line_y = [L1[1], intersection[1]]
extended_line_z = [L1[2], intersection[2]]
ax.plot(extended_line_x, extended_line_y, extended_line_z, label='Extended Line', color='green')

# Plot the plane
xx, yy = np.meshgrid(range(10), range(10))
z = (-N[0] * xx - N[1] * yy + np.dot(N, P0)) * 1. /N[2]
ax.plot_surface(xx, yy, z, alpha=0.5, rstride=100, cstride=100)

# Plot intersection point
ax.scatter(intersection[0], intersection[1], intersection[2], color='red', s=100, label='Intersection')

# Labels and legend
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()

# Show plot
plt.show()

