import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def generate_non_overlapping_circles(width, height, radii, num_circles, min_distance):
    circle_positions = []

    def does_circle_overlap(x, y, radius):
        for cx, cy, cradius in circle_positions:
            if np.sqrt((x - cx)**2 + (y - cy)**2) < radius + cradius + min_distance:
                return True
        return False

    for radius, num in zip(radii, num_circles):
        for _ in range(num):
            while True:
                x = np.random.uniform(radius, width - radius)
                y = np.random.uniform(radius, height - radius)
                if not does_circle_overlap(x, y, radius):
                    break
            circle_positions.append((x, y, radius))

    return circle_positions

# Define the rectangle dimensions
width = 1500
height = 1500

# Define the sizes of the three circles (radii)
radii = [5, 7, 3]

# Number of circles of each size
num_circles = [1000, 1000, 1000]

# Minimum specified distance between circles
min_distance = 1  # 1 is close, 5 is far

# Generate non-overlapping circles
circle_positions = generate_non_overlapping_circles(width, height, radii, num_circles, min_distance)

# Adjust figure size for higher quality
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_aspect('equal', adjustable='box')

# Hide axis and labels
ax.axis('off')

for x, y, radius in circle_positions:
    circle = Circle((x, y), radius, fill=True)  # fill=True for filled circles
    ax.add_patch(circle)

# Save the figure with higher dpi
plt.savefig('circles_1500by1500_N_3000_min_dist_1.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()
