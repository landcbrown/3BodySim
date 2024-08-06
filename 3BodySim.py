import pygame
import numpy as np
import time
import csv

# Constants
G = 1.0  # Gravitational constant
m = 3.0  # Mass of each body
dt = 0.01  # Time step for simulation
screen_size = 1000  # Size of the display window
trail_length = 10000  # Number of past positions to keep for each trail
fps = 25  # Frames per second for simulation

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((screen_size, screen_size))  # Set up the display window
pygame.display.set_caption("Three-Body Simulation")  # Set the window title
clock = pygame.time.Clock()  # Create a clock object to manage frame rate


def initialize_conditions():
    """
    Initialize the positions and velocities of the bodies.

    Returns:
    - positions: A NumPy array of shape (3, 2) containing the initial (x, y) positions of the three bodies.
    - velocities: A NumPy array of shape (3, 2) containing the initial (vx, vy) velocities of the three bodies.
    """
    angles = np.linspace(0, 2 * np.pi, 4)[:-1]  # Create angles for 3 bodies evenly spaced on a circle
    # Add random perturbation to both positions and velocities
    positions = np.array([[np.cos(angle) + np.random.uniform(-0.1, 0.1),
                           np.sin(angle) + np.random.uniform(-0.1, 0.1)]
                          for angle in angles])
    velocities = np.array([[-np.sin(angle) + np.random.uniform(-0.1, 0.1),
                            np.cos(angle) + np.random.uniform(-0.1, 0.1)]
                           for angle in angles])
    return positions, velocities


# Initialize positions and velocities of the bodies
positions, velocities = initialize_conditions()

# Define colors for each body
colors = [(255, 0, 0),  # Red
          (0, 255, 0),  # Green
          (0, 0, 255)]  # Blue

# Initialize trails for each body (for visualizing the paths)
trails = [[] for _ in range(len(positions))]

# Record the start time of the simulation
start_time = time.time()


def gravitational_force(pos1, pos2):
    """
    Calculate the gravitational force exerted on one body by another.

    Parameters:
    - pos1: A NumPy array with shape (2,) representing the position of the first body.
    - pos2: A NumPy array with shape (2,) representing the position of the second body.

    Returns:
    - force: A NumPy array with shape (2,) representing the gravitational force on the first body due to the second body.
    """
    r = pos2 - pos1
    distance = np.linalg.norm(r)  # Calculate the distance between the two bodies
    force_magnitude = G * m * m / distance ** 2  # Calculate the magnitude of the gravitational force
    force = force_magnitude * r / distance  # Calculate the force vector
    return force


def compute_accelerations(positions):
    """
    Compute the accelerations of all bodies based on gravitational forces.

    Parameters:
    - positions: A NumPy array of shape (N, 2) where N is the number of bodies, representing their positions.

    Returns:
    - accelerations: A NumPy array of shape (N, 2) where each row represents the acceleration of a body.
    """
    accelerations = np.zeros_like(positions)  # Initialize accelerations
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                accelerations[i] += gravitational_force(positions[i],
                                                        positions[j]) / m  # Sum up forces from all other bodies
    return accelerations


def runge_kutta_step(positions, velocities, dt):
    """
    Perform one step of the Runge-Kutta integration to update positions and velocities.

    Parameters:
    - positions: A NumPy array of shape (N, 2) where N is the number of bodies, representing their current positions.
    - velocities: A NumPy array of shape (N, 2) where N is the number of bodies, representing their current velocities.
    - dt: A float representing the time step for the integration.

    Returns:
    - new_positions: A NumPy array of shape (N, 2) representing the updated positions of the bodies.
    - new_velocities: A NumPy array of shape (N, 2) representing the updated velocities of the bodies.
    """
    k1_v = compute_accelerations(positions)  # Compute k1 for velocities
    k1_p = velocities  # k1 for positions

    k2_v = compute_accelerations(positions + 0.5 * dt * k1_p)  # Compute k2 for velocities
    k2_p = velocities + 0.5 * dt * k1_v  # k2 for positions

    k3_v = compute_accelerations(positions + 0.5 * dt * k2_p)  # Compute k3 for velocities
    k3_p = velocities + 0.5 * dt * k2_v  # k3 for positions

    k4_v = compute_accelerations(positions + dt * k3_p)  # Compute k4 for velocities
    k4_p = velocities + dt * k3_v  # k4 for positions

    # Update positions and velocities using weighted average of k1, k2, k3, k4
    new_positions = positions + (dt / 6) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
    new_velocities = velocities + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return new_positions, new_velocities


# Initialize lists to store data for later analysis
timestamps = []
positions_data = []
velocities_data = []

# Main simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Exit the loop if the window is closed
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False  # Exit the loop if the Escape key is pressed

    positions, velocities = runge_kutta_step(positions, velocities, dt)  # Update positions and velocities

    # Update trails (store past positions for visualization)
    for i in range(len(trails)):
        trails[i].append(tuple(positions[i]))
        if len(trails[i]) > trail_length:
            trails[i].pop(0)  # Maintain the maximum length of the trail

    # Compute elapsed time
    elapsed_time = time.time() - start_time
    timestamps.append(elapsed_time)  # Store the current time
    positions_data.append(positions.copy())  # Store current positions
    velocities_data.append(velocities.copy())  # Store current velocities

    screen.fill((0, 0, 0))  # Clear the screen with black

    # Draw trails
    for i, trail in enumerate(trails):
        if len(trail) > 1:
            pygame.draw.lines(screen, colors[i], False,
                              [(int(screen_size / 2 + x * 100), int(screen_size / 2 + y * 100)) for x, y in trail], 1)

    # Draw bodies
    for i, pos in enumerate(positions):
        x = int(screen_size / 2 + pos[0] * 100)
        y = int(screen_size / 2 + pos[1] * 100)
        pygame.draw.circle(screen, colors[i], (x, y), 5)  # Draw each body as a circle

    # Display elapsed time
    font = pygame.font.Font(None, 36)
    time_text = font.render(f"Time Elapsed: {elapsed_time:.2f} s", True, (255, 255, 255))
    screen.blit(time_text, (10, 10))  # Draw the elapsed time on the screen

    pygame.display.flip()  # Update the display
    clock.tick(fps)  # Control the frame rate

pygame.quit()  # Quit Pygame


def save_data_to_csv(filename, positions_data, velocities_data, timestamps):
    """
    Save the simulation data to a CSV file with color labels.

    Parameters:
    - filename: The name of the file to save the data to.
    - positions_data: A list of arrays where each array contains the positions of the bodies at a given time step.
    - velocities_data: A list of arrays where each array contains the velocities of the bodies at a given time step.
    - timestamps: A list of timestamps corresponding to each time step.
    """
    # Define color labels for CSV output
    color_labels = ['Red', 'Green', 'Blue']

    # Open the file for writing
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(['Time', 'Color', 'X_Position', 'Y_Position', 'X_Velocity', 'Y_Velocity'])

        # Write the data rows
        for t, (positions, velocities) in zip(timestamps, zip(positions_data, velocities_data)):
            for i in range(len(positions)):
                writer.writerow(
                    [t, color_labels[i], positions[i][0], positions[i][1], velocities[i][0], velocities[i][1]])


# Save data after the simulation ends
save_data_to_csv('simulation_data.csv', positions_data, velocities_data, timestamps)
