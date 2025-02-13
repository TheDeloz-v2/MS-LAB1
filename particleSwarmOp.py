'''
Lab 1 - Modeling and Simulation
Date: 07/2024

Implementation of a particle swarm optimization algorithm
to find the minimum of a function.
'''

import random as r
import numpy as np
import matplotlib.pyplot as plt

'''
Define the function to be minimized.

Parameters:
    x (float): x input value
    y (float): y input value
    
Returns:
    float: The output value of the function
'''
def f(x, y):
    return (x-3)**2 + (y-2)**2

'''
initialize the list of particles

Parameters:
    n (int): number of particles
    x_min (float): minimum value of x (x_min = y_min)
    x_max (float): maximum value of x (x_max = y_max)
    v_min (float): minimum value of the velocity
    v_max (float): maximum value of the velocity
    
Returns:
    list: list of particles
'''
def initialize_particles(n, x_min, x_max, v_min, v_max):
    particles = []

    for _ in range(n):
        x = r.uniform(x_min, x_max)
        y = r.uniform(x_min, x_max)
        vx = r.uniform(v_min, v_max)
        vy = r.uniform(v_min, v_max)
        particles.append({"x": x, "y": y, "vx": vx, "vy": vy, "best_x": x, "best_y": y, "best_f": f(x, y)})
        
    return particles

def plot_contour(particles, global_best, iteration):
    X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    Z = f(X, Y)
    plt.contourf(X, Y, Z, levels=10, cmap="plasma")
    for p in particles:
        plt.scatter(p["x"], p["y"], color='g')
    plt.scatter(global_best["best_x"], global_best["best_y"], color='r')
    plt.title(f'Iteration {iteration}')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()

'''
Main function
'''
def main():
    # Number of particles
    n = 40

    # Search space
    x_min = -10
    x_max = 10

    # Velocity
    v_min = -1
    v_max = 1

    # Inertia weight
    w = 0.5

    # Hiperparameters
    c1 = 1.5
    c2 = 1.5

    # Number of iterations
    iterations = 100
    threshold = 1e-15
    
    # Initialize particles
    particles = initialize_particles(n, x_min, x_max, v_min, v_max)
    
    # Global best particle
    global_best = particles[0]
    for i in range(1, n):
        if particles[i]["best_f"] < global_best["best_f"]:
            global_best = particles[i]
        
    for i in range(iterations):
        
        r.seed(777)
        
        if i == 0:
            plot_contour(particles, global_best, i)
        
        for particle in particles:
            r1 = r.random()
            r2 = r.random()
            
            # Update particle velocity
            particle["vx"] = w*particle["vx"] + c1*r1*(particle["best_x"] - particle["x"]) + c2*r2*(global_best["best_x"] - particle["x"])
            particle["vy"] = w*particle["vy"] + c1*r1*(particle["best_y"] - particle["y"]) + c2*r2*(global_best["best_y"] - particle["y"])
            
            # Update particle position
            particle["x"] += particle["vx"]
            particle["y"] += particle["vy"]
            
            # Update particle best position
            if f(particle["x"], particle["y"]) < particle["best_f"]:
                particle["best_x"] = particle["x"]
                particle["best_y"] = particle["y"]
                particle["best_f"] = f(particle["x"], particle["y"])
            
            # Update global best position
            if particle["best_f"] < global_best["best_f"]:
                global_best = particle
            
        print("Iteration", i, ":", global_best["best_f"])
        
        if i % 10 == 0 and i != 0 or i == iterations - 1:
            plot_contour(particles, global_best, i)
            
        if global_best["best_f"] < threshold:
            print(f"\nStopping early at iteration {i} with global best f-value: {global_best['best_f']}")
            break
    
    print(f"\nGlobal best particle: x = {global_best['x']}, y = {global_best['y']}, f(x,y) = {global_best['best_f']}")
    plot_contour(particles, global_best, i)    

if __name__ == '__main__':
    main()