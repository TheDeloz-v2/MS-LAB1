'''
Lab 1 - Modeling and Simulation
Date: 07/2024

Implementation of a particle swarm optimization algorithm
to find the minimum of a function.
'''

import random as r
import matplotlib.pyplot as plt
import numpy as np

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
Initialize the list of particles

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

'''
Function to plot particles
'''
def plot(particles, iteration):
    x = [p["x"] for p in particles]
    y = [p["y"] for p in particles]
    plt.scatter(x, y)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.title(f"Iteration {iteration}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

'''
Function to run the PSO algorithm

Parameters:
    w (float): inertia weight
    c1 (float): cognitive coefficient
    c2 (float): social coefficient
    n (int): number of particles (default is 40)
    iterations (int): number of iterations (default is 100)
    x_min (float): minimum value of x (default is -10)
    x_max (float): maximum value of x (default is 10)
    v_min (float): minimum value of the velocity (default is -1)
    v_max (float): maximum value of the velocity (default is 1)
    
Returns:
    tuple: A tuple containing the best particle (dict) and the number of iterations (int)
        - best particle (dict): The global best particle found with keys 'x', 'y', 'vx', 'vy', 'best_x', 'best_y', 'best_f'
        - iterations (int): The number of iterations taken to find the best particle
'''
def run_pso(w, c1, c2, n=40, iterations=100, x_min=-10, x_max=10, v_min=-1, v_max=1):
    particles = initialize_particles(n, x_min, x_max, v_min, v_max)
    global_best = particles[0]
    for i in range(1, n):
        if particles[i]["best_f"] < global_best["best_f"]:
            global_best = particles[i]

    for i in range(iterations):
        r.seed(777)
        
        for j in range(n):
            r1 = r.random()
            r2 = r.random()
            
            # Update particle velocity
            particles[j]["vx"] = w*particles[j]["vx"] + c1*r1*(particles[j]["best_x"] - particles[j]["x"]) + c2*r2*(global_best["best_x"] - particles[j]["x"])
            particles[j]["vy"] = w*particles[j]["vy"] + c1*r1*(particles[j]["best_y"] - particles[j]["y"]) + c2*r2*(global_best["best_y"] - particles[j]["y"])
            
            # Update particle position
            particles[j]["x"] = particles[j]["x"] + particles[j]["vx"]
            particles[j]["y"] = particles[j]["y"] + particles[j]["vy"]
            
            # Update particle best position
            if f(particles[j]["x"], particles[j]["y"]) < particles[j]["best_f"]:
                particles[j]["best_x"] = particles[j]["x"]
                particles[j]["best_y"] = particles[j]["y"]
                particles[j]["best_f"] = f(particles[j]["x"], particles[j]["y"])
            
            # Update global best position
            if particles[j]["best_f"] < global_best["best_f"]:
                global_best = particles[j]
            
        print("Iteration", i, ":", global_best["best_f"])
        
        if i == iterations // 2 or i == iterations - 1 or i == iterations // 4:
            plot(particles, i)
        
    return global_best, i

'''
Main function
'''
def main():
    # Sets of parameters to test
    parameter_sets = [
        {'w': 0.3, 'c1': 1.3, 'c2': 1.3},
        {'w': 0.7, 'c1': 1.7, 'c2': 1.7},
        {'w': 0.9, 'c1': 2.0, 'c2': 2.0}
    ]

    results = []

    for params in parameter_sets:
        print(f"Testing with parameters: w={params['w']}, c1={params['c1']}, c2={params['c2']}")
        result, iterations = run_pso(params['w'], params['c1'], params['c2'])
        results.append((params, result, iterations))

    # Print results
    for params, result, iterations in results:
        print(f"Parameters: w={params['w']}, c1={params['c1']}, c2={params['c2']}")
        print(f"Global best particle: {result}")
        print(f"Took {iterations} iterations")

    # Contour plot of the function
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    plt.contour(X, Y, Z, levels=50, cmap='jet')
    plt.title('Contour plot of the function')

    # Plot the global best positions found
    for params, result, _ in results:
        plt.plot(result['x'], result['y'], 'o', label=f"w={params['w']}, c1={params['c1']}, c2={params['c2']}")

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    main()
