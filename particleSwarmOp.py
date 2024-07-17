'''
Lab 1 - Modeling and Simulation
Date: 07/2024

Implementation of a particle swarm optimization algorithm
to find the minimum of a function.
'''

import random as r
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

def plot(particles):
    x = [p["x"] for p in particles]
    y = [p["y"] for p in particles]
    plt.scatter(x, y)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
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
    
    # Initialize particles
    particles = initialize_particles(n, x_min, x_max, v_min, v_max)

    plot(particles)
    
    # Global best particle
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
        
        if i == iterations/2 or i == iterations-1 or i == iterations/4:
            plot(particles)
        
    print("Global best particle:", global_best)
        

if __name__ == '__main__':
    main()