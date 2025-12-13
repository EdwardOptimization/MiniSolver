#!/usr/bin/env python3
"""
Generate a challenging model for line search benchmarking based on the Rosenbrock function.
The Rosenbrock function (banana function) is a classic test problem for optimization algorithms.
It has a narrow, curved valley which makes it difficult to converge and heavily reliant on line search.
"""

import sys
import os

# Add path to MiniModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

from minisolver.MiniModel import OptimalControlModel
import sympy as sp

if __name__ == "__main__":
    print("DEBUG: Running the NEW generate_model.py script")
    model = OptimalControlModel("LineSearchBenchmarkModel")

    # States: position (x, y)
    x, y = model.state("x", "y")
    
    # Controls: velocity (vx, vy)
    vx, vy = model.control("vx", "vy")

    # Parameters for Rosenbrock function
    # f(x,y) = (a - x)^2 + b * (y - x^2)^2
    # Standard values are a=1, b=100
    a_param = model.parameter("a_param")
    b_param = model.parameter("b_param")
    
    # Weights
    w_u = model.parameter("w_u") # Regularization for controls

    # Dynamics (Simple Kinematics)
    # x_dot = vx
    # y_dot = vy
    model.set_dynamics(x, vx)
    model.set_dynamics(y, vy)

    # Rosenbrock Cost
    # The solver will try to follow the valley y = x^2 towards (a, a^2)
    rosenbrock_cost = (a_param - x)**2 + b_param * (y - x**2)**2
    
    # Control regularization
    control_cost = w_u * (vx**2 + vy**2)

    model.minimize(rosenbrock_cost + control_cost)

    # Constraints
    # Add some bounds to make it a constrained problem
    model.subject_to(vx <= 5.0)
    model.subject_to(vx >= -5.0)
    model.subject_to(vy <= 5.0)
    model.subject_to(vy >= -5.0)

    # Add a non-convex obstacle to make line search even more critical
    # Circular obstacle at (0, 0.5) with radius 0.2
    # (x - 0)^2 + (y - 0.5)^2 >= 0.2^2
    # obs_x = model.parameter("obs_x")
    # obs_y = model.parameter("obs_y")
    # obs_r = model.parameter("obs_r")
    # model.subject_to((x - obs_x)**2 + (y - obs_y)**2 >= obs_r**2)

    # Generate the model
    output_dir = os.path.join(os.path.dirname(__file__), "generated")
    print(f"DEBUG: Output directory absolute path: {os.path.abspath(output_dir)}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model.generate(output_dir)
    print(f"Generated LineSearchBenchmarkModel in {output_dir}")
