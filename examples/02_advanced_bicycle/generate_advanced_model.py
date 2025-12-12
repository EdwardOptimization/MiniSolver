import sys
import os

# Add path to MiniModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

from minisolver.MiniModel import OptimalControlModel
import sympy as sp

# -----------------------------------------------------------
# Extended Kinematic Bicycle Model (6 States)
# States:
#   x, y: Position
#   theta: Heading
#   kappa: Curvature (steering angle / wheelbase approx)
#   v: Velocity
#   a: Acceleration
#
# Controls:
#   dkappa: Rate of change of curvature (steering rate)
#   jerk: Rate of change of acceleration
# -----------------------------------------------------------

import argparse

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--no-fused', action='store_true', help='Disable Fused Riccati Kernel')
args = parser.parse_args()

use_fused = not args.no_fused
print(f"Generating Advanced Bicycle Model (Fused Riccati: {use_fused})...")

model = OptimalControlModel(name="BicycleExtModel")

# 1. Define Variables
x, y, theta, kappa, v, a = model.state("x", "y", "theta", "kappa", "v", "a")

dkappa, jerk = model.control("dkappa", "jerk")

# 2. Define Parameters
v_ref = model.parameter("v_ref")
x_ref = model.parameter("x_ref")
y_ref = model.parameter("y_ref")

# Obstacle
obs_x = model.parameter("obs_x")
obs_y = model.parameter("obs_y")
obs_rad = model.parameter("obs_rad")

# Vehicle Params
L = model.parameter("L") # Wheelbase is effectively integrated into kappa logic if we treat kappa = tan(delta)/L
# Actually for this model, x_dot = v * cos(theta), theta_dot = v * kappa. 
# So kappa is path curvature.

car_rad = model.parameter("car_rad")

# Weights
w_pos = model.parameter("w_pos")
w_vel = model.parameter("w_vel")
w_theta = model.parameter("w_theta")
w_kappa = model.parameter("w_kappa")
w_a = model.parameter("w_a")
w_dkappa = model.parameter("w_dkappa")
w_jerk = model.parameter("w_jerk")

# 3. Dynamics
# x_dot = v * cos(theta)
# y_dot = v * sin(theta)
# theta_dot = v * kappa
# kappa_dot = dkappa
# v_dot = a
# a_dot = jerk

model.set_dynamics(x, v * sp.cos(theta))
model.set_dynamics(y, v * sp.sin(theta))
model.set_dynamics(theta, v * kappa)
model.set_dynamics(kappa, dkappa)
model.set_dynamics(v, a)
model.set_dynamics(a, jerk)

# 4. Objective
# Track reference
cost_track = w_pos*((x - x_ref)**2 + (y - y_ref)**2) + \
             w_vel*(v - v_ref)**2 + \
             w_theta*(theta**2)

# Regularize states and controls
cost_reg = w_kappa*(kappa**2) + \
           w_a*(a**2) + \
           w_dkappa*(dkappa**2) + \
           w_jerk*(jerk**2)

model.minimize(cost_track + cost_reg)

# 5. Constraints
# State limits
model.subject_to(v - 15.0 <= 0)  # Max velocity
model.subject_to(-v + 0.0 <= 0)  # Min velocity (non-negative)
model.subject_to(a - 5.0 <= 0)   # Max accel
model.subject_to(-a - 5.0 <= 0)  # Min accel
model.subject_to(kappa - 0.5 <= 0) # Max curvature (approx turning radius 2m)
model.subject_to(-kappa - 0.5 <= 0)

# Control limits (Restored but relaxed)
model.subject_to(jerk - 50.0 <= 0)
model.subject_to(-jerk - 50.0 <= 0)
model.subject_to(dkappa - 2.0 <= 0)
model.subject_to(-dkappa - 2.0 <= 0)

# # Obstacle Avoidance (Restored)
# model.subject_to_quad(sp.eye(2), [x, y], center=[obs_x, obs_y], rhs=(car_rad + obs_rad)**2, sense='>=')

# 6. Generate
output_dir = os.path.join(os.path.dirname(__file__), "generated")
model.generate(output_dir, use_fused_riccati=use_fused)

