import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_trajectory(csv_file):
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found. Run the C++ solver first.")
        return

    # Extract Parameters (Assumed constant for visualization)
    obs_x = data['obs_x'].iloc[0]
    obs_y = data['obs_y'].iloc[0]
    obs_rad = data['obs_rad'].iloc[0]
    car_rad = 1.0 # Matched with C++ model

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2)

    # 1. Trajectory (XY)
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(data['x'], data['y'], 'b.-', label='Car Path')
    
    # Draw Obstacle
    obs_circle = plt.Circle((obs_x, obs_y), obs_rad, color='r', alpha=0.5, label='Obstacle')
    safe_circle = plt.Circle((obs_x, obs_y), obs_rad + car_rad, color='r', fill=False, linestyle='--', label='Safety Boundary')
    
    ax_traj.add_patch(obs_circle)
    ax_traj.add_patch(safe_circle)
    
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('X [m]')
    ax_traj.set_ylabel('Y [m]')
    ax_traj.set_title('Vehicle Trajectory')
    ax_traj.legend()
    ax_traj.grid(True)

    # 2. Velocity & Theta
    ax_state = fig.add_subplot(gs[0, 1])
    ax_state.plot(data['t'], data['v'], 'g-', label='Velocity [m/s]')
    ax_state.plot(data['t'], data['theta'], 'k--', label='Theta [rad]')
    ax_state.set_ylabel('State')
    ax_state.set_title('State Profiles')
    ax_state.legend()
    ax_state.grid(True)

    # 3. Controls
    ax_ctrl = fig.add_subplot(gs[1, 1])
    ax_ctrl.step(data['t'], data['acc'], 'm-', where='post', label='Accel [m/s^2]')
    ax_ctrl.step(data['t'], data['steer'], 'c-', where='post', label='Steer [rad]')
    
    # Control Limits
    ax_ctrl.axhline(y=3.0, color='r', linestyle=':', alpha=0.3)
    ax_ctrl.axhline(y=-3.0, color='r', linestyle=':', alpha=0.3)
    ax_ctrl.axhline(y=0.5, color='r', linestyle=':', alpha=0.3)
    ax_ctrl.axhline(y=-0.5, color='r', linestyle=':', alpha=0.3)
    
    ax_ctrl.set_ylabel('Control')
    ax_ctrl.set_title('Control Inputs')
    ax_ctrl.legend()
    ax_ctrl.grid(True)

    # 4. Obstacle Constraint Value
    ax_cons = fig.add_subplot(gs[2, 1])
    # Constraint: (R+r)^2 - dist^2 <= 0  => dist^2 >= (R+r)^2
    # The solver outputs g_val. If g_val <= 0, we are safe.
    ax_cons.plot(data['t'], data['g_obs'], 'r-', label='Obstacle Constraint (g)')
    ax_cons.axhline(y=0, color='k', linestyle='-', linewidth=2)
    ax_cons.fill_between(data['t'], 0, np.maximum(0, data['g_obs']), color='r', alpha=0.3, label='Violation')
    
    ax_cons.set_xlabel('Time [s]')
    ax_cons.set_ylabel('Value (<= 0 is Safe)')
    ax_cons.set_title('Constraint Satisfaction')
    ax_cons.legend()
    ax_cons.grid(True)

    plt.tight_layout()
    plt.savefig('trajectory_plot.png')
    print("Plot saved to trajectory_plot.png")
    # plt.show() # Uncomment if running locally with display

if __name__ == "__main__":
    file_path = "trajectory.csv"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    plot_trajectory(file_path)

