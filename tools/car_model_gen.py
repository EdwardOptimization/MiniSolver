import sympy as sp
from MiniModel import OptimalControlModel

if __name__ == "__main__":
    model = OptimalControlModel("CarModel")
    
    # 1. Variables
    x = model.state("x")
    y = model.state("y")
    theta = model.state("theta")
    v = model.state("v")
    
    acc = model.control("acc")
    steer = model.control("steer")
    
    # 2. Parameters
    v_ref = model.parameter("v_ref")
    x_ref = model.parameter("x_ref")
    y_ref = model.parameter("y_ref")
    obs_x = model.parameter("obs_x")
    obs_y = model.parameter("obs_y")
    obs_rad = model.parameter("obs_rad")
    
    # 3. Dynamics
    L = 2.5
    model.set_dynamics(x, v * sp.cos(theta))
    model.set_dynamics(y, v * sp.sin(theta))
    model.set_dynamics(theta, (v / L) * sp.tan(steer))
    model.set_dynamics(v, acc)
    
    # 4. Objectives
    model.minimize( 1.0 * (x - x_ref)**2 )
    model.minimize( 1.0 * (y - y_ref)**2 )
    model.minimize( 0.1 * theta**2 )
    model.minimize( 1.0 * (v - v_ref)**2 )
    model.minimize( 0.1 * acc**2 )
    model.minimize( 1.0 * steer**2 )
    
    # 5. Constraints
    model.subject_to(acc <= 3.0)
    model.subject_to(acc >= -3.0)
    model.subject_to(steer <= 0.5)
    model.subject_to(steer >= -0.5)
    
    # Obstacle (Manual Optimization for Robustness)
    eps = 1e-6
    dist = sp.sqrt((x - obs_x)**2 + (y - obs_y)**2 + eps)
    car_rad = 1.0
    model.subject_to( (obs_rad + car_rad) - dist <= 0 )
    
    # 6. Generate
    model.generate("include/model")
    print("Generated CarModel using DSL.")

