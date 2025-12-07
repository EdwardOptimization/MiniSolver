import sympy as sp
from MiniModel import OptimalControlModel

if __name__ == "__main__":
    model = OptimalControlModel("CarModel")
    
    # 1. Variables
    x, y, theta, v = model.state("x", "y", "theta", "v")
    
    acc, steer = model.control("acc", "steer")
    
    # 2. Parameters
    v_ref = model.parameter("v_ref")
    x_ref = model.parameter("x_ref")
    y_ref = model.parameter("y_ref")
    obs_x = model.parameter("obs_x")
    obs_y = model.parameter("obs_y")
    obs_rad = model.parameter("obs_rad")
    
    # Physical Constants as Parameters
    L = model.parameter("L")
    car_rad = model.parameter("car_rad")
    
    # Weights as Parameters
    w_pos = model.parameter("w_pos")
    w_vel = model.parameter("w_vel")
    w_theta = model.parameter("w_theta")
    w_acc = model.parameter("w_acc")
    w_steer = model.parameter("w_steer")
    
    # 3. Dynamics
    model.set_dynamics(x, v * sp.cos(theta))
    model.set_dynamics(y, v * sp.sin(theta))
    model.set_dynamics(theta, (v / L) * sp.tan(steer))
    model.set_dynamics(v, acc)
    
    # 4. Objectives
    model.minimize( w_pos * (x - x_ref)**2 )
    model.minimize( w_pos * (y - y_ref)**2 )
    model.minimize( w_theta * theta**2 )
    model.minimize( w_vel * (v - v_ref)**2 )
    model.minimize( w_acc * acc**2 )
    model.minimize( w_steer * steer**2 )
    
    # 5. Constraints
    model.subject_to(acc <= 3.0)
    model.subject_to(acc >= -3.0)
    model.subject_to(steer <= 0.5)
    model.subject_to(steer >= -0.5)
    
    # Obstacle - Use robust boundary linearization
    # Quadratic form: (x-obs_x)^2 + (y-obs_y)^2
    # Q is identity for x,y.
    Q_obs = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) # 4x4 for [x, y, theta, v]
    center = [obs_x, obs_y, 0, 0]
    total_rad = obs_rad + car_rad
    rhs = total_rad**2
    
    # Use the new robust mode
    model.subject_to_quad(Q_obs, [x,y,theta,v], center=center, rhs=rhs, type='outside', linearize_at_boundary=True)
    
    # 6. Generate
    model.generate("include/model")
    print("Generated CarModel using DSL.")
