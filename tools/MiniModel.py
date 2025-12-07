import sympy as sp
import os
import re

class OptimalControlModel:
    def __init__(self, name="Model"):
        self.name = name
        self.states = []
        self.controls = []
        self.parameters = []
        
        # Dynamics: map state_symbol -> expr
        self.dynamics_rhs = {}
        
        # Objective
        self.objective = 0.0
        
        # Constraints (g <= 0)
        self.constraints = []
        
        # Flags
        self.use_rk4 = True

    def state(self, name):
        s = sp.symbols(name)
        self.states.append(s)
        return s

    def control(self, name):
        u = sp.symbols(name)
        self.controls.append(u)
        return u

    def parameter(self, name):
        p = sp.symbols(name)
        self.parameters.append(p)
        return p

    def set_dynamics(self, state, expr):
        """
        Set continuous dynamics: dot(state) = expr
        """
        if state not in self.states:
            raise ValueError(f"Unknown state: {state}")
        self.dynamics_rhs[state] = expr

    def minimize(self, expr):
        """
        Add term to Lagrange cost function
        """
        self.objective += expr

    def subject_to(self, constraint):
        """
        Add inequality constraint.
        Accepts: 
        - expr <= 0
        - expr >= 0
        - expr (assumed <= 0)
        """
        expr = constraint
        if isinstance(constraint, sp.LessThan): # lhs <= rhs -> lhs - rhs <= 0
            expr = constraint.lhs - constraint.rhs
        elif isinstance(constraint, sp.GreaterThan): # lhs >= rhs -> rhs - lhs <= 0
            expr = constraint.rhs - constraint.lhs
            
        self.constraints.append(expr)

    def _generate_unpack_block(self, source_kp=True):
        code = ""
        # Unpack State/Control/Params
        if source_kp:
            for i, s in enumerate(self.states):
                code += f"        T {s} = kp.x({i});\n"
            for i, u in enumerate(self.controls):
                code += f"        T {u} = kp.u({i});\n"
            for i, p in enumerate(self.parameters):
                code += f"        T {p} = kp.p({i});\n"
        else:
            # Source from function args x_in, u_in (and p_in)
            for i, s in enumerate(self.states):
                code += f"        T {s} = x_in({i});\n"
            for i, u in enumerate(self.controls):
                code += f"        T {u} = u_in({i});\n"
            # Add parameter unpacking for continuous dynamics
            for i, p in enumerate(self.parameters):
                code += f"        T {p} = p_in({i});\n"
        return code

    def _generate_assign_block(self, assignments, reduced):
        code = ""
        for name, idx, rows, cols in assignments:
            if idx >= len(reduced): continue
            mat = reduced[idx]
            code += f"\n        // {name}\n"
            for r in range(rows):
                for c in range(cols):
                    if rows == 1 or cols == 1:
                        val = mat[r] if rows > 1 else mat[c]
                    else:
                        val = mat[r, c]
                    
                    if val == 0:
                        code += f"        kp.{name}({r},{c}) = 0;\n"
                    else:
                        code += f"        kp.{name}({r},{c}) = {sp.ccode(val)};\n"
        return code

    def generate(self, output_dir="include/model"):
        # 1. Vectorize
        x_vec = sp.Matrix(self.states)
        u_vec = sp.Matrix(self.controls)
        
        nx = len(self.states)
        nu = len(self.controls)
        np_param = len(self.parameters)
        nc = len(self.constraints)
        
        # 2. Dynamics (Continuous & Discrete)
        dt_sym = sp.symbols('dt')
        
        # Continuous f(x,u)
        f_cont_list = []
        for s in self.states:
            f_cont_list.append(self.dynamics_rhs.get(s, 0))
        f_cont = sp.Matrix(f_cont_list)
        
        # Discrete x_next (RK4)
        def get_f_subs(x_in, u_in):
            repl = {self.states[i]: x_in[i] for i in range(nx)}
            return f_cont.subs(repl)

        k1 = f_cont
        k2 = get_f_subs(x_vec + 0.5*dt_sym*k1, u_vec)
        k3 = get_f_subs(x_vec + 0.5*dt_sym*k2, u_vec)
        k4 = get_f_subs(x_vec + dt_sym*k3, u_vec)
        
        x_next = x_vec + (dt_sym / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 3. Derivatives
        print("Computing derivatives...")
        A_expr = x_next.jacobian(x_vec)
        B_expr = x_next.jacobian(u_vec)
        
        xu_vec = sp.Matrix.vstack(x_vec, u_vec)
        grad_cost = sp.Matrix([self.objective]).jacobian(xu_vec).T
        hess_cost = sp.hessian(self.objective, xu_vec)
        
        q_grad = grad_cost[:nx, :]
        r_grad = grad_cost[nx:, :]
        Q_hess = hess_cost[:nx, :nx]
        R_hess = hess_cost[nx:, nx:]
        H_hess = hess_cost[nx:, :nx]
        
        g_vec = sp.Matrix(self.constraints) if nc > 0 else sp.Matrix.zeros(0,1)
        C_expr = g_vec.jacobian(x_vec) if nc > 0 else sp.Matrix.zeros(0, nx)
        D_expr = g_vec.jacobian(u_vec) if nc > 0 else sp.Matrix.zeros(0, nu)
        
        # 4. CSE Optimization
        print("Optimizing expressions (CSE)...")
        repl_dyn, reduced_dyn = sp.cse([x_next, A_expr, B_expr], symbols=sp.numbered_symbols("tmp_d"))
        repl_con, reduced_con = sp.cse([g_vec, C_expr, D_expr], symbols=sp.numbered_symbols("tmp_c"))
        repl_cost, reduced_cost = sp.cse([q_grad, r_grad, Q_hess, R_hess, H_hess, self.objective], symbols=sp.numbered_symbols("tmp_j"))
        
        # 5. Code Construction
        
        # Constants
        code_constants = f"static const int NX={nx};\n    static const int NU={nu};\n    static const int NC={nc};\n    static const int NP={np_param};"
        
        # Name Arrays
        code_names = f"static constexpr std::array<const char*, NX> state_names = {{\n"
        for s in self.states: code_names += f'        "{s.name}",\n'
        code_names += "    };\n\n"
        
        code_names += f"    static constexpr std::array<const char*, NU> control_names = {{\n"
        for u in self.controls: code_names += f'        "{u.name}",\n'
        code_names += "    };\n\n"
        
        code_names += f"    static constexpr std::array<const char*, NP> param_names = {{\n"
        for p in self.parameters: code_names += f'        "{p.name}",\n'
        code_names += "    };\n"

        # Continuous Dynamics Body
        code_dyn_cont = self._generate_unpack_block(source_kp=False)
        code_dyn_cont += "\n        MSVec<T, NX> xdot;\n"
        for i in range(nx):
            code_dyn_cont += f"        xdot({i}) = {sp.ccode(f_cont[i])};\n"
        code_dyn_cont += "        return xdot;"

        # Compute Dynamics Body
        code_compute_dyn = self._generate_unpack_block(source_kp=True)
        code_compute_dyn += "\n"
        for name, val in repl_dyn:
            code_compute_dyn += f"        T {name} = {sp.ccode(val)};\n"
        assign_dyn = [("f_resid", 0, nx, 1), ("A", 1, nx, nx), ("B", 2, nx, nu)]
        code_compute_dyn += self._generate_assign_block(assign_dyn, reduced_dyn)

        # Compute Constraints Body
        code_compute_con = self._generate_unpack_block(source_kp=True)
        code_compute_con += "\n"
        for name, val in repl_con:
            code_compute_con += f"        T {name} = {sp.ccode(val)};\n"
        assign_con = [("g_val", 0, nc, 1), ("C", 1, nc, nx), ("D", 2, nc, nu)]
        code_compute_con += self._generate_assign_block(assign_con, reduced_con)

        # Compute Cost Body
        code_compute_cost = self._generate_unpack_block(source_kp=True)
        code_compute_cost += "\n"
        for name, val in repl_cost:
            code_compute_cost += f"        T {name} = {sp.ccode(val)};\n"
        assign_cost = [("q", 0, nx, 1), ("r", 1, nu, 1), ("Q", 2, nx, nx), ("R", 3, nu, nu), ("H", 4, nu, nx)]
        code_compute_cost += self._generate_assign_block(assign_cost, reduced_cost)
        # Scalar cost
        if len(reduced_cost) > 5:
            code_compute_cost += f"\n        kp.cost = {sp.ccode(reduced_cost[5])};\n"

        # 6. Read Template & Replace
        template_path = os.path.join(os.path.dirname(__file__), "templates", "model.h.in")
        with open(template_path, 'r') as f:
            template_content = f.read()
            
        content = template_content.replace("{{MODEL_NAME}}", self.name)
        content = content.replace("{{CONSTANTS}}", code_constants)
        content = content.replace("{{NAME_ARRAYS}}", code_names)
        content = content.replace("{{DYNAMICS_CONTINUOUS_BODY}}", code_dyn_cont)
        content = content.replace("{{COMPUTE_DYNAMICS_BODY}}", code_compute_dyn)
        content = content.replace("{{COMPUTE_CONSTRAINTS_BODY}}", code_compute_con)
        content = content.replace("{{COMPUTE_COST_BODY}}", code_compute_cost)

        # 7. Write Output
        file_name = "car_model.h" 
        if self.name != "CarModel":
            file_name = f"{self.name.lower()}.h"
        
        output_path = os.path.join(output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {output_path}...")
        with open(output_path, 'w') as f:
            f.write(content)
