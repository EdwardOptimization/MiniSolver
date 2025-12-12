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

        # Special Constraints logic
        # Store tuples: (type, data_dict)
        self.special_constraints = []
        
        # Soft Constraints Meta Data
        # list of {index, type='L1'/'L2', weight}
        self.soft_constraints = []

        self.use_sparse_kernels = True

    def state(self, *names):
        symbols = []
        for name in names:
            s = sp.symbols(name)
            self.states.append(s)
            symbols.append(s)
        if len(symbols) == 1:
            return symbols[0]
        return tuple(symbols)

    def control(self, *names):
        symbols = []
        for name in names:
            u = sp.symbols(name)
            self.controls.append(u)
            symbols.append(u)
        if len(symbols) == 1:
            return symbols[0]
        return tuple(symbols)

    def parameter(self, *names):
        symbols = []
        for name in names:
            p = sp.symbols(name)
            self.parameters.append(p)
            symbols.append(p)
        if len(symbols) == 1:
            return symbols[0]
        return tuple(symbols)

    def set_dynamics(self, state, expr):
        """
        Set continuous dynamics: dot(state) = expr
        """
        if state not in self.states:
            raise ValueError(f"Unknown state: {state}")
        self.dynamics_rhs[state] = expr

    def minimize(self, *exprs):
        """
        Add term to Lagrange cost function
        """
        for expr in exprs:
            self.objective += expr

    def subject_to(self, *constraints, weight=None, loss='L2'):
        """
        Add inequality constraint.
        Accepts: 
        - expr <= 0
        - expr >= 0
        - expr (assumed <= 0)
        
        If weight > 0 is provided, marks the constraint as "Soft".
        This does NOT modify the objective function directly.
        Instead, it registers metadata so the Solver can apply Dual Regularization (L2) 
        or Dual Bounding (L1) during the solve phase.
        """
        for constraint in constraints:
            expr = constraint
            if isinstance(constraint, sp.LessThan): # lhs <= rhs -> lhs - rhs <= 0
                expr = constraint.lhs - constraint.rhs
            elif isinstance(constraint, sp.GreaterThan): # lhs >= rhs -> rhs - lhs <= 0
                expr = constraint.rhs - constraint.lhs
                
            # Add to constraints list
            self.constraints.append(expr)
            idx = len(self.constraints) - 1
            
            if weight is not None and weight > 0.0:
                self.soft_constraints.append({
                    'index': idx,
                    'type': loss,
                    'weight': weight
                })

    def subject_to_quad(self, Q, x, center=None, rhs=0.0, sense='<=', type='outside', linearize_at_boundary=False):
        """
        Add a quadratic constraint: (x-center)^T Q (x-center) {sense} rhs
        """
        # Helper to process Q
        if not isinstance(Q, sp.Matrix):
            Q_mat = sp.Matrix(Q)
        else:
            Q_mat = Q
            
        x_vec = sp.Matrix(x)
        
        if center is None:
            c_vec = sp.Matrix([0]*len(x))
        else:
            c_vec = sp.Matrix(center)
            
        if len(x) != Q_mat.shape[0] or Q_mat.shape[0] != Q_mat.shape[1]:
             raise ValueError("Dimension mismatch in Q and x")

        # Form the quadratic term: (x-c)^T Q (x-c)
        diff = x_vec - c_vec
        quad_term = (diff.T * Q_mat * diff)[0]
        
        # Logic for Robust Formulation
        is_exclusion = (sense == '>=' or type == 'outside')
        
        if is_exclusion:
            if linearize_at_boundary:
                xp_syms = [sp.symbols(f"xp_{len(self.special_constraints)}_{i}") for i in range(len(x))]
                xp_vec = sp.Matrix(xp_syms)
                
                # Gradient at boundary: 2 Q (xp - c)
                grad_at_boundary = 2 * Q_mat * (xp_vec - c_vec)
                
                # Linearized Constraint: - grad^T (x - xp) <= 0
                boundary_linear_expr = - (grad_at_boundary.T * (x_vec - xp_vec))[0]
                
                self.constraints.append(boundary_linear_expr)
                
                # Store info to generate the xp calculation code
                self.special_constraints.append({
                    'type': 'quad_boundary_proj',
                    'index': len(self.constraints) - 1,
                    'xp_syms': xp_syms,
                    'Q': Q_mat,
                    'c': c_vec,
                    'x': x_vec,
                    'rhs': rhs
                })
                
            else:
                # Reformulate: sqrt(quad_term + eps) >= sqrt(rhs)
                # -> sqrt(rhs) - sqrt(quad_term + eps) <= 0
                epsilon = 1e-6 # Tighter eps
                robust_dist = sp.sqrt(quad_term + epsilon)
                robust_rhs = sp.sqrt(rhs)
                self.constraints.append(robust_rhs - robust_dist)
        else:
            # Standard form
            if sense == '<=':
                self.constraints.append(quad_term - rhs)
            else:
                self.constraints.append(rhs - quad_term)

    def _generate_unpack_block(self, source_kp=True, expressions=None):
        code = ""
        
        # Determine used symbols
        used = None
        if expressions is not None:
            used = set()
            # Handle single expression or list/matrix
            if isinstance(expressions, (list, tuple)):
                for expr in expressions:
                    if hasattr(expr, 'free_symbols'):
                        used.update(expr.free_symbols)
            elif hasattr(expressions, 'free_symbols'):
                used.update(expressions.free_symbols)
            
        # Helper to check if we should unpack
        def is_used(sym):
            if used is None: return True
            return sym in used

        # Unpack State/Control/Params
        if source_kp:
            for i, s in enumerate(self.states):
                if is_used(s):
                    code += f"        T {s} = kp.x({i});\n"
            for i, u in enumerate(self.controls):
                if is_used(u):
                    code += f"        T {u} = kp.u({i});\n"
            for i, p in enumerate(self.parameters):
                if is_used(p):
                    code += f"        T {p} = kp.p({i});\n"
        else:
            # Source from function args x_in, u_in (and p_in)
            unpacked_x = False
            for i, s in enumerate(self.states):
                if is_used(s):
                    code += f"        T {s} = x_in({i});\n"
                    unpacked_x = True
            if not unpacked_x:
                code += "        (void)x_in;\n"

            unpacked_u = False
            for i, u in enumerate(self.controls):
                if is_used(u):
                    code += f"        T {u} = u_in({i});\n"
                    unpacked_u = True
            if not unpacked_u:
                code += "        (void)u_in;\n"

            # Add parameter unpacking for continuous dynamics
            unpacked_p = False
            for i, p in enumerate(self.parameters):
                if is_used(p):
                    code += f"        T {p} = p_in({i});\n"
                    unpacked_p = True
            if not unpacked_p:
                code += "        (void)p_in;\n"
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
                    
                    if sp.sympify(val).is_zero is True:
                        code += f"        kp.{name}({r},{c}) = 0;\n"
                    else:
                        code += f"        kp.{name}({r},{c}) = {sp.ccode(val)};\n"
        return code

    def _generate_assign_block_exact(self, assignments_obj, reduced, offset_obj, assignments_con, offset_con):
        """
        Special assignment block for Hessians that conditionally adds Constraint Hessians.
        """
        code = ""
        # We assume reduced vector contains [..., Q_obj, R_obj, H_obj, Q_con, R_con, H_con, ...]
        # Q_out = Q_obj + (Exact ? Q_con : 0)
        
        target_names = ["Q", "R", "H"]
        
        for i, name in enumerate(target_names):
            idx_obj = offset_obj + i
            idx_con = offset_con + i
            
            mat_obj = reduced[idx_obj]
            mat_con = reduced[idx_con]
            
            rows = mat_obj.shape[0]
            cols = mat_obj.shape[1]
            
            code += f"\n        // {name} (Conditionally Exact)\n"
            for r in range(rows):
                for c in range(cols):
                    val_obj = mat_obj[r, c]
                    val_con = mat_con[r, c]
                    
                    code_val_obj = sp.ccode(val_obj)
                    code_val_con = sp.ccode(val_con)
                    
                    # C++ logic: kp.Q(r,c) = Q_obj + (Exact ? Q_con : 0);
                    if val_con == 0:
                        code += f"        kp.{name}({r},{c}) = {code_val_obj};\n"
                    else:
                        code += f"        kp.{name}({r},{c}) = {code_val_obj};\n"
                        code += f"        if constexpr (Exact) kp.{name}({r},{c}) += {code_val_con};\n"
        return code

    def _generate_special_constraint_preamble(self):
        code = "\n        // --- Special Constraints Pre-Calculation ---\n"
        for info in self.special_constraints:
            if info['type'] == 'quad_boundary_proj':
                # Calculate Projection
                x_vec = info['x']
                c_vec = info['c']
                Q_mat = info['Q']
                diff = x_vec - c_vec
                d2_expr = (diff.T * Q_mat * diff)[0]
                
                rhs_val = info['rhs'] # Usually a number or symbol
                
                # Note: Declaring variables in outer scope to be visible to CSE expressions
                code += f"        T d2 = {sp.ccode(d2_expr)};\n"
                code += f"        T rhs = {sp.ccode(rhs_val)};\n"
                code += f"        T scale = sqrt(rhs / (d2 + 1e-9));\n"
                
                for i, xp_sym in enumerate(info['xp_syms']):
                    # xp_i = c_i + scale * (x_i - c_i)
                    val_expr = c_vec[i] + sp.symbols("scale") * (x_vec[i] - c_vec[i])
                    code += f"        T {xp_sym} = {sp.ccode(val_expr)};\n"
        return code

    def _generate_sparse_matmul(self, mat_symbol, non_zeros, input_name, output_name, R, C, common_dim):
        """
        Generates C++ code for: output = input * mat
        Exploits sparsity of mat (defined by non_zeros set).
        input: (common_dim x R)
        mat:   (R x C)
        output:(common_dim x C)
        """
        code = f"        {output_name}.setZero();\n"
        
        # Iterate over columns of the Right Matrix (Mat)
        for c in range(C):
            # Iterate over rows of the Right Matrix (Mat)
            for r in range(R):
                if (r, c) in non_zeros:
                    code += f"        // {mat_symbol}({r},{c}) contributes\n"
                    
                    # Fully unroll the inner loop (over rows of Input)
                    # For Riccati, common_dim is NX. 
                    # If NX <= 12, unrolling is safe and fast.
                    if common_dim <= 12:
                        for i in range(common_dim):
                            code += f"        {output_name}({i}, {c}) += {input_name}({i}, {r}) * kp.{mat_symbol}({r}, {c});\n"
                    else:
                         code += f"        for(int i=0; i<{common_dim}; ++i) {{\n"
                         code += f"            {output_name}(i, {c}) += {input_name}(i, {r}) * kp.{mat_symbol}({r}, {c});\n"
                         code += f"        }}\n"
        return code

    def _generate_fused_riccati_step(self, A_expr, B_expr, nx, nu):
        """
        Generates fused Riccati update kernel.
        Updates Qxx, Quu, Qux, qx, ru simultaneously.
        """
        print(f"Generating Fused Riccati Kernel (NX={nx}, NU={nu})...")
        
        # 1. Define Symbolic Variables
        
        # Vxx (Symmetric)
        Vxx = sp.zeros(nx, nx)
        for r in range(nx):
            for c in range(r, nx):
                s = sp.symbols(f"P_{r}_{c}")
                Vxx[r, c] = s
                Vxx[c, r] = s
        
        # Vx (Vector)
        Vx = sp.Matrix([sp.symbols(f"p_{r}") for r in range(nx)])
        
        # A (Sparse Structure)
        A_sym = sp.zeros(nx, nx)
        for r in range(nx):
            for c in range(nx):
                if sp.sympify(A_expr[r, c]).is_zero is not True:
                    A_sym[r, c] = sp.symbols(f"A_{r}_{c}")
                    
        # B (Sparse Structure)
        B_sym = sp.zeros(nx, nu)
        for r in range(nx):
            for c in range(nu):
                if sp.sympify(B_expr[r, c]).is_zero is not True:
                    B_sym[r, c] = sp.symbols(f"B_{r}_{c}")
                    
        # 2. Symbolic Computation
        # Use lazy evaluation, just building expression tree
        Update_Qxx = A_sym.T * Vxx * A_sym
        Update_Quu = B_sym.T * Vxx * B_sym
        Update_Qux = B_sym.T * Vxx * A_sym
        Update_qx  = A_sym.T * Vx
        Update_ru  = B_sym.T * Vx
        
        # 3. Extract expressions for all non-zeros for CSE
        # We only care about upper triangle for Hessian due to symmetry
        exprs_to_compute = []
        targets = [] # (matrix_name, row, col)
        
        # Qxx (Upper Tri)
        for r in range(nx):
            for c in range(r, nx):
                if sp.sympify(Update_Qxx[r, c]).is_zero is not True:
                    exprs_to_compute.append(Update_Qxx[r, c])
                    targets.append( ('Q_bar', r, c) )
                    
        # Quu (Upper Tri)
        for r in range(nu):
            for c in range(r, nu):
                if sp.sympify(Update_Quu[r, c]).is_zero is not True:
                    exprs_to_compute.append(Update_Quu[r, c])
                    targets.append( ('R_bar', r, c) )
                    
        # Qux (Full Matrix)
        for r in range(nu):
            for c in range(nx):
                if sp.sympify(Update_Qux[r, c]).is_zero is not True:
                    exprs_to_compute.append(Update_Qux[r, c])
                    targets.append( ('H_bar', r, c) )
                    
        # qx
        for r in range(nx):
            if sp.sympify(Update_qx[r]).is_zero is not True:
                exprs_to_compute.append(Update_qx[r])
                targets.append( ('q_bar', r, 0) )
                
        # ru
        for r in range(nu):
            if sp.sympify(Update_ru[r]).is_zero is not True:
                exprs_to_compute.append(Update_ru[r])
                targets.append( ('r_bar', r, 0) )
                
        # 4. CSE
        replacements, reduced_exprs = sp.cse(exprs_to_compute, symbols=sp.numbered_symbols("tmp_ric"))
        
        # 5. Generate C++ Code
        code = ""
        
        # 5.1 Load Data (Inputs)
        # Vxx
        for r in range(nx):
            for c in range(r, nx):
                code += f"        T P_{r}_{c} = Vxx({r},{c});\n"
        # Vx
        for r in range(nx):
            code += f"        T p_{r} = Vx({r});\n"
            
        # A (Non-zeros)
        for r in range(nx):
            for c in range(nx):
                if A_sym[r, c] != 0:
                    code += f"        T A_{r}_{c} = kp.A({r},{c});\n"
        
        # B (Non-zeros)
        for r in range(nx):
            for c in range(nu):
                if B_sym[r, c] != 0:
                    code += f"        T B_{r}_{c} = kp.B({r},{c});\n"
                    
        code += "\n        // CSE Intermediate Variables\n"
        for sym, expr in replacements:
            code += f"        T {sym} = {sp.ccode(expr)};\n"
            
        code += "\n        // Accumulate Results\n"
        for i, (name, r, c) in enumerate(targets):
            val = reduced_exprs[i]
            # Accumulate: kp.Q_bar += val
            code += f"        kp.{name}({r},{c}) += {sp.ccode(val)};\n"
            
        # 5.2 Fill Lower Triangles (Symmetry)
        code += "\n        // Fill Lower Triangles (Symmetry)\n"
        # Qxx
        for r in range(nx):
            for c in range(r + 1, nx):
                code += f"        kp.Q_bar({c},{r}) = kp.Q_bar({r},{c});\n"
        # Quu
        for r in range(nu):
            for c in range(r + 1, nu):
                code += f"        kp.R_bar({c},{r}) = kp.R_bar({r},{c});\n"
                
        return code

    def generate(self, output_dir="include/model", use_fused_riccati=True, integrator_type="RK4_EXPLICIT"):
        # 1. Vectorize
        x_vec = sp.Matrix(self.states)
        u_vec = sp.Matrix(self.controls)
        
        nx = len(self.states)
        nu = len(self.controls)
        np_param = len(self.parameters)
        nc = len(self.constraints)

        # Heuristic: Disable Fused Riccati if dimensions are very small, 
        # as the overhead of huge code generation might outweigh benefits,
        # or if it's too large (compilation time).
        # For now, default to True but warn if tiny.
        if use_fused_riccati and (nx <= 4):
             print(f"Note: Dimension NX={nx} is small. Fused Riccati might not be necessary, but enabling as requested.")
        
        self.use_fused_riccati = use_fused_riccati
        
        # 2. Dynamics (Continuous)
        dt_sym = sp.symbols('dt')
        
        # Continuous f(x,u)
        f_cont_list = []
        for s in self.states:
            f_cont_list.append(self.dynamics_rhs.get(s, 0))
        f_cont = sp.Matrix(f_cont_list)
        
        # Helper for variable substitution
        def get_f_subs(x_in, u_in):
            repl = {self.states[i]: x_in[i] for i in range(nx)}
            return f_cont.subs(repl)

        # --- Discrete Integrators (Explicit) ---
        integrators = {}
        
        # Euler Explicit
        x_next_euler = x_vec + dt_sym * f_cont
        integrators['EULER_EXPLICIT'] = x_next_euler
        
        # RK2 Explicit (Heun's / Midpoint)
        # k1 = f(x)
        # k2 = f(x + 0.5*dt*k1)
        # x_next = x + dt*k2
        k1_rk2 = f_cont
        k2_rk2 = get_f_subs(x_vec + 0.5*dt_sym*k1_rk2, u_vec)
        x_next_rk2 = x_vec + dt_sym * k2_rk2
        integrators['RK2_EXPLICIT'] = x_next_rk2

        # RK4 Explicit
        k1_rk4 = f_cont
        k2_rk4 = get_f_subs(x_vec + 0.5*dt_sym*k1_rk4, u_vec)
        k3_rk4 = get_f_subs(x_vec + 0.5*dt_sym*k2_rk4, u_vec)
        k4_rk4 = get_f_subs(x_vec + dt_sym*k3_rk4, u_vec)
        x_next_rk4 = x_vec + (dt_sym / 6.0) * (k1_rk4 + 2*k2_rk4 + 2*k3_rk4 + k4_rk4)
        integrators['RK4_EXPLICIT'] = x_next_rk4

        # Map Implicit to Explicit for Jacobian approximation (or generate same code)
        integrators['EULER_IMPLICIT'] = x_next_euler 
        integrators['RK2_IMPLICIT'] = x_next_rk2
        integrators['RK4_IMPLICIT'] = x_next_rk4

        # 3. Derivatives & CSE per Integrator
        code_compute_dyn_body = "        switch(type) {\n"
        
        groups = [
            (['EULER_EXPLICIT', 'EULER_IMPLICIT'], integrators['EULER_EXPLICIT']),
            (['RK2_EXPLICIT', 'RK2_IMPLICIT'], integrators['RK2_EXPLICIT']),
            (['RK4_EXPLICIT', 'RK4_IMPLICIT'], integrators['RK4_EXPLICIT'])
        ]
        
        # Track sparsity union across all integrators
        non_zeros_A = set()
        non_zeros_B = set()
        
        # Keep track of the most general expressions for fused kernel generation (usually RK4)
        target_A_expr = None
        target_B_expr = None
        
        for labels, x_next_expr in groups:
            print(f"Generating derivatives for {labels}...")
            
            A_expr = x_next_expr.jacobian(x_vec)
            B_expr = x_next_expr.jacobian(u_vec)
            
            if integrator_type in labels:
                target_A_expr = A_expr
                target_B_expr = B_expr
            
            # Analyze sparsity
            for r in range(nx):
                for c in range(nx):
                    if sp.sympify(A_expr[r, c]).is_zero is not True: 
                        non_zeros_A.add((r, c))
            for r in range(nx):
                for c in range(nu):
                    if sp.sympify(B_expr[r, c]).is_zero is not True: 
                        non_zeros_B.add((r, c))
            
            # CSE
            repl_dyn, reduced_dyn = sp.cse([x_next_expr, A_expr, B_expr], symbols=sp.numbered_symbols("tmp_d"))
            
            # Generate Case Statements
            for label in labels:
                code_compute_dyn_body += f"            case IntegratorType::{label}:\n"
            code_compute_dyn_body += "            {\n"
            
            # Body
            for name, val in repl_dyn:
                code_compute_dyn_body += f"                T {name} = {sp.ccode(val)};\n"
            
            # Assignments
            # f_resid (x_next)
            if nx > 0:
                mat = reduced_dyn[0]
                for r in range(nx):
                    val = mat[r]
                    code_compute_dyn_body += f"                kp.f_resid({r}) = {sp.ccode(val)};\n"
            
            # A
            if nx > 0:
                code_compute_dyn_body += f"                kp.A.setZero();\n"
                mat = reduced_dyn[1]
                for r in range(nx):
                    for c in range(nx):
                        val = mat[r,c]
                        if sp.sympify(val).is_zero is True:
                             pass # Already zero
                        else:
                             code_compute_dyn_body += f"                kp.A({r},{c}) = {sp.ccode(val)};\n"
            
            # B
            if nx > 0 and nu > 0:
                code_compute_dyn_body += f"                kp.B.setZero();\n"
                mat = reduced_dyn[2]
                for r in range(nx):
                    for c in range(nu):
                        val = mat[r,c]
                        if sp.sympify(val).is_zero is True:
                             pass # Already zero
                        else:
                             code_compute_dyn_body += f"                kp.B({r},{c}) = {sp.ccode(val)};\n"

            code_compute_dyn_body += "                break;\n"
            code_compute_dyn_body += "            }\n"

        code_compute_dyn_body += "        }"

        # 3.5 Derivatives for Cost & Constraints (Independent of Integrator)
        xu_vec = sp.Matrix.vstack(x_vec, u_vec)
        
        # Cost Derivatives
        grad_cost = sp.Matrix([self.objective]).jacobian(xu_vec).T
        q_grad = grad_cost[:nx, :]
        r_grad = grad_cost[nx:, :]
        
        # Hessian Cost
        hess_cost = sp.hessian(self.objective, xu_vec)
        Q_hess_obj = hess_cost[:nx, :nx]
        R_hess_obj = hess_cost[nx:, nx:]
        H_hess_obj = hess_cost[nx:, :nx]
        
        # Constraint Derivatives
        g_vec = sp.Matrix(self.constraints) if nc > 0 else sp.Matrix.zeros(0,1)
        C_expr = g_vec.jacobian(x_vec) if nc > 0 else sp.Matrix.zeros(0, nx)
        D_expr = g_vec.jacobian(u_vec) if nc > 0 else sp.Matrix.zeros(0, nu)
        
        # Hessian Constraints
        lam_sym = [sp.symbols(f"lam_{i}") for i in range(nc)]
        hess_con_total = sp.zeros(nx+nu, nx+nu)
        if nc > 0:
            for i in range(nc):
                hess_g_i = sp.hessian(self.constraints[i], xu_vec)
                hess_con_total += lam_sym[i] * hess_g_i
        
        Q_hess_con = hess_con_total[:nx, :nx]
        R_hess_con = hess_con_total[nx:, nx:]
        H_hess_con = hess_con_total[nx:, :nx]
        
        # CSE Constraints/Cost
        print("Optimizing expressions (CSE)...")
        repl_con, reduced_con = sp.cse([g_vec, C_expr, D_expr], symbols=sp.numbered_symbols("tmp_c"))
        repl_cost, reduced_cost = sp.cse(
            [q_grad, r_grad, Q_hess_obj, R_hess_obj, H_hess_obj, Q_hess_con, R_hess_con, H_hess_con, self.objective], 
            symbols=sp.numbered_symbols("tmp_j")
        )
        
        # [NEW] Gauss-Newton Hessian (Objective Only)
        # H_gn = J^T W J. Wait, self.objective might not be in LS form explicitly.
        # But commonly we assume H_obj is dominated by J^T J if residuals are small.
        # Acados assumes cost is explicitly y - y_ref.
        # For general objective, if it is convex, H_obj is PSD.
        # If we want GN, we need the user to provide LS terms? 
        # Or we just assume H_obj is good, but we DROP the constraint Hessian.
        # "Gauss-Newton" in SQP context usually means dropping the Constraint Hessian (Lagrangian term).
        # So H_approx = H_obj (assuming H_obj is PSD).
        # Let's generate a separate set of expressions where we don't add hess_con_total.
        
        # Actually, reduced_cost contains 8 matrices: q, r, Q_obj, R_obj, H_obj, Q_con, R_con, H_con.
        # We can just construct a separate compute_gn function that uses Q_obj but ignores Q_con!
        # The CSE `reduced_cost` already separates them.
        # So we just need to generate a new C++ function `compute_gn` that assigns Q = Q_obj (instead of Q_obj + Q_con).
        
        # 5. Code Construction
        
        # Constants
        code_constants = f"static const int NX={nx};\n    static const int NU={nu};\n    static const int NC={nc};\n    static const int NP={np_param};"
        
        # Generate Soft Constraints Meta-Data Array
        soft_weights_str = "{"
        soft_types_str = "{"
        weights_list = [0.0] * nc
        types_list = [0] * nc 
        for sc in self.soft_constraints:
            idx = sc['index']
            w = sc['weight']
            t = 2 if sc['type'] == 'L2' else 1
            if idx < nc:
                weights_list[idx] = w
                types_list[idx] = t
        soft_weights_str += ", ".join([str(w) for w in weights_list]) + "}"
        soft_types_str += ", ".join([str(t) for t in types_list]) + "}"
        code_soft = f"    static constexpr std::array<double, NC> constraint_weights = {soft_weights_str};\n"
        code_soft += f"    static constexpr std::array<int, NC> constraint_types = {soft_types_str};\n"
        code_constants += "\n\n" + code_soft
        
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
        code_dyn_cont = ""
        code_dyn_cont += self._generate_unpack_block(source_kp=False, expressions=f_cont)
        code_dyn_cont += "\n        MSVec<T, NX> xdot;\n"
        for i in range(nx):
            code_dyn_cont += f"        xdot({i}) = {sp.ccode(f_cont[i])};\n"
        code_dyn_cont += "        return xdot;\n"

        # Compute Dynamics Body
        # Note: Discrete dynamics depends on f_cont logic (and thus same symbols), 
        # plus potentially x itself (x_next = x + ...).
        # We pass f_cont and x_vec to be safe.
        dyn_exprs = [f_cont, x_vec]
        code_compute_dyn = ""
        code_compute_dyn += self._generate_unpack_block(source_kp=True, expressions=dyn_exprs)
        code_compute_dyn += "\n"
        code_compute_dyn += code_compute_dyn_body

        # Compute Constraints Body
        # Constraints depend on g_vec. 
        # Special constraints might use x directly (e.g. quad boundary).
        con_exprs = [g_vec]
        for info in self.special_constraints:
             con_exprs.append(info['x'])
             con_exprs.append(info['Q']) # Q might be symbolic?
             # c and rhs might be symbolic
        
        code_compute_con = ""
        code_compute_con += self._generate_unpack_block(source_kp=True, expressions=con_exprs)
        code_compute_con += self._generate_special_constraint_preamble()
        code_compute_con += "\n"
        for name, val in repl_con:
            code_compute_con += f"        T {name} = {sp.ccode(val)};\n"
        assign_con = [("g_val", 0, nc, 1), ("C", 1, nc, nx), ("D", 2, nc, nu)]
        code_compute_con += self._generate_assign_block(assign_con, reduced_con)

        # Compute Cost Body
        # Template param: 0 = Exact (with Con Hessian), 1 = GN (without Con Hessian)
        # Note: We also had 'Exact' bool before to toggle Con Hessian?
        # Let's redefine: compute_cost_impl<T, Mode>
        # Mode 0: Gauss-Newton (Q = Q_obj only)
        # Mode 1: Exact (Q = Q_obj + Q_con)
        
        code_cost_impl = "template<typename T, int Mode>\n"
        code_cost_impl += "    static void compute_cost_impl(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        
        # Determine symbols used in Cost + Constraint Hessians
        # We must use the *derivative* expressions, because parameters used only in linear terms
        # of the cost/constraints will disappear from the derivatives (gradients/Hessians),
        # but would still be flagged as "used" if we checked the primal expressions.
        cost_exprs = [q_grad, r_grad, Q_hess_obj, R_hess_obj, H_hess_obj, Q_hess_con, R_hess_con, H_hess_con, self.objective]
        
        code_unpack = self._generate_unpack_block(source_kp=True, expressions=cost_exprs)
        if nc > 0:
            # Check if lambdas are used in the Constraint Hessian expressions
            # H_con = sum(lam_i * H_g_i)
            # We can check if lam_i is in free_symbols of Q_hess_con, R_hess_con, H_hess_con
            
            # Combine all Hessian expressions for check
            hess_exprs = [Q_hess_con, R_hess_con, H_hess_con]
            used_syms = set()
            for expr in hess_exprs:
                if hasattr(expr, 'free_symbols'):
                    used_syms.update(expr.free_symbols)
            
            for i in range(nc):
                # We need to construct the symbol matching what sympy used
                # In step 3.5, we defined: lam_sym = [sp.symbols(f"lam_{i}") for i in range(nc)]
                # We should re-create or reuse that list. 
                # It's local to generate(). Ideally we should have stored it.
                # Re-creating with same name works in SymPy.
                s_lam = sp.symbols(f"lam_{i}")
                if s_lam in used_syms:
                    code_unpack += f"        T lam_{i} = kp.lam({i});\n"
        
        code_cse = "\n"
        for name, val in repl_cost:
            code_cse += f"        T {name} = {sp.ccode(val)};\n"
            
        code_assign = ""
        assign_grads = [("q", 0, nx, 1), ("r", 1, nu, 1)]
        code_assign += self._generate_assign_block(assign_grads, reduced_cost)
        
        # The key difference: generate_assign_block_exact handles the conditional addition
        # Mode 1 -> Exact. Mode 0 -> GN.
        # We need to adapt _generate_assign_block_exact to use the integer template.
        
        # Inline modified generation logic here
        target_names = ["Q", "R", "H"]
        offset_obj = 2
        offset_con = 5
        
        for i, name in enumerate(target_names):
            idx_obj = offset_obj + i
            idx_con = offset_con + i
            
            mat_obj = reduced_cost[idx_obj]
            mat_con = reduced_cost[idx_con]
            
            rows = mat_obj.shape[0]
            cols = mat_obj.shape[1]
            
            code_assign += f"\n        // {name} (Mode 0=GN, 1=Exact)\n"
            for r in range(rows):
                for c in range(cols):
                    val_obj = mat_obj[r, c]
                    val_con = mat_con[r, c]
                    
                    code_val_obj = sp.ccode(val_obj)
                    code_val_con = sp.ccode(val_con)
                    
                    # kp.Q = Q_obj
                    if val_con == 0:
                        code_assign += f"        kp.{name}({r},{c}) = {code_val_obj};\n"
                    else:
                        code_assign += f"        kp.{name}({r},{c}) = {code_val_obj};\n"
                        # Add constraint hessian only if Mode == 1
                        code_assign += f"        if constexpr (Mode == 1) kp.{name}({r},{c}) += {code_val_con};\n"
        
        if len(reduced_cost) > 8:
            code_assign += f"\n        kp.cost = {sp.ccode(reduced_cost[8])};\n"
            
        code_cost_impl += code_unpack + code_cse + code_assign
        code_cost_impl += "    }\n\n"
        
        # Wrappers
        # compute_cost -> GN (Default? No, existing code expects Exact usually? Or GN?)
        # Let's align with Acados philosophy: GN is preferred for control.
        # But strictly, compute_cost usually means evaluate everything.
        # Solver calls compute() which calls compute_cost().
        # We need to exposing a way to select.
        # The current C++ code in MiniSolver calls Model::compute_cost(kp).
        # We should make compute_cost use a template or runtime switch?
        # Runtime switch inside compute is nice.
        
        # New approach: compute_cost takes a flag? 
        # But signature is fixed in solver.
        
        # Let's provide explicit named functions.
        code_wrappers = "template<typename T>\n"
        code_wrappers += "    static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        code_wrappers += "        compute_cost_impl<T, 0>(kp);\n"
        code_wrappers += "    }\n\n"
        
        code_wrappers += "    template<typename T>\n"
        code_wrappers += "    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        code_wrappers += "        compute_cost_impl<T, 1>(kp);\n"
        code_wrappers += "    }\n\n"
        
        # Default alias (Exact for backward compat, or GN?)
        # Let's make it Exact to be safe.
        code_wrappers += "    template<typename T>\n"
        code_wrappers += "    static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        code_wrappers += "        compute_cost_impl<T, 1>(kp);\n"
        code_wrappers += "    }\n"
        
        code_compute_cost = code_cost_impl + code_wrappers
        
        # [NEW] Generate Fused Riccati Kernel
        code_fused_riccati = ""
        if self.use_fused_riccati:
            if target_A_expr is not None:
                try:
                    # Use the specified integrator expressions
                    code_fused_riccati = self._generate_fused_riccati_step(target_A_expr, target_B_expr, nx, nu)
                except Exception as e:
                    print(f"Warning: Failed to generate fused kernel: {e}")
            else:
                print(f"Warning: Integrator type '{integrator_type}' not found in generated groups. Fused Riccati Kernel will be empty.")

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
        content = content.replace("{{COMPUTE_COST_SECTION}}", code_compute_cost)
        
        if self.use_fused_riccati:
            content = content.replace("{{FUSED_RICCATI_STEP_BODY}}", code_fused_riccati)
            # Remove markers
            content = re.sub(r"// \[\[SPARSE_KERNELS_START\]\]", "", content)
            content = re.sub(r"// \[\[SPARSE_KERNELS_END\]\]", "", content)
        else:
            # Remove the whole section
            content = re.sub(r"// \[\[SPARSE_KERNELS_START\]\].*?// \[\[SPARSE_KERNELS_END\]\]", "", content, flags=re.DOTALL)

        # 7. Write Output
        file_name = "car_model.h" 
        if self.name != "CarModel":
            file_name = f"{self.name.lower()}.h"
        
        output_path = os.path.join(output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {output_path}...")
        with open(output_path, 'w') as f:
            f.write(content)
