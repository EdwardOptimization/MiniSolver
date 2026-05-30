import sympy as sp
import hashlib
import os
import re

CPP_KEYWORDS = {
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool",
    "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t", "class",
    "compl", "concept", "const", "consteval", "constexpr", "constinit", "const_cast",
    "continue", "co_await", "co_return", "co_yield", "decltype", "default", "delete",
    "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern",
    "false", "float", "for", "friend", "goto", "if", "inline", "int", "long",
    "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator",
    "or", "or_eq", "private", "protected", "public", "register", "reinterpret_cast",
    "requires", "return", "short", "signed", "sizeof", "static", "static_assert",
    "static_cast", "struct", "switch", "template", "this", "thread_local", "throw",
    "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
    "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"
}

CPP_GENERATED_RESERVED_NAMES = {
    "kp", "dt", "type", "x_in", "u_in", "p_in", "xdot", "jac",
    "current_dt", "scale", "rhs", "d2", "diff", "quad_term", "robust_dist",
    "robust_rhs", "x_next", "k1", "k2", "k3", "k4", "tmp", "res",
}

CPP_GENERATED_RESERVED_PREFIXES = (
    "tmp_", "tmp_d", "tmp_c", "tmp_j", "tmp_jc", "lam_", "P_", "A_", "B_", "xp_",
    "d2_", "rhs_", "scale_",
)


class DynamicsEquation:
    def __init__(self, mode, state, rhs):
        self.mode = mode
        self.state = state
        self.rhs = rhs


class DynamicsRef:
    def __init__(self, mode, state):
        self.mode = mode
        self.state = state

    def __eq__(self, rhs):
        return DynamicsEquation(self.mode, self.state, rhs)

    def __repr__(self):
        return f"{self.mode}({self.state})"


def Dot(state):
    return DynamicsRef("dot", state)


def Next(state):
    return DynamicsRef("next", state)


class OptimalControlModel:
    def __init__(self, name="Model"):
        self.states = []
        self.controls = []
        self.parameters = []
        self._validate_cpp_identifier("model", name)
        self.name = name
        
        # Dynamics: Dot(state) for continuous ODEs or Next(state) for direct
        # discrete maps. The two modes intentionally cannot be mixed.
        self.dynamics_mode = None
        self.dynamics_rhs = {}
        self.next_state_rhs = {}
        
        # Objective
        self.objective = 0.0
        self.residuals = []
        self.residual_weights = []
        
        # Constraints (g <= 0)
        self.constraints = []
        self.true_constraints = []
        self.constraint_include_terminal = []
        
        # Flags
        self.use_rk4 = True

        # Special Constraints logic
        # Store tuples: (type, data_dict)
        self.special_constraints = []
        
        # Soft Constraints Meta Data
        # list of {index, type='L1'/'L2', weight}. The row/type structure is
        # fixed by Python modeling; generated C++ only refreshes runtime weights.
        self.soft_constraints = []

        self.use_sparse_kernels = True

    def _all_variable_names(self):
        return [s.name for s in self.states] + [u.name for u in self.controls] + [
            p.name for p in self.parameters
        ]

    def _declared_symbols(self):
        return set(self.states) | set(self.controls) | set(self.parameters)

    def _validate_declared_symbols(self, label, *values):
        unknown = []
        declared = self._declared_symbols()
        for value in values:
            if value is None:
                continue
            for expr in self._flatten_expr_list(value):
                unknown.extend(sorted(expr.free_symbols - declared, key=lambda sym: sym.name))
        if unknown:
            names = ", ".join(sym.name for sym in unknown)
            raise ValueError(f"{label} references undeclared symbols: {names}")

    def _validate_cpp_identifier(self, kind, name):
        if not isinstance(name, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ValueError(f"{kind} name {name!r} is not a valid C++ identifier")
        if name in CPP_KEYWORDS:
            raise ValueError(f"{kind} name {name!r} is a reserved C++ keyword")
        if name in CPP_GENERATED_RESERVED_NAMES or any(
            name.startswith(prefix) for prefix in CPP_GENERATED_RESERVED_PREFIXES
        ):
            raise ValueError(f"{kind} name {name!r} is reserved by MiniSolver codegen")
        if name in self._all_variable_names():
            raise ValueError(f"duplicate variable name {name!r}; choose unique state/control/parameter names")

    def state(self, *names):
        symbols = []
        for name in names:
            self._validate_cpp_identifier("state", name)
            s = sp.symbols(name)
            self.states.append(s)
            symbols.append(s)
        if len(symbols) == 1:
            return symbols[0]
        return tuple(symbols)

    def control(self, *names):
        symbols = []
        for name in names:
            self._validate_cpp_identifier("control", name)
            u = sp.symbols(name)
            self.controls.append(u)
            symbols.append(u)
        if len(symbols) == 1:
            return symbols[0]
        return tuple(symbols)

    def parameter(self, *names):
        symbols = []
        for name in names:
            self._validate_cpp_identifier("parameter", name)
            p = sp.symbols(name)
            self.parameters.append(p)
            symbols.append(p)
        if len(symbols) == 1:
            return symbols[0]
        return tuple(symbols)

    def _set_dynamics_equation(self, equation):
        if equation.state not in self.states:
            raise ValueError(f"Unknown state: {equation.state}")
        if self.dynamics_mode is None:
            self.dynamics_mode = equation.mode
        elif self.dynamics_mode != equation.mode:
            raise ValueError("Cannot mix Dot and Next dynamics in one model")

        expr = sp.sympify(equation.rhs)
        self._validate_declared_symbols("dynamics", expr)

        rhs_map = self.dynamics_rhs if equation.mode == "dot" else self.next_state_rhs
        if equation.state in rhs_map:
            raise ValueError(f"Dynamics already specified for state: {equation.state}")
        rhs_map[equation.state] = expr

    def _is_numeric_expr(self, expr):
        return len(sp.sympify(expr).free_symbols) == 0

    def _validate_numeric_positive(self, label, expr):
        if self._is_numeric_expr(expr) and float(sp.N(expr)) <= 0.0:
            raise ValueError(f"{label} must be positive for outside quadratic constraints")

    def _validate_numeric_psd(self, Q_mat):
        entries = list(Q_mat)
        if any(not self._is_numeric_expr(entry) for entry in entries):
            return
        if Q_mat != Q_mat.T:
            raise ValueError("Q must be symmetric PSD for quadratic constraints")
        evals = Q_mat.eigenvals()
        for eig in evals:
            if float(sp.N(eig)) < -1e-12:
                raise ValueError("Q must be PSD for quadratic constraints")

    def minimize(self, *exprs):
        """
        Add term to Lagrange cost function
        """
        for expr in exprs:
            expr = sp.sympify(expr)
            self._validate_declared_symbols("objective", expr)
            self.objective += expr

    def _flatten_expr_list(self, value):
        if isinstance(value, sp.MatrixBase):
            return [sp.sympify(item) for item in value]
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                result.extend(self._flatten_expr_list(item))
            return result
        return [sp.sympify(value)]

    def _is_sequence_like(self, value):
        return isinstance(value, (list, tuple, sp.MatrixBase))

    def _validate_residual_weight(self, weight):
        weight_expr = sp.sympify(weight)
        self._validate_declared_symbols("residual weight", weight_expr)
        if self._is_numeric_expr(weight_expr) and float(sp.N(weight_expr)) < 0.0:
            raise ValueError("residual weight must be non-negative")
        decision_symbols = set(self.states) | set(self.controls)
        if weight_expr.free_symbols.intersection(decision_symbols):
            raise ValueError("residual weight must not depend on state or control variables")
        return weight_expr

    def _validate_soft_constraint_weight(self, weight):
        weight_expr = sp.sympify(weight)
        self._validate_declared_symbols("soft constraint weight", weight_expr)
        if self._is_numeric_expr(weight_expr) and float(sp.N(weight_expr)) < 0.0:
            raise ValueError("soft constraint weight must be non-negative")
        decision_symbols = set(self.states) | set(self.controls)
        if weight_expr.free_symbols.intersection(decision_symbols):
            raise ValueError("soft constraint weight must not depend on state or control variables")
        return weight_expr

    def add_residual(self, residual, weight=1.0):
        """
        Add least-squares residual cost terms:
            0.5 * weight_i * residual_i^2

        `residual` may be a scalar, list/tuple, or SymPy Matrix. `weight` may be
        a scalar broadcast to every residual or a same-length diagonal weight
        list/tuple/Matrix. Dense weight matrices are intentionally not supported.
        """
        residuals = self._flatten_expr_list(residual)
        if not residuals:
            raise ValueError("add_residual requires at least one residual")

        weight_is_sequence = self._is_sequence_like(weight)
        weights = self._flatten_expr_list(weight)
        if not weight_is_sequence and len(weights) == 1:
            weights = weights * len(residuals)
        elif len(weights) != len(residuals):
            raise ValueError("residual weight length must match residual length")

        for residual_expr, weight_expr in zip(residuals, weights):
            self._validate_declared_symbols("residual", residual_expr)
            self.residuals.append(sp.sympify(residual_expr))
            self.residual_weights.append(self._validate_residual_weight(weight_expr))

    def _least_squares_objective(self, substitutions=None):
        total = sp.sympify(0)
        for residual, weight in zip(self.residuals, self.residual_weights):
            r_expr = residual
            w_expr = weight
            if substitutions:
                r_expr = sp.simplify(r_expr.subs(substitutions))
                w_expr = sp.simplify(w_expr.subs(substitutions))
            total += sp.Rational(1, 2) * w_expr * r_expr**2
        return sp.simplify(total)

    def _total_objective(self, substitutions=None):
        general = sp.sympify(self.objective)
        if substitutions:
            general = sp.simplify(general.subs(substitutions))
        return sp.simplify(general + self._least_squares_objective(substitutions))

    def _fingerprint_token(self, value):
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
            return repr(value)
        if isinstance(value, sp.MatrixBase):
            rows = [
                ",".join(self._fingerprint_token(value[row, col]) for col in range(value.cols))
                for row in range(value.rows)
            ]
            return f"Matrix({value.rows},{value.cols})[" + ";".join(rows) + "]"
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(self._fingerprint_token(item) for item in value) + "]"
        if isinstance(value, dict):
            items = []
            for key in sorted(value.keys()):
                items.append(f"{self._fingerprint_token(key)}:{self._fingerprint_token(value[key])}")
            return "{" + ",".join(items) + "}"
        if isinstance(value, sp.Basic):
            return sp.srepr(value)
        return repr(value)

    def _compute_model_fingerprint(self, integrator_type):
        parts = []

        def add(label, value):
            parts.append(f"{label}={self._fingerprint_token(value)}")

        add("schema", "MiniModelFingerprintV1")
        add("model", self.name)
        add("integrator", integrator_type)
        add("use_fused_riccati", self.use_fused_riccati)
        add("states", [s.name for s in self.states])
        add("controls", [u.name for u in self.controls])
        add("parameters", [p.name for p in self.parameters])
        add("dynamics_mode", self.dynamics_mode)
        for state in self.states:
            add(f"dot:{state.name}", self.dynamics_rhs.get(state, 0))
            add(f"next:{state.name}", self.next_state_rhs.get(state, 0))
        add("objective", self.objective)
        add("residuals", self.residuals)
        add("residual_weights", self.residual_weights)
        add("constraints", self.constraints)
        add("true_constraints", self.true_constraints)
        add("constraint_include_terminal", self.constraint_include_terminal)
        add("soft_constraints", self.soft_constraints)
        add("special_constraints", self.special_constraints)

        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _residual_gauss_newton_hessian(self, xu_vec, substitutions=None):
        dim = xu_vec.rows
        hess = sp.zeros(dim, dim)
        for residual, weight in zip(self.residuals, self.residual_weights):
            r_expr = residual
            w_expr = weight
            if substitutions:
                r_expr = sp.simplify(r_expr.subs(substitutions))
                w_expr = sp.simplify(w_expr.subs(substitutions))
            jac = sp.Matrix([r_expr]).jacobian(xu_vec)
            hess += w_expr * (jac.T * jac)
        return sp.simplify(hess)

    def _flatten_constraint_args(self, value):
        if isinstance(value, DynamicsEquation):
            return [value]
        if isinstance(value, (sp.LessThan, sp.GreaterThan, sp.Equality)):
            return [value]
        if isinstance(value, sp.MatrixBase):
            return [item for item in value]
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                result.extend(self._flatten_constraint_args(item))
            return result
        return [value]

    def _sequence_items(self, value):
        if isinstance(value, sp.MatrixBase):
            return [item for item in value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return None

    def _normalize_row_soft_entries(self, row_index, weight, loss):
        if weight is None:
            return []

        weight_items = self._sequence_items(weight)
        loss_items = self._sequence_items(loss)

        if loss_items is None:
            if weight_items is not None:
                if len(weight_items) != 1:
                    raise ValueError("soft constraint weight/loss lengths must match")
                weight_items = [weight_items[0]]
            else:
                weight_items = [weight]
            loss_items = [loss]
        else:
            if weight_items is None:
                weight_items = [weight] * len(loss_items)
            if len(weight_items) != len(loss_items):
                raise ValueError("soft constraint weight/loss lengths must match")

        entries = []
        seen_losses = set()
        for weight_item, loss_item in zip(weight_items, loss_items):
            if loss_item not in ("L1", "L2"):
                raise ValueError("soft constraint loss must be L1 or L2")
            if loss_item in seen_losses:
                raise ValueError("duplicate soft constraint loss for one row")
            seen_losses.add(loss_item)

            weight_expr = self._validate_soft_constraint_weight(weight_item)
            entries.append({
                'index': row_index,
                'type': loss_item,
                'weight': weight_expr
            })
        return entries

    def _row_soft_specs(self, row_count, weight, loss):
        if weight is None:
            return [(None, None)] * row_count

        weight_items = self._sequence_items(weight)
        loss_items = self._sequence_items(loss)

        if row_count > 1 and weight_items is not None and len(weight_items) == row_count:
            row_weights = weight_items
        else:
            row_weights = [weight] * row_count

        if row_count > 1 and loss_items is not None and len(loss_items) == row_count:
            row_losses = loss_items
        else:
            row_losses = [loss] * row_count

        if len(row_weights) != row_count or len(row_losses) != row_count:
            raise ValueError("soft constraint weight/loss row counts must match constraints")
        return list(zip(row_weights, row_losses))

    def subject_to(self, *constraints, weight=None, loss='L2', include_terminal=True):
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
        constraint_rows = []
        for constraint_arg in constraints:
            constraint_rows.extend(self._flatten_constraint_args(constraint_arg))

        soft_specs = self._row_soft_specs(len(constraint_rows), weight, loss)

        for constraint, (row_weight, row_loss) in zip(constraint_rows, soft_specs):
            if isinstance(constraint, DynamicsEquation):
                self._set_dynamics_equation(constraint)
                continue
            if isinstance(constraint, (bool, sp.Equality)):
                raise ValueError("Use Dot/Next dynamics syntax, e.g. Dot(state) == expr or "
                                 "Next(state) == expr; "
                                 "general equality constraints are not supported")

            expr = constraint
            if isinstance(constraint, sp.LessThan): # lhs <= rhs -> lhs - rhs <= 0
                expr = constraint.lhs - constraint.rhs
            elif isinstance(constraint, sp.GreaterThan): # lhs >= rhs -> rhs - lhs <= 0
                expr = constraint.rhs - constraint.lhs

            expr = sp.sympify(expr)
            self._validate_declared_symbols("constraint", expr)
                
            # Add to constraints list
            self.constraints.append(expr)
            self.true_constraints.append(expr)
            self.constraint_include_terminal.append(bool(include_terminal))
            idx = len(self.constraints) - 1

            self.soft_constraints.extend(
                self._normalize_row_soft_entries(idx, row_weight, row_loss))

    def subject_to_quad(self, Q, x, center=None, rhs=0.0, sense='<=', type='outside',
                        linearize_at_boundary=False, rhs_mode='quadratic',
                        include_terminal=True):
        """
        Add a quadratic constraint: (x-center)^T Q (x-center) {sense} rhs.

        By default, rhs_mode='quadratic' generates a quadratic-form constraint:

            (x-center)^T Q (x-center) {sense} rhs

        With
        rhs_mode='norm2', rhs is interpreted as the right-hand side of a
        Euclidean norm constraint:

            sqrt((x-center)^T Q (x-center) + eps) - rhs <= 0

        This is the preferred form when the right-hand side is a decision
        variable or parameter, because the squared form q(x) - rhs**2 <= 0 has
        the same feasible set only when rhs >= 0 and loses the convex cone
        expression used by the exact Hessian path.

        MiniModel can statically reject negative numeric rhs values and
        non-PSD numeric Q matrices. If rhs or Q contains symbolic parameters,
        the model author must ensure rhs >= 0 and Q is PSD over the operating
        domain, for example by adding explicit model constraints.
        """
        if rhs_mode not in ('quadratic', 'norm2'):
            raise ValueError("rhs_mode must be 'quadratic' or 'norm2'")

        # Helper to process Q
        if not isinstance(Q, sp.Matrix):
            Q_mat = sp.Matrix(Q)
        else:
            Q_mat = Q
            
        x_vec = sp.Matrix(x)
        self._validate_declared_symbols("quadratic constraint variables", x_vec)
        
        if center is None:
            c_vec = sp.Matrix([0]*x_vec.rows)
        else:
            c_vec = sp.Matrix(center)
            if c_vec.rows != x_vec.rows:
                raise ValueError("Dimension mismatch in center and x")
            self._validate_declared_symbols("quadratic constraint center", c_vec)
            
        self._validate_declared_symbols("quadratic constraint Q", Q_mat)
        self._validate_declared_symbols("quadratic constraint rhs", rhs)

        if x_vec.rows != Q_mat.shape[0] or Q_mat.shape[0] != Q_mat.shape[1]:
             raise ValueError("Dimension mismatch in Q and x")

        # Form the quadratic term: (x-c)^T Q (x-c)
        diff = x_vec - c_vec
        quad_term = (diff.T * Q_mat * diff)[0]
        
        # Logic for Robust Formulation
        is_exclusion = (sense == '>=' or type == 'outside')
        if rhs_mode == 'norm2':
            if linearize_at_boundary:
                raise ValueError("linearize_at_boundary requires rhs_mode='quadratic'")
            self._validate_numeric_psd(Q_mat)
            epsilon = 1e-10
            rhs_expr = sp.sympify(rhs)
            if self._is_numeric_expr(rhs_expr) and float(sp.N(rhs_expr)) < 0.0:
                raise ValueError("norm2 rhs must be non-negative")
            robust_dist = sp.sqrt(quad_term + epsilon)
            if is_exclusion:
                self.constraints.append(rhs_expr - robust_dist)
                self.true_constraints.append(rhs_expr - robust_dist)
            else:
                self.constraints.append(robust_dist - rhs_expr)
                self.true_constraints.append(robust_dist - rhs_expr)
            self.constraint_include_terminal.append(bool(include_terminal))
            return

        if is_exclusion:
            self._validate_numeric_positive("rhs", rhs)
            self._validate_numeric_psd(Q_mat)
        
        if is_exclusion:
            if linearize_at_boundary:
                epsilon = 1e-6
                true_expr = sp.sqrt(rhs) - sp.sqrt(quad_term + epsilon)
                xp_syms = [sp.symbols(f"xp_{len(self.special_constraints)}_{i}") for i in range(x_vec.rows)]
                xp_vec = sp.Matrix(xp_syms)
                
                # Gradient at boundary: 2 Q (xp - c)
                grad_at_boundary = 2 * Q_mat * (xp_vec - c_vec)
                
                # Linearized Constraint: - grad^T (x - xp) <= 0
                boundary_linear_expr = - (grad_at_boundary.T * (x_vec - xp_vec))[0]
                
                self.constraints.append(boundary_linear_expr)
                self.true_constraints.append(true_expr)
                self.constraint_include_terminal.append(bool(include_terminal))
                
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
                self.true_constraints.append(robust_rhs - robust_dist)
                self.constraint_include_terminal.append(bool(include_terminal))
        else:
            # Standard form
            if sense == '<=':
                self.constraints.append(quad_term - rhs)
                self.true_constraints.append(quad_term - rhs)
            else:
                self.constraints.append(rhs - quad_term)
                self.true_constraints.append(rhs - quad_term)
            self.constraint_include_terminal.append(bool(include_terminal))

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
        clear_names = [
            name
            for name, idx, rows, cols in assignments
            if idx < len(reduced) and not self._packet_fully_assigned(reduced[idx], rows, cols)
        ]
        if clear_names:
            code += self._emit_clear_block(
                "kp",
                clear_names,
                "Clear generated output packets; nonzero entries are assigned below.",
            )
        for name, idx, rows, cols in assignments:
            if idx >= len(reduced): continue
            mat = reduced[idx]
            code += f"\n        // {name}\n"
            code += self._emit_sparse_packet_assign("kp", name, mat, rows, cols)
        return code

    @staticmethod
    def _is_nonzero_expr(expr):
        return sp.sympify(expr).is_zero is not True

    @staticmethod
    def _matrix_entry(mat, rows, cols, r, c):
        if rows == 1 or cols == 1:
            return mat[r] if rows > 1 else mat[c]
        return mat[r, c]

    def _packet_fully_assigned(self, mat, rows, cols):
        if rows * cols == 0:
            return False
        for r in range(rows):
            for c in range(cols):
                if not self._is_nonzero_expr(self._matrix_entry(mat, rows, cols, r, c)):
                    return False
        return True

    def _emit_cse_assignments(self, replacements, indent="        "):
        code = ""
        for name, val in replacements:
            code += f"{indent}T {name} = {sp.ccode(val)};\n"
        return code

    def _emit_clear_block(self, owner, names, comment, indent="        "):
        if not names:
            return ""
        code = f"\n{indent}// {comment}\n"
        for name in names:
            code += f"{indent}{owner}.{name}.setZero();\n"
        return code

    def _emit_sparse_packet_assign(self, owner, name, mat, rows, cols, indent="        "):
        code = ""
        for r in range(rows):
            for c in range(cols):
                val = self._matrix_entry(mat, rows, cols, r, c)
                if self._is_nonzero_expr(val):
                    code += f"{indent}{owner}.{name}({r},{c}) = {sp.ccode(val)};\n"
        return code

    def _emit_sparse_packet_add_if(
        self,
        owner,
        name,
        mat,
        rows,
        cols,
        condition,
        indent="        ",
    ):
        code = ""
        for r in range(rows):
            for c in range(cols):
                val = self._matrix_entry(mat, rows, cols, r, c)
                if self._is_nonzero_expr(val):
                    code += f"{indent}if constexpr ({condition}) {owner}.{name}({r},{c}) += {sp.ccode(val)};\n"
        return code

    def _generate_mode_hessian_packets(self, target_names, gn_mats, delta_mats, label_prefix=""):
        code = self._emit_clear_block(
            "kp",
            target_names,
            "Clear Hessian packets; nonzero entries are assigned below.",
        )
        for name, mat_gn, mat_delta in zip(target_names, gn_mats, delta_mats):
            rows = mat_gn.shape[0]
            cols = mat_gn.shape[1]
            code += f"\n        // {label_prefix}{name} (Mode 0=GN, 1=Exact)\n"
            code += self._emit_sparse_packet_assign("kp", name, mat_gn, rows, cols)
            code += self._emit_sparse_packet_add_if("kp", name, mat_delta, rows, cols, "Mode == 1")
        return code

    def _generate_special_constraint_preamble(self):
        code = "\n        // --- Special Constraints Pre-Calculation ---\n"
        for local_idx, info in enumerate(self.special_constraints):
            if info['type'] == 'quad_boundary_proj':
                # Calculate Projection
                x_vec = info['x']
                c_vec = info['c']
                Q_mat = info['Q']
                diff = x_vec - c_vec
                d2_expr = (diff.T * Q_mat * diff)[0]
                
                rhs_val = info['rhs'] # Usually a number or symbol
                
                d2_name = f"d2_{local_idx}"
                rhs_name = f"rhs_{local_idx}"
                scale_name = f"scale_{local_idx}"

                # Note: Declaring variables in outer scope to be visible to CSE expressions.
                # Names must be unique because multiple projected constraints share a scope.
                code += f"        T {d2_name} = {sp.ccode(d2_expr)};\n"
                code += f"        T {rhs_name} = {sp.ccode(rhs_val)};\n"
                code += f"        T {scale_name} = sqrt({rhs_name} / ({d2_name} + 1e-9));\n"
                
                for i, xp_sym in enumerate(info['xp_syms']):
                    # xp_i = c_i + scale * (x_i - c_i)
                    offset_expr = x_vec[i] - c_vec[i]
                    code += f"        T {xp_sym} = {sp.ccode(c_vec[i])} + {scale_name} * ({sp.ccode(offset_expr)});\n"
        return code

    def _terminal_substitutions(self):
        return {u: 0 for u in self.controls}

    def _generate_terminal_constraints_body(self, x_vec, u_vec):
        nc = len(self.constraints)
        nx = len(self.states)
        nu = len(self.controls)
        u_zero = self._terminal_substitutions()
        terminal_exprs = [
            sp.simplify(c.subs(u_zero)) if self.constraint_include_terminal[i] else sp.Integer(0)
            for i, c in enumerate(self.constraints)
        ]
        g_terminal = sp.Matrix(terminal_exprs) if nc > 0 else sp.Matrix.zeros(0, 1)
        C_terminal = g_terminal.jacobian(x_vec) if nc > 0 else sp.Matrix.zeros(0, nx)
        D_terminal = g_terminal.jacobian(u_vec) if nc > 0 else sp.Matrix.zeros(0, nu)

        con_exprs = [g_terminal, C_terminal, D_terminal]
        for info in self.special_constraints:
            con_exprs.append(info['x'])
            con_exprs.append(info['Q'])

        code = ""
        code += self._generate_unpack_block(source_kp=True, expressions=con_exprs)
        code += self._generate_special_constraint_preamble()
        code += "\n"
        assign_con = [("g_val", 0, nc, 1), ("C", 1, nc, nx), ("D", 2, nc, nu)]
        code += self._generate_assign_block(assign_con, [g_terminal, C_terminal, D_terminal])
        return code

    def _generate_true_constraints_body(self, terminal=False):
        nc = len(self.true_constraints)
        if nc == 0:
            return "        (void)kp;\n"

        exprs = self.true_constraints
        if terminal:
            u_zero = self._terminal_substitutions()
            exprs = [
                sp.simplify(c.subs(u_zero)) if self.constraint_include_terminal[i] else sp.Integer(0)
                for i, c in enumerate(exprs)
            ]

        g_true = sp.Matrix(exprs)
        code = self._generate_unpack_block(source_kp=True, expressions=[g_true])
        code += "\n"
        code += self._generate_assign_block([("g_true", 0, nc, 1)], [g_true])
        return code

    def _generate_soc_constraints_body(self):
        code = "        compute_qp_constraints(trial_kp);\n"
        code += "        compute_true_constraints(trial_kp);\n"
        projected = [
            info for info in self.special_constraints
            if info['type'] == 'quad_boundary_proj'
        ]
        if not projected:
            code += "        (void)active_kp;\n"
            return code

        active_map = {}
        trial_map = {}
        active_used = set()
        trial_used = set()

        def collect_symbols(expr, out):
            if isinstance(expr, sp.MatrixBase):
                for item in expr:
                    collect_symbols(item, out)
            elif hasattr(expr, 'free_symbols'):
                out.update(expr.free_symbols)

        for info in projected:
            for expr in (info['x'], info['Q'], info['c'], sp.sympify(info['rhs'])):
                collect_symbols(expr, active_used)
            collect_symbols(info['x'], trial_used)

        for sym in list(self.states) + list(self.controls) + list(self.parameters):
            active_sym = sp.symbols(f"soc_active_{sym.name}")
            trial_sym = sp.symbols(f"soc_trial_{sym.name}")
            active_map[sym] = active_sym
            trial_map[sym] = trial_sym

        for i, sym in enumerate(self.states):
            if sym in active_used:
                code += f"        T {active_map[sym]} = active_kp.x({i});\n"
        for i, sym in enumerate(self.controls):
            if sym in active_used:
                code += f"        T {active_map[sym]} = active_kp.u({i});\n"
        for i, sym in enumerate(self.parameters):
            if sym in active_used:
                code += f"        T {active_map[sym]} = active_kp.p({i});\n"

        for i, sym in enumerate(self.states):
            if sym in trial_used:
                code += f"        T {trial_map[sym]} = trial_kp.x({i});\n"
        for i, sym in enumerate(self.controls):
            if sym in trial_used:
                code += f"        T {trial_map[sym]} = trial_kp.u({i});\n"
        for i, sym in enumerate(self.parameters):
            if sym in trial_used:
                code += f"        T {trial_map[sym]} = trial_kp.p({i});\n"

        for local_idx, info in enumerate(projected):
            x_active = info['x'].xreplace(active_map)
            x_trial = info['x'].xreplace(trial_map)
            c_active = info['c'].xreplace(active_map)
            q_active = info['Q'].xreplace(active_map)
            rhs_active = sp.sympify(info['rhs']).xreplace(active_map)

            diff_active = x_active - c_active
            d2_expr = (diff_active.T * q_active * diff_active)[0]
            rhs_name = f"soc_rhs_{local_idx}"
            d2_name = f"soc_d2_{local_idx}"
            scale_name = f"soc_scale_{local_idx}"

            code += f"        T {d2_name} = {sp.ccode(d2_expr)};\n"
            code += f"        T {rhs_name} = {sp.ccode(rhs_active)};\n"
            code += f"        T {scale_name} = sqrt({rhs_name} / ({d2_name} + 1e-9));\n"

            xp_syms = [sp.symbols(f"soc_xp_{local_idx}_{i}") for i in range(info['x'].rows)]
            xp_vec = sp.Matrix(xp_syms)
            for i, xp_sym in enumerate(xp_syms):
                offset_expr = x_active[i] - c_active[i]
                code += (
                    f"        T {xp_sym} = {sp.ccode(c_active[i])} + {scale_name} "
                    f"* ({sp.ccode(offset_expr)});\n"
                )

            grad_at_boundary = 2 * q_active * (xp_vec - c_active)
            soc_residual = -(grad_at_boundary.T * (x_trial - xp_vec))[0]
            code += f"        trial_kp.g_val({info['index']}) = {sp.ccode(soc_residual)};\n"

        return code

    def _generate_terminal_cost_section(self, x_vec, u_vec):
        nx = len(self.states)
        nu = len(self.controls)
        nc = len(self.constraints)
        xu_vec = sp.Matrix.vstack(x_vec, u_vec)
        u_zero = self._terminal_substitutions()

        objective_terminal = self._total_objective(u_zero)
        general_objective_terminal = sp.simplify(sp.sympify(self.objective).subs(u_zero))
        constraints_terminal = [
            sp.simplify(c.subs(u_zero)) if self.constraint_include_terminal[i] else sp.Integer(0)
            for i, c in enumerate(self.constraints)
        ]

        grad_cost = sp.Matrix([objective_terminal]).jacobian(xu_vec).T
        q_grad = grad_cost[:nx, :]
        r_grad = grad_cost[nx:, :]

        hess_cost_exact = sp.hessian(objective_terminal, xu_vec)
        hess_cost_gn = sp.hessian(general_objective_terminal, xu_vec)
        hess_cost_gn += self._residual_gauss_newton_hessian(xu_vec, u_zero)
        hess_cost_delta = sp.simplify(hess_cost_exact - hess_cost_gn)
        Q_hess_gn = hess_cost_gn[:nx, :nx]
        R_hess_gn = hess_cost_gn[nx:, nx:]
        H_hess_gn = hess_cost_gn[nx:, :nx]

        lam_sym = [sp.symbols(f"lam_{i}") for i in range(nc)]
        hess_con_total = sp.zeros(nx + nu, nx + nu)
        if nc > 0:
            for i in range(nc):
                hess_con_total += lam_sym[i] * sp.hessian(constraints_terminal[i], xu_vec)

        hess_mode1_delta = sp.simplify(hess_cost_delta + hess_con_total)
        Q_hess_delta = hess_mode1_delta[:nx, :nx]
        R_hess_delta = hess_mode1_delta[nx:, nx:]
        H_hess_delta = hess_mode1_delta[nx:, :nx]

        exprs = [
            q_grad, r_grad, Q_hess_gn, R_hess_gn, H_hess_gn,
            Q_hess_delta, R_hess_delta, H_hess_delta, objective_terminal
        ]

        code = "template<typename T, int Mode>\n"
        code += "    static void compute_terminal_cost_impl(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        code += self._generate_unpack_block(source_kp=True, expressions=exprs)

        used_syms = set()
        for expr in [Q_hess_delta, R_hess_delta, H_hess_delta]:
            if hasattr(expr, 'free_symbols'):
                used_syms.update(expr.free_symbols)
        for i in range(nc):
            s_lam = sp.symbols(f"lam_{i}")
            if s_lam in used_syms:
                code += f"        T lam_{i} = kp.lam({i});\n"

        code += "\n"
        code += self._generate_assign_block(
            [("q", 0, nx, 1), ("r", 1, nu, 1)],
            [q_grad, r_grad],
        )

        code += self._generate_mode_hessian_packets(
            ["Q", "R", "H"],
            [Q_hess_gn, R_hess_gn, H_hess_gn],
            [Q_hess_delta, R_hess_delta, H_hess_delta],
            label_prefix="terminal ",
        )

        code += f"\n        kp.cost = {sp.ccode(objective_terminal)};\n"
        code += "    }\n\n"
        code += "    template<typename T>\n"
        code += "    static void compute_terminal_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        code += "        compute_terminal_cost_impl<T, 0>(kp);\n"
        code += "    }\n\n"
        code += "    template<typename T>\n"
        code += "    static void compute_terminal_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        code += "        compute_terminal_cost_impl<T, 1>(kp);\n"
        code += "    }\n"
        return code

    def _generate_stage_cost_section(self, x_vec, u_vec):
        nx = len(self.states)
        nu = len(self.controls)
        nc = len(self.constraints)
        xu_vec = sp.Matrix.vstack(x_vec, u_vec)

        total_objective = self._total_objective()
        general_objective = sp.sympify(self.objective)

        grad_cost = sp.Matrix([total_objective]).jacobian(xu_vec).T
        q_grad = grad_cost[:nx, :]
        r_grad = grad_cost[nx:, :]

        # Mode 0 uses exact general-objective Hessian plus the residual
        # Gauss-Newton term. Mode 1 adds residual second-order terms and
        # constraint Hessians.
        hess_cost_exact = sp.hessian(total_objective, xu_vec)
        hess_cost_gn = sp.hessian(general_objective, xu_vec)
        hess_cost_gn += self._residual_gauss_newton_hessian(xu_vec)
        hess_cost_delta = sp.simplify(hess_cost_exact - hess_cost_gn)
        Q_hess_gn = hess_cost_gn[:nx, :nx]
        R_hess_gn = hess_cost_gn[nx:, nx:]
        H_hess_gn = hess_cost_gn[nx:, :nx]

        lam_sym = [sp.symbols(f"lam_{i}") for i in range(nc)]
        hess_con_total = sp.zeros(nx + nu, nx + nu)
        if nc > 0:
            for i in range(nc):
                hess_g_i = sp.hessian(self.constraints[i], xu_vec)
                hess_con_total += lam_sym[i] * hess_g_i

        hess_mode1_delta = sp.simplify(hess_cost_delta + hess_con_total)
        Q_hess_delta = hess_mode1_delta[:nx, :nx]
        R_hess_delta = hess_mode1_delta[nx:, nx:]
        H_hess_delta = hess_mode1_delta[nx:, :nx]

        repl_cost, reduced_cost = sp.cse(
            [
                q_grad, r_grad, Q_hess_gn, R_hess_gn, H_hess_gn,
                Q_hess_delta, R_hess_delta, H_hess_delta, total_objective,
            ],
            symbols=sp.numbered_symbols("tmp_j"),
        )

        code_cost_impl = "template<typename T, int Mode>\n"
        code_cost_impl += "    static void compute_cost_impl(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"

        # Determine symbols used in cost + constraint Hessians. Parameters used
        # only in linear primal expressions disappear from derivatives and do
        # not need unpacking here.
        cost_exprs = [
            q_grad, r_grad, Q_hess_gn, R_hess_gn, H_hess_gn,
            Q_hess_delta, R_hess_delta, H_hess_delta, total_objective,
        ]

        code_unpack = self._generate_unpack_block(source_kp=True, expressions=cost_exprs)
        if nc > 0:
            hess_exprs = [Q_hess_delta, R_hess_delta, H_hess_delta]
            used_syms = set()
            for expr in hess_exprs:
                if hasattr(expr, 'free_symbols'):
                    used_syms.update(expr.free_symbols)

            for i in range(nc):
                s_lam = sp.symbols(f"lam_{i}")
                if s_lam in used_syms:
                    code_unpack += f"        T lam_{i} = kp.lam({i});\n"

        code_cse = "\n"
        code_cse += self._emit_cse_assignments(repl_cost)

        code_assign = ""
        code_assign += self._generate_assign_block(
            [("q", 0, nx, 1), ("r", 1, nu, 1)],
            reduced_cost,
        )
        code_assign += self._generate_mode_hessian_packets(
            ["Q", "R", "H"],
            reduced_cost[2:5],
            reduced_cost[5:8],
        )
        if len(reduced_cost) > 8:
            code_assign += f"\n        kp.cost = {sp.ccode(reduced_cost[8])};\n"

        code_cost_impl += code_unpack + code_cse + code_assign
        code_cost_impl += "    }\n\n"

        code_wrappers = "template<typename T>\n"
        code_wrappers += "    static void compute_cost_gn(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        code_wrappers += "        compute_cost_impl<T, 0>(kp);\n"
        code_wrappers += "    }\n\n"

        code_wrappers += "    template<typename T>\n"
        code_wrappers += "    static void compute_cost_exact(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        code_wrappers += "        compute_cost_impl<T, 1>(kp);\n"
        code_wrappers += "    }\n\n"

        # Default alias stays exact for backward compatibility.
        code_wrappers += "    template<typename T>\n"
        code_wrappers += "    static void compute_cost(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        code_wrappers += "        compute_cost_impl<T, 1>(kp);\n"
        code_wrappers += "    }\n"

        return code_cost_impl + code_wrappers, r_grad

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

    def _nonzero_pattern(self, mat, rows, cols):
        pattern = set()
        for r in range(rows):
            for c in range(cols):
                if sp.sympify(mat[r, c]).is_zero is not True:
                    pattern.add((r, c))
        return pattern

    def _identity_pattern(self, n):
        return {(i, i) for i in range(n)}

    def _matmul_pattern(self, lhs, rhs, rows, inner, cols):
        lhs_by_row = {}
        rhs_by_row = {}
        for r, c in lhs:
            lhs_by_row.setdefault(r, set()).add(c)
        for r, c in rhs:
            rhs_by_row.setdefault(r, set()).add(c)

        result = set()
        for r in range(rows):
            for k in lhs_by_row.get(r, ()):
                for c in rhs_by_row.get(k, ()):
                    result.add((r, c))
        return result

    def _inverse_pattern(self, pattern, n):
        reachable = [[False for _ in range(n)] for _ in range(n)]
        for i in range(n):
            reachable[i][i] = True
        for r, c in pattern:
            reachable[r][c] = True

        for k in range(n):
            for i in range(n):
                if not reachable[i][k]:
                    continue
                for j in range(n):
                    if reachable[k][j]:
                        reachable[i][j] = True

        return {(i, j) for i in range(n) for j in range(n) if reachable[i][j]}

    def _backward_euler_riccati_patterns(self, Jx_pattern, Ju_pattern, nx, nu):
        m_pattern = self._identity_pattern(nx) | Jx_pattern
        m_inv_pattern = self._inverse_pattern(m_pattern, nx)
        A_pattern = m_inv_pattern
        B_pattern = self._matmul_pattern(m_inv_pattern, Ju_pattern, nx, nx, nu)
        return A_pattern, B_pattern

    def _implicit_midpoint_riccati_patterns(self, Jx_pattern, Ju_pattern, nx, nu):
        m_minus_pattern = self._identity_pattern(nx) | Jx_pattern
        m_plus_pattern = self._identity_pattern(nx) | Jx_pattern
        m_inv_pattern = self._inverse_pattern(m_minus_pattern, nx)
        A_pattern = self._matmul_pattern(m_inv_pattern, m_plus_pattern, nx, nx, nx)
        B_pattern = self._matmul_pattern(m_inv_pattern, Ju_pattern, nx, nx, nu)
        return A_pattern, B_pattern

    def _gauss_legendre_2_riccati_patterns(self, Jx_pattern, Ju_pattern, nx, nu):
        n2 = 2 * nx
        jk_pattern = set()

        # J = [I - a11*dt*Jx,  -a12*dt*Jx]
        #     [-a21*dt*Jx,      I - a22*dt*Jx]
        for r in range(nx):
            jk_pattern.add((r, r))
            jk_pattern.add((nx + r, nx + r))
        for r, c in Jx_pattern:
            jk_pattern.add((r, c))
            jk_pattern.add((r, nx + c))
            jk_pattern.add((nx + r, c))
            jk_pattern.add((nx + r, nx + c))

        jk_inv_pattern = self._inverse_pattern(jk_pattern, n2)

        rhs_x_pattern = set()
        for r, c in Jx_pattern:
            rhs_x_pattern.add((r, c))
            rhs_x_pattern.add((nx + r, c))

        rhs_u_pattern = set()
        for r, c in Ju_pattern:
            rhs_u_pattern.add((r, c))
            rhs_u_pattern.add((nx + r, c))

        dK_dx_pattern = self._matmul_pattern(jk_inv_pattern, rhs_x_pattern, n2, n2, nx)
        dK_du_pattern = self._matmul_pattern(jk_inv_pattern, rhs_u_pattern, n2, n2, nu)

        A_pattern = self._identity_pattern(nx)
        for r, c in dK_dx_pattern:
            if r < nx:
                A_pattern.add((r, c))
            else:
                A_pattern.add((r - nx, c))

        B_pattern = set()
        for r, c in dK_du_pattern:
            if r < nx:
                B_pattern.add((r, c))
            else:
                B_pattern.add((r - nx, c))

        return A_pattern, B_pattern

    def _generate_implicit_riccati_patterns(self, integrator_type, Jx_cont, Ju_cont, nx, nu):
        """
        Conservative discrete A/B sparsity for implicit integrators.

        Implicit methods form inverse systems such as (I - dt*Jx)^-1. A sparse
        Jx can fill in through that inverse, so using the explicit integrator
        pattern is unsafe. Dispatch by the requested implicit method so future
        integrator-specific tightening stays local.
        """
        Jx_pattern = self._nonzero_pattern(Jx_cont, nx, nx)
        Ju_pattern = self._nonzero_pattern(Ju_cont, nx, nu)

        if integrator_type == 'EULER_IMPLICIT':
            return self._backward_euler_riccati_patterns(Jx_pattern, Ju_pattern, nx, nu)
        if integrator_type == 'RK2_IMPLICIT':
            return self._implicit_midpoint_riccati_patterns(Jx_pattern, Ju_pattern, nx, nu)
        if integrator_type == 'RK4_IMPLICIT':
            return self._gauss_legendre_2_riccati_patterns(Jx_pattern, Ju_pattern, nx, nu)

        raise ValueError(f"Unsupported implicit integrator: {integrator_type}")

    def _generate_fused_riccati_step(self, A_expr, B_expr, nx, nu):
        A_pattern = self._nonzero_pattern(A_expr, nx, nx)
        B_pattern = self._nonzero_pattern(B_expr, nx, nu)
        return self._generate_riccati_step_from_patterns(A_pattern, B_pattern, nx, nu)

    def _generate_riccati_step_from_patterns(self, A_pattern, B_pattern, nx, nu, label="Fused"):
        """
        Generates fused Riccati update kernel.
        Updates Qxx, Quu, Qux, qx, ru simultaneously.
        """
        print(f"Generating {label} Riccati Kernel (NX={nx}, NU={nu})...")
        
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
                if (r, c) in A_pattern:
                    A_sym[r, c] = sp.symbols(f"A_{r}_{c}")
                    
        # B (Sparse Structure)
        B_sym = sp.zeros(nx, nu)
        for r in range(nx):
            for c in range(nu):
                if (r, c) in B_pattern:
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

    def _resolve_integrator_type(self, integrator_type):
        if integrator_type is None:
            integrator_type = "DISCRETE" if self.dynamics_mode == "next" else "RK4_EXPLICIT"

        valid_integrators = {
            "EULER_EXPLICIT", "EULER_IMPLICIT", "RK2_EXPLICIT",
            "RK2_IMPLICIT", "RK4_EXPLICIT", "RK4_IMPLICIT", "DISCRETE",
        }
        if integrator_type not in valid_integrators:
            raise ValueError(f"Unknown integrator_type: {integrator_type}")

        if self.dynamics_mode is None:
            missing_dynamics = [s.name for s in self.states]
        else:
            dynamics_map = self.dynamics_rhs if self.dynamics_mode == "dot" else self.next_state_rhs
            missing_dynamics = [s.name for s in self.states if s not in dynamics_map]
        if missing_dynamics:
            names = ", ".join(missing_dynamics)
            raise ValueError(f"Missing dynamics for state(s): {names}")
        if self.dynamics_mode == "dot" and integrator_type == "DISCRETE":
            raise ValueError("DISCRETE integrator requires Next(state) dynamics")
        if self.dynamics_mode == "next" and integrator_type != "DISCRETE":
            raise ValueError("Next(state) dynamics require integrator_type='DISCRETE'")

        return integrator_type

    def _generate_model_constants(self, nx, nu, nc, np_param, integrator_type):
        code = f"static const int NX={nx};\n    static const int NU={nu};\n    static const int NC={nc};\n    static const int NP={np_param};"

        fingerprint = self._compute_model_fingerprint(integrator_type)
        code += f"\n\n    static constexpr std::uint64_t model_fingerprint = 0x{fingerprint:016x}ull;"

        # Marker: which integrator the fused Riccati kernel was CSE'd against.
        # MiniSolver's constructor reads this (SFINAE-detected) and warns on a
        # mismatched runtime config.integrator. Riccati dispatch then skips the
        # fused kernel because its sparsity pattern is pinned to this integrator.
        code += f"\n\n    static constexpr IntegratorType generated_integrator = IntegratorType::{integrator_type};"

        has_l1 = [False] * nc
        has_l2 = [False] * nc
        for sc in self.soft_constraints:
            idx = sc['index']
            if idx < nc:
                if sc['type'] == 'L1':
                    has_l1[idx] = True
                elif sc['type'] == 'L2':
                    has_l2[idx] = True

        l1_flags = "{" + ", ".join("true" if flag else "false" for flag in has_l1) + "}"
        l2_flags = "{" + ", ".join("true" if flag else "false" for flag in has_l2) + "}"
        any_l1 = "true" if any(has_l1) else "false"
        any_l2 = "true" if any(has_l2) else "false"
        code += "\n\n"
        code += f"    static constexpr std::array<bool, NC> constraint_has_l1 = {l1_flags};\n"
        code += f"    static constexpr std::array<bool, NC> constraint_has_l2 = {l2_flags};\n"
        code += f"    static constexpr bool any_l1_constraints = {any_l1};\n"
        code += f"    static constexpr bool any_l2_constraints = {any_l2};\n"
        return code

    def _generate_soft_constraint_weight_updater(self, loss_type):
        entries = [sc for sc in self.soft_constraints if sc['type'] == loss_type]
        target = "l1_weight" if loss_type == "L1" else "l2_weight"
        function_name = (
            "update_l1_soft_constraint_weights"
            if loss_type == "L1"
            else "update_l2_soft_constraint_weights"
        )

        code = "    template<typename T>\n"
        code += f"    static void {function_name}(KnotPoint<T,NX,NU,NC,NP>& kp) {{\n"
        code += f"        kp.{target}.setZero();\n"
        weight_exprs = [sc['weight'] for sc in entries]
        code += self._generate_unpack_block(source_kp=True, expressions=weight_exprs)
        for sc in entries:
            code += f"        kp.{target}({sc['index']}) = {sp.ccode(sc['weight'])};\n"
        code += "    }\n"
        return code

    def _generate_soft_constraint_weights_section(self):
        code = "    // --- 1.5 Update Soft Constraint Weights ---\n"
        if not self.soft_constraints:
            code += "    template<typename T>\n"
            code += "    static void update_soft_constraint_weights(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
            code += "        (void)kp;\n"
            code += "    }\n"
            return code

        any_l1 = any(sc['type'] == 'L1' for sc in self.soft_constraints)
        any_l2 = any(sc['type'] == 'L2' for sc in self.soft_constraints)
        if any_l1:
            code += self._generate_soft_constraint_weight_updater("L1")
            code += "\n"
        if any_l2:
            code += self._generate_soft_constraint_weight_updater("L2")
            code += "\n"

        code += "    template<typename T>\n"
        code += "    static void update_soft_constraint_weights(KnotPoint<T,NX,NU,NC,NP>& kp) {\n"
        if any_l1:
            code += "        update_l1_soft_constraint_weights<T>(kp);\n"
        if any_l2:
            code += "        update_l2_soft_constraint_weights<T>(kp);\n"
        code += "    }\n"
        return code

    def _generate_name_arrays(self):
        code = "static constexpr std::array<const char*, NX> state_names = {\n"
        for s in self.states:
            code += f'        "{s.name}",\n'
        code += "    };\n\n"
        code += "    static constexpr std::array<const char*, NU> control_names = {\n"
        for u in self.controls:
            code += f'        "{u.name}",\n'
        code += "    };\n\n"
        code += "    static constexpr std::array<const char*, NP> param_names = {\n"
        for p in self.parameters:
            code += f'        "{p.name}",\n'
        code += "    };\n"
        return code

    def _generate_continuous_dynamics_body(self, f_cont, nx):
        code = self._generate_unpack_block(source_kp=False, expressions=f_cont)
        code += "\n        MSVec<T, NX> xdot;\n"
        for i in range(nx):
            code += f"        xdot({i}) = {sp.ccode(f_cont[i])};\n"
        code += "        return xdot;\n"
        return code

    def _generate_continuous_dynamics_section(self, f_cont, x_vec, u_vec, nx, nu):
        if self.dynamics_mode != "dot":
            code = "    // Next(state) models are direct discrete maps and do not define\n"
            code += "    // continuous dynamics or continuous Jacobians."
            return code, None, None

        code_dyn_cont = self._generate_continuous_dynamics_body(f_cont, nx)
        code_jac_body, Jx_cont, Ju_cont = self._generate_continuous_jacobian_body(
            f_cont, x_vec, u_vec, nx, nu)

        code = """    // --- Continuous Dynamics ---
    template<typename T>
    static MSVec<T, NX> dynamics_continuous(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& u_in,
        const MSVec<T, NP>& p_in)
    {
"""
        code += code_dyn_cont
        code += """
    }

    // --- Continuous Dynamics Jacobians (for implicit integrators) ---
    template<typename T>
    static ContinuousJacobians<T, NX, NU> jacobian_continuous(
        const MSVec<T, NX>& x_in,
        const MSVec<T, NU>& u_in,
        const MSVec<T, NP>& p_in)
    {
"""
        code += code_jac_body
        code += "\n    }"
        return code, Jx_cont, Ju_cont

    def _generate_integrate_body(self, x_next_direct, nx):
        if self.dynamics_mode == "dot":
            return """        switch(type) {
            case IntegratorType::EULER_EXPLICIT:
                return x_in + dynamics_continuous(x_in, u_in, p_in) * dt;

            case IntegratorType::RK2_EXPLICIT:
            {
               auto k1 = dynamics_continuous(x_in, u_in, p_in);
               auto k2 = dynamics_continuous<T>(x_in + k1 * (0.5 * dt), u_in, p_in);
               return x_in + k2 * dt;
            }

            case IntegratorType::EULER_IMPLICIT:
            {
                // Simple Fixed-Point Iteration for x_next = x + f(x_next, u) * dt
                MSVec<T, NX> x_next = x_in; // Guess
                for(int i=0; i<5; ++i) {
                    x_next = x_in + dynamics_continuous(x_next, u_in, p_in) * dt;
                }
                return x_next;
            }

            case IntegratorType::RK2_IMPLICIT:
            {
                // Implicit Midpoint: k = f(x + 0.5*dt*k). x_next = x + dt*k
                MSVec<T, NX> k = dynamics_continuous(x_in, u_in, p_in); // Guess k0
                for(int i=0; i<5; ++i) {
                    k = dynamics_continuous<T>(x_in + k * (0.5 * dt), u_in, p_in);
                }
                return x_in + k * dt;
            }

            case IntegratorType::RK4_EXPLICIT:
            case IntegratorType::RK4_IMPLICIT:
            {
               auto k1 = dynamics_continuous(x_in, u_in, p_in);
               auto k2 = dynamics_continuous<T>(x_in + k1 * (0.5 * dt), u_in, p_in);
               auto k3 = dynamics_continuous<T>(x_in + k2 * (0.5 * dt), u_in, p_in);
               auto k4 = dynamics_continuous<T>(x_in + k3 * dt, u_in, p_in);
               return x_in + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
            }

            case IntegratorType::DISCRETE:
                throw std::invalid_argument("DISCRETE integrator requires Next(state) dynamics");
        }
        throw std::invalid_argument("Unsupported integrator type");
"""

        code = ""
        code += "        if (type != IntegratorType::DISCRETE) {\n"
        code += "            throw std::invalid_argument(\"Next(state) dynamics require IntegratorType::DISCRETE\");\n"
        code += "        }\n"
        code += "        (void)dt;\n"
        code += self._generate_unpack_block(source_kp=False, expressions=x_next_direct)
        code += "\n        MSVec<T, NX> x_next;\n"
        for i in range(nx):
            code += f"        x_next({i}) = {sp.ccode(x_next_direct[i])};\n"
        code += "        return x_next;\n"
        return code

    def _build_dynamics_expressions(self):
        if self.dynamics_mode == "dot":
            f_cont_list = [self.dynamics_rhs[s] for s in self.states]
            x_next_direct = None
        else:
            f_cont_list = []
            x_next_direct = sp.Matrix([self.next_state_rhs[s] for s in self.states])
        return sp.Matrix(f_cont_list), x_next_direct

    def _generate_integrator_groups(self, x_vec, u_vec, f_cont, x_next_direct, dt_sym):
        integrators = {}

        if self.dynamics_mode == "dot":
            def get_f_subs(x_in, u_in):
                repl = {self.states[i]: x_in[i] for i in range(len(self.states))}
                return f_cont.subs(repl)

            x_next_euler = x_vec + dt_sym * f_cont
            integrators['EULER_EXPLICIT'] = x_next_euler

            k1_rk2 = f_cont
            k2_rk2 = get_f_subs(x_vec + 0.5*dt_sym*k1_rk2, u_vec)
            x_next_rk2 = x_vec + dt_sym * k2_rk2
            integrators['RK2_EXPLICIT'] = x_next_rk2

            k1_rk4 = f_cont
            k2_rk4 = get_f_subs(x_vec + 0.5*dt_sym*k1_rk4, u_vec)
            k3_rk4 = get_f_subs(x_vec + 0.5*dt_sym*k2_rk4, u_vec)
            k4_rk4 = get_f_subs(x_vec + dt_sym*k3_rk4, u_vec)
            x_next_rk4 = x_vec + (dt_sym / 6.0) * (k1_rk4 + 2*k2_rk4 + 2*k3_rk4 + k4_rk4)
            integrators['RK4_EXPLICIT'] = x_next_rk4

            # Map implicit to explicit for the generated switch cases.
            # The runtime ImplicitIntegrator handles the actual implicit solve.
            integrators['EULER_IMPLICIT'] = x_next_euler
            integrators['RK2_IMPLICIT'] = x_next_rk2
            integrators['RK4_IMPLICIT'] = x_next_rk4

            return [
                (['EULER_EXPLICIT', 'EULER_IMPLICIT'], integrators['EULER_EXPLICIT']),
                (['RK2_EXPLICIT', 'RK2_IMPLICIT'], integrators['RK2_EXPLICIT']),
                (['RK4_EXPLICIT', 'RK4_IMPLICIT'], integrators['RK4_EXPLICIT']),
            ]

        integrators['DISCRETE'] = x_next_direct
        return [(['DISCRETE'], integrators['DISCRETE'])]

    def _generate_compute_dynamics_dispatch(self, groups, integrator_type, x_vec, u_vec, nx, nu):
        code = ""
        if self.dynamics_mode == "dot":
            code = "        switch(type) {\n"

        target_A_expr = None
        target_B_expr = None

        for labels, x_next_expr in groups:
            print(f"Generating derivatives for {labels}...")

            A_expr = x_next_expr.jacobian(x_vec)
            B_expr = x_next_expr.jacobian(u_vec)

            if integrator_type in labels:
                target_A_expr = A_expr
                target_B_expr = B_expr

            repl_dyn, reduced_dyn = sp.cse(
                [x_next_expr, A_expr, B_expr],
                symbols=sp.numbered_symbols("tmp_d"),
            )

            # Generate Case Statements. DISCRETE models are direct maps, so the
            # runtime integrator enum and dt are intentionally ignored.
            if self.dynamics_mode == "dot":
                for label in labels:
                    code += f"            case IntegratorType::{label}:\n"
                code += "            {\n"
            else:
                code += "        // Direct discrete dynamics generated from Next(state) equations.\n"
                code += "        if (type != IntegratorType::DISCRETE) {\n"
                code += "            throw std::invalid_argument(\"Next(state) dynamics require IntegratorType::DISCRETE\");\n"
                code += "        }\n"
                code += "        (void)dt;\n"

            code += self._emit_cse_assignments(repl_dyn, indent="                ")

            if nx > 0:
                mat = reduced_dyn[0]
                for r in range(nx):
                    val = mat[r]
                    code += f"                kp.f_resid({r}) = {sp.ccode(val)};\n"

                mat = reduced_dyn[1]
                code += self._emit_clear_block(
                    "kp",
                    ["A"],
                    "Clear dynamics Jacobian A; nonzero entries are assigned below.",
                    indent="                ",
                )
                code += self._emit_sparse_packet_assign(
                    "kp", "A", mat, nx, nx, indent="                ")

            if nx > 0 and nu > 0:
                mat = reduced_dyn[2]
                code += self._emit_clear_block(
                    "kp",
                    ["B"],
                    "Clear dynamics Jacobian B; nonzero entries are assigned below.",
                    indent="                ",
                )
                code += self._emit_sparse_packet_assign(
                    "kp", "B", mat, nx, nu, indent="                ")

            if self.dynamics_mode == "dot":
                code += "                break;\n"
                code += "            }\n"

        if self.dynamics_mode == "dot":
            code += "            case IntegratorType::DISCRETE:\n"
            code += "                throw std::invalid_argument(\"DISCRETE integrator requires Next(state) dynamics\");\n"
            code += "        }"

        return code, target_A_expr, target_B_expr

    def _generate_continuous_jacobian_body(self, f_cont, x_vec, u_vec, nx, nu):
        print("Generating continuous Jacobians for implicit integrators...")
        Jx_cont = f_cont.jacobian(x_vec)
        Ju_cont = f_cont.jacobian(u_vec)

        repl_jac, reduced_jac = sp.cse(
            [Jx_cont, Ju_cont],
            symbols=sp.numbered_symbols("tmp_jc"))

        jac_exprs = [Jx_cont, Ju_cont]
        code = self._generate_unpack_block(source_kp=False, expressions=jac_exprs)
        code += "\n        ContinuousJacobians<T, NX, NU> jac;\n"
        code += self._emit_cse_assignments(repl_jac)
        code += self._emit_clear_block(
            "jac",
            ["Jx", "Ju"],
            "Clear continuous Jacobian packets; nonzero entries are assigned below.",
        )

        code += "\n        // Jx = df/dx\n"
        code += self._emit_sparse_packet_assign("jac", "Jx", reduced_jac[0], nx, nx)
        code += "\n        // Ju = df/du\n"
        code += self._emit_sparse_packet_assign("jac", "Ju", reduced_jac[1], nx, nu)
        code += "\n        return jac;\n"

        return code, Jx_cont, Ju_cont

    def _render_model_header(self, replacements, code_fused_riccati):
        template_path = os.path.join(os.path.dirname(__file__), "templates", "model.h.in")
        with open(template_path, 'r') as f:
            content = f.read()

        for marker, code in replacements.items():
            content = content.replace(f"{{{{{marker}}}}}", code)

        if self.use_fused_riccati:
            content = content.replace("{{FUSED_RICCATI_STEP_BODY}}", code_fused_riccati)
            content = re.sub(r"// \[\[SPARSE_KERNELS_START\]\]", "", content)
            content = re.sub(r"// \[\[SPARSE_KERNELS_END\]\]", "", content)
        else:
            content = re.sub(
                r"// \[\[SPARSE_KERNELS_START\]\].*?// \[\[SPARSE_KERNELS_END\]\]",
                "",
                content,
                flags=re.DOTALL,
            )

        return "\n".join(line.rstrip() for line in content.splitlines()) + "\n"

    def _warn_terminal_control_projection(self, r_grad, D_expr, nu, nc):
        if nu <= 0:
            return

        u_used_in_cost = any(not sp.sympify(r_grad[i]).is_zero for i in range(nu))
        u_used_in_con = False
        if nc > 0:
            for r in range(nc):
                for c in range(nu):
                    if not sp.sympify(D_expr[r, c]).is_zero:
                        u_used_in_con = True
                        break
                if u_used_in_con:
                    break

        if u_used_in_cost:
            print(f"WARNING: cost depends on u. Terminal stage (k=N) has no "
                  f"control decision; terminal cost will project controls to 0.")
        if u_used_in_con:
            print(f"WARNING: constraints depend on u. Terminal stage (k=N) has "
                  f"no control decision; terminal constraints will project controls to 0.")

    def _write_generated_header(self, output_dir, content):
        file_name = "car_model.h"
        if self.name != "CarModel":
            file_name = f"{self.name.lower()}.h"

        output_path = os.path.join(output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating {output_path}...")
        with open(output_path, 'w') as f:
            f.write(content)

    def generate(self, output_dir="include/model", use_fused_riccati=True, integrator_type=None):
        integrator_type = self._resolve_integrator_type(integrator_type)

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

        # 2. Dynamics
        dt_sym = sp.symbols('dt')
        f_cont, x_next_direct = self._build_dynamics_expressions()
        groups = self._generate_integrator_groups(
            x_vec, u_vec, f_cont, x_next_direct, dt_sym)
        code_compute_dyn_body, target_A_expr, target_B_expr = (
            self._generate_compute_dynamics_dispatch(
                groups, integrator_type, x_vec, u_vec, nx, nu)
        )
        code_continuous_section, Jx_cont, Ju_cont = self._generate_continuous_dynamics_section(
            f_cont, x_vec, u_vec, nx, nu)

        # 3.5 Derivatives for Cost & Constraints (Independent of Integrator)
        g_vec = sp.Matrix(self.constraints) if nc > 0 else sp.Matrix.zeros(0,1)
        C_expr = g_vec.jacobian(x_vec) if nc > 0 else sp.Matrix.zeros(0, nx)
        D_expr = g_vec.jacobian(u_vec) if nc > 0 else sp.Matrix.zeros(0, nu)

        # CSE Constraints/Cost
        print("Optimizing expressions (CSE)...")
        repl_con, reduced_con = sp.cse([g_vec, C_expr, D_expr], symbols=sp.numbered_symbols("tmp_c"))
        code_compute_cost, r_grad = self._generate_stage_cost_section(x_vec, u_vec)
        
        # 5. Code Construction
        
        code_constants = self._generate_model_constants(
            nx, nu, nc, np_param, integrator_type)
        code_names = self._generate_name_arrays()

        code_integrate = self._generate_integrate_body(x_next_direct, nx)

        # Compute Dynamics Body
        # Note: Discrete dynamics depends on f_cont logic (and thus same symbols), 
        # plus potentially x itself (x_next = x + ...).
        # We pass f_cont and x_vec to be safe.
        dyn_exprs = [f_cont, x_vec]
        if self.dynamics_mode == "next":
            dyn_exprs.append(x_next_direct)
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
        code_compute_con += self._emit_cse_assignments(repl_con)
        assign_con = [("g_val", 0, nc, 1), ("C", 1, nc, nx), ("D", 2, nc, nu)]
        code_compute_con += self._generate_assign_block(assign_con, reduced_con)
        code_compute_true_con = self._generate_true_constraints_body(terminal=False)
        code_compute_terminal_con = self._generate_terminal_constraints_body(x_vec, u_vec)
        code_compute_terminal_true_con = self._generate_true_constraints_body(terminal=True)
        code_compute_soc_con = self._generate_soc_constraints_body()
        code_update_soft_weights = self._generate_soft_constraint_weights_section()

        code_compute_terminal_cost = self._generate_terminal_cost_section(x_vec, u_vec)
        
        # [NEW] Generate Fused Riccati Kernel
        code_fused_riccati = ""
        if self.use_fused_riccati:
            implicit_integrators = {'EULER_IMPLICIT', 'RK2_IMPLICIT', 'RK4_IMPLICIT'}
            if integrator_type in implicit_integrators:
                try:
                    A_pattern, B_pattern = self._generate_implicit_riccati_patterns(
                        integrator_type, Jx_cont, Ju_cont, nx, nu)
                    code_fused_riccati = self._generate_riccati_step_from_patterns(
                        A_pattern, B_pattern, nx, nu, label=f"{integrator_type} Sparse")
                except Exception as e:
                    print(f"Warning: Failed to generate implicit sparse Riccati kernel: {e}")
            elif target_A_expr is not None:
                try:
                    # Use the specified integrator expressions
                    code_fused_riccati = self._generate_fused_riccati_step(target_A_expr, target_B_expr, nx, nu)
                except Exception as e:
                    print(f"Warning: Failed to generate fused kernel: {e}")
            else:
                print(f"Warning: Integrator type '{integrator_type}' not found in generated groups. Fused Riccati Kernel will be empty.")

        # 6. Read Template & Replace
        content = self._render_model_header({
            "MODEL_NAME": self.name,
            "CONSTANTS": code_constants,
            "NAME_ARRAYS": code_names,
            "CONTINUOUS_DYNAMICS_SECTION": code_continuous_section,
            "INTEGRATE_BODY": code_integrate,
            "COMPUTE_DYNAMICS_BODY": code_compute_dyn,
            "UPDATE_SOFT_CONSTRAINT_WEIGHTS_SECTION": code_update_soft_weights,
            "COMPUTE_CONSTRAINTS_BODY": code_compute_con,
            "COMPUTE_TRUE_CONSTRAINTS_BODY": code_compute_true_con,
            "COMPUTE_TERMINAL_CONSTRAINTS_BODY": code_compute_terminal_con,
            "COMPUTE_TERMINAL_TRUE_CONSTRAINTS_BODY": code_compute_terminal_true_con,
            "COMPUTE_SOC_CONSTRAINTS_BODY": code_compute_soc_con,
            "COMPUTE_COST_SECTION": code_compute_cost,
            "COMPUTE_TERMINAL_COST_SECTION": code_compute_terminal_cost,
        }, code_fused_riccati)

        terminal_D_expr = sp.zeros(nc, nu)
        if nc > 0:
            for r in range(nc):
                if self.constraint_include_terminal[r]:
                    for c in range(nu):
                        terminal_D_expr[r, c] = D_expr[r, c]
        self._warn_terminal_control_projection(r_grad, terminal_D_expr, nu, nc)

        # 7. Write Output
        self._write_generated_header(output_dir, content)
