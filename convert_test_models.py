#!/usr/bin/env python3
"""
Script to convert old KnotPointV2-based test models to new Split Architecture.
"""

import re
import sys

def convert_compute_function(content):
    """Convert old compute() function to new split architecture."""
    
    # Pattern to match the old compute function
    pattern = r'(template<typename T>\s+static void compute\(KnotPointV2<T,NX,NU,NC,NP>& kp,.*?\n\s*\})'
    
    def replacer(match):
        old_func = match.group(1)
        lines = old_func.split('\n')
        
        # Extract function body
        body_lines = []
        in_body = False
        for line in lines:
            if '{' in line:
                in_body = True
                continue
            if '}' in line:
                break
            if in_body:
                body_lines.append(line)
        
        # Separate into dynamics, cost, and constraints
        dynamics_lines = []
        cost_lines = []
        constraint_lines = []
        
        current_section = None
        for line in body_lines:
            line_stripped = line.strip()
            
            # Detect section based on variables
            if 'f_resid' in line_stripped or '.A(' in line_stripped or '.B(' in line_stripped:
                current_section = 'dynamics'
            elif '.cost' in line_stripped or '.Q(' in line_stripped or '.R(' in line_stripped or '.q(' in line_stripped or '.r(' in line_stripped or '.H' in line_stripped:
                current_section = 'cost'
            elif '.g_val' in line_stripped or '.C(' in line_stripped or '.D(' in line_stripped:
                current_section = 'constraints'
            
            # Add to appropriate section
            if current_section == 'dynamics':
                dynamics_lines.append(line)
            elif current_section == 'cost':
                cost_lines.append(line)
            elif current_section == 'constraints':
                constraint_lines.append(line)
            elif line_stripped and not line_stripped.startswith('//'):
                # Variables/comments - add to all sections that reference them
                if any(x in line_stripped for x in ['double', 'int', 'T ']):
                    cost_lines.append(line)
        
        # Replace kp. with state. or model. appropriately
        def transform_line(line, for_section):
            result = line
            if for_section == 'dynamics':
                result = result.replace('kp.f_resid', 'model.f_resid')
                result = result.replace('kp.A', 'model.A')
                result = result.replace('kp.B', 'model.B')
                result = result.replace('kp.x', 'state.x')
                result = result.replace('kp.u', 'state.u')
                result = result.replace('kp.p', 'state.p')
            elif for_section == 'cost':
                result = result.replace('kp.cost', 'state.cost')
                result = result.replace('kp.Q', 'model.Q')
                result = result.replace('kp.R', 'model.R')
                result = result.replace('kp.H', 'model.H')
                result = result.replace('kp.q', 'model.q')
                result = result.replace('kp.r', 'model.r')
                result = result.replace('kp.x', 'state.x')
                result = result.replace('kp.u', 'state.u')
                result = result.replace('kp.p', 'state.p')
                result = result.replace('kp.lam', 'state.lam')
            elif for_section == 'constraints':
                result = result.replace('kp.g_val', 'state.g_val')
                result = result.replace('kp.C', 'model.C')
                result = result.replace('kp.D', 'model.D')
                result = result.replace('kp.x', 'state.x')
                result = result.replace('kp.u', 'state.u')
                result = result.replace('kp.p', 'state.p')
            return result
        
        # Build new functions
        new_code = '''    // NEW SPLIT ARCHITECTURE: Separate compute functions
    template<typename T>
    static void compute_dynamics(
        const StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model,
        IntegratorType /*type*/,
        double /*dt*/)
    {
'''
        for line in dynamics_lines:
            new_code += transform_line(line, 'dynamics') + '\n'
        new_code += '''    }
    
    template<typename T>
    static void compute_cost_gn(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model)
    {
'''
        for line in cost_lines:
            new_code += transform_line(line, 'cost') + '\n'
        
        # Add missing initializations
        if not any('model.R' in line for line in cost_lines):
            new_code += '        model.R.setZero();\n'
        if not any('model.q' in line for line in cost_lines):
            new_code += '        model.q.setZero();\n'
        if not any('model.r' in line for line in cost_lines):
            new_code += '        model.r.setZero();\n'
        if not any('model.H' in line for line in cost_lines):
            new_code += '        model.H.setZero();\n'
        
        new_code += '''    }
    
    template<typename T>
    static void compute_cost_exact(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model)
    {
        compute_cost_gn(state, model);
    }
    
    template<typename T>
    static void compute_constraints(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model)
    {
'''
        for line in constraint_lines:
            new_code += transform_line(line, 'constraints') + '\n'
        
        if not constraint_lines:
            new_code += '        /* No constraints */\n'
        else:
            # Add missing initialization
            if not any('model.D' in line for line in constraint_lines):
                new_code += '        model.D.setZero();\n'
        
        new_code += '''    }
    
    // Convenience wrapper for backward compatibility
    template<typename T>
    static void compute(
        StateNode<T,NX,NU,NC,NP>& state,
        ModelData<T,NX,NU,NC>& model,
        IntegratorType type,
        double dt)
    {
        compute_dynamics(state, model, type, dt);
        compute_cost_exact(state, model);
        compute_constraints(state, model);
    }'''
        
        return new_code
    
    # Remove old four wrapper functions
    content = re.sub(
        r'\s*template<typename T>\s+static void compute_cost_gn\(KnotPointV2<T,NX,NU,NC,NP>& kp\).*?\n',
        '',
        content
    )
    content = re.sub(
        r'\s*template<typename T>\s+static void compute_cost_exact\(KnotPointV2<T,NX,NU,NC,NP>& kp\).*?\n',
        '',
        content
    )
    content = re.sub(
        r'\s*template<typename T>\s+static void compute_dynamics\(KnotPointV2<T,NX,NU,NC,NP>& kp,.*?\n',
        '',
        content
    )
    content = re.sub(
        r'\s*template<typename T>\s+static void compute_constraints\(KnotPointV2<T,NX,NU,NC,NP>& kp\).*?\n',
        '',
        content
    )
    
    # Replace the compute function
    content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    return content

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_test_models.py <test_file.cpp>")
        sys.exit(1)
    
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        content = f.read()
    
    converted = convert_compute_function(content)
    
    with open(filename, 'w') as f:
        f.write(converted)
    
    print(f"Converted {filename}")
