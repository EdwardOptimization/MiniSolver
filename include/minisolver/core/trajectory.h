#pragma once
#include <array>
#include <vector>
#include <memory>
#include <algorithm>
#include "minisolver/core/types.h"

namespace minisolver {

template<typename Knot, int MAX_N>
class Trajectory {
public:
    using TrajArray = std::array<Knot, MAX_N + 1>;
    using StateType = typename Knot::StateType;
    using MatricesType = typename Knot::MatricesType;
    
    // Double Buffering
    // We use unique_ptr to manage memory ownership clearly, 
    // but raw pointers for swapping to avoid overhead.
    // Actually, std::array is on stack/inline. MiniSolver has them as members.
    // To allow Trajectory class to own data, we can store them here.
    
    TrajArray memory_A;
    TrajArray memory_B;
    
    TrajArray* active_ptr;
    TrajArray* candidate_ptr;
    
    int N; // Current valid horizon

    Trajectory(int initial_N) : N(initial_N) {
        active_ptr = &memory_A;
        candidate_ptr = &memory_B;
        
        // Initialize
        for(auto& kp : *active_ptr) kp.initialize_defaults();
        for(auto& kp : *candidate_ptr) kp.initialize_defaults();
    }
    
    void resize(int new_n) {
        if (new_n > MAX_N) return; // Error handling needed
        N = new_n;
    }
    
    // Accessors
    Knot& operator[](int k) { return (*active_ptr)[k]; }
    const Knot& operator[](int k) const { return (*active_ptr)[k]; }
    
    TrajArray& active() { return *active_ptr; }
    TrajArray& candidate() { return *candidate_ptr; }
    const TrajArray& active() const { return *active_ptr; }
    
    void swap() {
        std::swap(active_ptr, candidate_ptr);
    }
    
    // Lightweight copy: only vectors and scalars (KnotState base).
    // Skips large matrices (KnotMatrices) which are recomputed each iteration.
    // Use this for Line Search candidate preparation.
    void prepare_candidate() {
        auto& src = *active_ptr;
        auto& dst = *candidate_ptr;
        for (int k = 0; k <= N; ++k) {
            dst[k].copy_state_from(src[k]);
        }
    }
    
    // Full copy: copies the entire KnotPoint (state + matrices).
    // Use this when the candidate needs complete data (e.g., Mehrotra predictor,
    // Iterative Refinement backup) where the linear solver requires derivatives.
    void prepare_candidate_full() {
        auto& src = *active_ptr;
        auto& dst = *candidate_ptr;
        for (int k = 0; k <= N; ++k) {
            dst[k] = src[k];
        }
    }
    // Shifts the trajectory for Warm Start (MPC)
    // Moves primal/dual variables: x[k] <- x[k+1], u[k] <- u[k+1], etc.
    // Parameters (p) are also shifted; user should update them externally if time-dependent.
    // Duplicates the last step for the terminal node.
    void shift() {
        auto& traj = *active_ptr;
        for(int k=0; k < N; ++k) {
            traj[k].x = traj[k+1].x;
            traj[k].u = traj[k+1].u;
            traj[k].s = traj[k+1].s;
            traj[k].lam = traj[k+1].lam;
            traj[k].soft_s = traj[k+1].soft_s;
            traj[k].soft_dual = traj[k+1].soft_dual;
            traj[k].p = traj[k+1].p;
        }
        // Duplicate last step (safe approximation; ideally integrate dynamics)
        traj[N].x = traj[N-1].x;
        traj[N].u = traj[N-1].u;
        traj[N].s = traj[N-1].s;
        traj[N].lam = traj[N-1].lam;
        traj[N].soft_s = traj[N-1].soft_s;
        traj[N].soft_dual = traj[N-1].soft_dual;
        // Do not duplicate last parameter — let user set new value
    }
    
    // Reset trajectory data to initial construction state (Zero x/u/p, Default s/lam)
    void reset() {
        for(auto& kp : *active_ptr) { 
            kp.set_zero(); 
            kp.initialize_defaults(); 
        }
        for(auto& kp : *candidate_ptr) { 
            kp.set_zero(); 
            kp.initialize_defaults(); 
        }
    }
    
    int size() const { return N + 1; }
};

}

