#pragma once
#include <array>
#include <vector>
#include <memory>
#include <algorithm>
#include "minisolver/core/types.h"

namespace minisolver {

// =============================================================================
// TRAJECTORY CLASS: True Split Architecture (State/Model/Workspace)
// =============================================================================
template<typename Knot, int MAX_N>
class Trajectory {
public:
    using TrajArray = std::array<Knot, MAX_N + 1>;
    using KnotType = Knot;  // For compatibility
    
    // Extract types from Knot (assuming Knot has these typedefs or we deduce them)
    static const int NX = Knot::NX;
    static const int NU = Knot::NU;
    static const int NC = Knot::NC;
    static const int NP = Knot::NP;
    
    using State = StateNode<double, NX, NU, NC, NP>;
    using Model = ModelData<double, NX, NU, NC>;
    using Work = SolverWorkspace<double, NX, NU, NC>;
    
    // === THREE-LAYER STORAGE ===
    // 1. State: Double-buffered (Active / Candidate)
    std::array<State, MAX_N + 1> state_A;
    std::array<State, MAX_N + 1> state_B;
    State* active_state;
    State* candidate_state;
    
    // 2. ModelData: Single-buffered (read-only during Line Search/SOC)
    std::array<Model, MAX_N + 1> model_data;
    
    // 3. Workspace: Single-buffered (recomputed as needed)
    std::array<Work, MAX_N + 1> workspace;
    
    int N; // Current valid horizon

    Trajectory(int initial_N) : N(initial_N) {
        active_state = &state_A[0];
        candidate_state = &state_B[0];
        
        // Initialize all memory
        for(int i = 0; i <= MAX_N; ++i) {
            state_A[i].initialize_defaults();
            state_B[i].initialize_defaults();
            model_data[i].set_zero();
            workspace[i].set_zero();
        }
    }
    
    void resize(int new_n) {
        if (new_n > MAX_N) return;
        N = new_n;
    }
    
    // === ACCESSORS FOR THREE-LAYER ARCHITECTURE ===
    State* get_active_state() { return active_state; }
    const State* get_active_state() const { return active_state; }
    
    State* get_candidate_state() { return candidate_state; }
    const State* get_candidate_state() const { return candidate_state; }
    
    Model* get_model_data() { return &model_data[0]; }
    const Model* get_model_data() const { return &model_data[0]; }
    
    Work* get_workspace() { return &workspace[0]; }
    const Work* get_workspace() const { return &workspace[0]; }
    
    // Legacy accessor for backward compatibility (returns active state)
    State& operator[](int k) { return active_state[k]; }
    const State& operator[](int k) const { return active_state[k]; }
    
    // Legacy array accessors for backward compatibility
    // Note: These return State*, but old code expects TrajArray&
    // We'll create temporary compatibility wrappers
    State* active() { return active_state; }
    const State* active() const { return active_state; }
    
    State* candidate() { return candidate_state; }
    const State* candidate() const { return candidate_state; }
    
    // Swap active and candidate states
    void swap() {
        std::swap(active_state, candidate_state);
    }
    
    // === LIGHTWEIGHT PREPARE_CANDIDATE (KEY OPTIMIZATION) ===
    // Only copy State (vectors), NOT matrices!
    // This is the 98% bandwidth saving mentioned in the original scheme
    void prepare_candidate() {
        for(int k = 0; k <= N; ++k) {
            candidate_state[k].copy_from(active_state[k]);
        }
    }
    // Shifts the trajectory for Warm Start (MPC)
    void shift() {
        for(int k = 0; k < N; ++k) {
            active_state[k].x = active_state[k+1].x;
            active_state[k].u = active_state[k+1].u;
            active_state[k].s = active_state[k+1].s;
            active_state[k].lam = active_state[k+1].lam;
            active_state[k].soft_s = active_state[k+1].soft_s;
            active_state[k].soft_dual = active_state[k+1].soft_dual;
            active_state[k].p = active_state[k+1].p;
        }
        // Duplicate last step
        active_state[N].x = active_state[N-1].x;
        active_state[N].u = active_state[N-1].u;
        active_state[N].s = active_state[N-1].s;
        active_state[N].lam = active_state[N-1].lam;
        active_state[N].soft_s = active_state[N-1].soft_s;
        active_state[N].soft_dual = active_state[N-1].soft_dual;
    }
    
    // Reset trajectory data
    void reset() {
        for(int i = 0; i <= MAX_N; ++i) {
            state_A[i].set_zero();
            state_A[i].initialize_defaults();
            state_B[i].set_zero();
            state_B[i].initialize_defaults();
            model_data[i].set_zero();
            workspace[i].set_zero();
        }
    }
    
    int size() const { return N + 1; }
};

}

