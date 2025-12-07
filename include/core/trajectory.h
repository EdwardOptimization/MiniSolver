#pragma once
#include <array>
#include <vector>
#include <memory>
#include <algorithm>
#include "core/types.h"

namespace minisolver {

template<typename Knot, int MAX_N>
class Trajectory {
public:
    using TrajArray = std::array<Knot, MAX_N + 1>;
    
    // Double Buffering
    // We use unique_ptr to manage memory ownership clearly, 
    // but raw pointers for swapping to avoid overhead.
    // Actually, std::array is on stack/inline. PDIPMSolver has them as members.
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
    
    // Helper to copy active to candidate (for trial steps)
    void prepare_candidate() {
        // Deep copy needed for base state?
        // In Line Search, we update candidate based on active.
        // x_cand = x_act + alpha * dx_act.
        // So we don't necessarily need full copy if we write all fields.
        // But for safety (params, etc.), copy is good.
        // Optimization: Only copy params once?
        *candidate_ptr = *active_ptr; 
    }
    
    int size() const { return N + 1; }
};

}

