#include <gtest/gtest.h>
#include <new>
#include <atomic>
#include <iostream>
#include "minisolver/solver/solver.h"
#include "../examples/01_car_tutorial/generated/car_model.h"

// --- Memory Instrumentation ---
std::atomic<bool> g_memory_check_active(false);
std::atomic<int> g_allocation_count(0);

// Weak linkage allows overriding in the executable
void* operator new(size_t size) {
    if (g_memory_check_active) {
        g_allocation_count++;
    }
    void* p = malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void* operator new[](size_t size) {
    if (g_memory_check_active) {
        g_allocation_count++;
    }
    void* p = malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete(void* p) noexcept {
    free(p);
}

void operator delete[](void* p) noexcept {
    free(p);
}

void operator delete(void* p, size_t) noexcept {
    free(p);
}

void operator delete[](void* p, size_t) noexcept {
    free(p);
}

using namespace minisolver;

TEST(MemoryTest, ZeroMalloc_Compliance_Test) {
    // 1. Setup Solver
    int N = 10;
    SolverConfig config;
    config.print_level = PrintLevel::NONE; // Disable logging to avoid allocations
    config.max_iters = 5;
    config.enable_profiling = false;
    
    // Ensure we use Custom Matrix if possible, or verify Eigen doesn't allocate for fixed size
    // Eigen fixed-size matrices generally don't allocate on heap.
    
    MiniSolver<CarModel, 20> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);
    
    // Warmup / Pre-allocation (if any lazy init exists)
    // The constructor should have done everything.
    
    // 2. Start Monitoring
    g_allocation_count = 0;
    g_memory_check_active = true;
    
    // 3. Run Solve
    solver.solve();
    
    // 4. Stop Monitoring
    g_memory_check_active = false;
    
    // 5. Verify
    // Note: If using Eigen, verify EIGEN_NO_MALLOC is respected or fixed-size used.
    // If using std::vector inside solver (e.g. for resizing), it might fail.
    // MiniSolver uses std::vector for maps, but those are only read during solve if lookup used.
    // solve() itself shouldn't look up names if we don't call set_param by name inside loop.
    // The loop uses direct access.
    
    EXPECT_EQ(g_allocation_count, 0) << "Detected " << g_allocation_count << " heap allocations during solve() loop!";
}

