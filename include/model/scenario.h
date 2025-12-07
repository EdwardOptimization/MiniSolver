#pragma once

namespace minisolver {

struct ScenarioConfig {
    static constexpr int N = 60;
    static constexpr double DT = 0.1;
    
    // Obstacle
    static constexpr double OBS_X = 12.0;
    static constexpr double OBS_Y = 0.0;
    static constexpr double OBS_RAD = 1.5;
    
    // Vehicle
    static constexpr double CAR_L = 2.5;
    static constexpr double CAR_RAD = 1.0;
    
    // Target
    static constexpr double TARGET_V = 5.0;
    
    // Weights
    static constexpr double W_POS = 1.0;
    static constexpr double W_VEL = 1.0;
    static constexpr double W_THETA = 0.1;
    static constexpr double W_ACC = 0.1;
    static constexpr double W_STEER = 1.0;
};

}

