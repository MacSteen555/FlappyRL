#include "env_flappy/env_flappy.h"
#include "rl_dqn/rl_dqn.h"
#include "core/core.h"
#include <iostream>

int main() {
    std::cout << "FlappyRL - Training Application" << std::endl;
    core::init();
    rl_dqn::init();
    
    // Example: create and test environment
    env_flappy::FlappyEnv env(12345);
    env_flappy::Observation obs = env.reset(12345);
    std::cout << "Environment initialized. Initial y: " << obs.y << std::endl;
    
    // TODO: Implement DQN training loop here
    std::cout << "Training loop not yet implemented." << std::endl;
    
    return 0;
}

