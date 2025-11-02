#include "env_flappy/env_flappy.h"
#include "rl_dqn/rl_dqn.h"
#include "core/core.h"
#include <iostream>

int main() {
    std::cout << "Training application" << std::endl;
    core::init();
    env_flappy::init();
    rl_dqn::init();
    return 0;
}

