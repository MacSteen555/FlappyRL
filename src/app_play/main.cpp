#include "env_flappy/env_flappy.h"
#include "render_sdl/render_sdl.h"
#include "core/core.h"
#include <iostream>

int main() {
    std::cout << "Play application" << std::endl;
    core::init();
    env_flappy::init();
    render_sdl::init();
    return 0;
}

