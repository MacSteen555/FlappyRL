#include "env_flappy/env_flappy.h"
#include "render_sdl/render_sdl.h"
#include "rl_dqn/dqn_agent.h"
#include "core/core.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#ifdef HAVE_SDL2
#include <SDL2/SDL.h>
#endif

namespace {

void print_usage() {
    std::cout << "Usage: app_play [--agent CHECKPOINT] [--seed N]\n"
                 "  (no --agent)  you play: SPACE to flap\n"
                 "  --agent PATH  watch a trained DQN play\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string agent_path;
    std::uint64_t seed = 12345;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--agent") && i + 1 < argc) agent_path = argv[++i];
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else { print_usage(); return 1; }
    }

    std::cout << "FlappyRL - Play Application" << std::endl;
    core::init();

    // Load the agent before opening a window so a bad path fails fast.
    std::unique_ptr<rl_dqn::DQNAgent> agent;
    if (!agent_path.empty()) {
        try {
            agent = std::make_unique<rl_dqn::DQNAgent>();
            agent->load_weights(agent_path);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load agent: " << e.what() << std::endl;
            return 1;
        }
        std::cout << "Agent loaded from " << agent_path << std::endl;
    }

    render_sdl::Renderer renderer;
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize SDL renderer. Make sure SDL2 is installed." << std::endl;
        return 1;
    }

    env_flappy::FlappyEnv env(seed);
    env_flappy::Observation obs = env.reset(seed);

    std::cout << "Controls:\n";
    if (!agent) std::cout << "  SPACE - Flap (tap, don't hold)\n";
    std::cout << "  R     - Restart after game over (new pipe layout)\n"
                 "  ESC/Q - Quit" << std::endl;

    const auto frame_time = std::chrono::microseconds(1000000 / 60);
    int pipes = 0;
    bool reported = false;

    bool running = true;
    while (running) {
        auto frame_start = std::chrono::steady_clock::now();

        renderer.poll_events();
        if (renderer.should_close()) break;

        if (env.done()) {
            if (!reported) {
                std::cout << "Game over: " << pipes << " pipes in " << env.steps()
                          << " steps (seed " << seed << ")" << std::endl;
                reported = true;
            }
#ifdef HAVE_SDL2
            if (renderer.is_key_just_pressed(SDL_SCANCODE_R)) {
                ++seed;
                obs = env.reset(seed);
                pipes = 0;
                reported = false;
            }
#endif
        } else {
            env_flappy::Action action = env_flappy::Action::NO_FLAP;
            if (agent) {
                action = agent->greedy_action(obs);
            }
#ifdef HAVE_SDL2
            else if (renderer.is_key_just_pressed(SDL_SCANCODE_SPACE)) {
                action = env_flappy::Action::FLAP;
            }
#endif
            env_flappy::StepResult result = env.step(action);
            obs = result.observation;
            if (result.reward > 0.0f) {
                ++pipes;
                std::cout << "Pipes: " << pipes << std::endl;
            }
        }

        renderer.render(env);

        auto elapsed = std::chrono::steady_clock::now() - frame_start;
        if (elapsed < frame_time) {
            std::this_thread::sleep_for(frame_time - elapsed);
        }
    }

    renderer.shutdown();
    return 0;
}
