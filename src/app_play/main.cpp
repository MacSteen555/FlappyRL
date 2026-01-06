#include "env_flappy/env_flappy.h"
#include "render_sdl/render_sdl.h"
#include "core/core.h"
#include <iostream>
#include <chrono>
#include <thread>

#ifdef HAVE_SDL2
#include <SDL2/SDL.h>
#endif

int main() {
    std::cout << "FlappyRL - Play Application" << std::endl;
    
    core::init();
    
    // Initialize SDL renderer
    render_sdl::Renderer renderer;
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize SDL renderer. Make sure SDL2 is installed." << std::endl;
        return 1;
    }
    
    // Create environment
    env_flappy::FlappyEnv env(12345);
    env.reset(12345);  // Initialize environment
    
    std::cout << "Controls:" << std::endl;
    std::cout << "  SPACE - Flap (tap, don't hold)" << std::endl;
    std::cout << "  R - Restart after game over" << std::endl;
    std::cout << "  ESC/Q - Quit" << std::endl;
    
    const float target_fps = 60.0f;
    const auto frame_time = std::chrono::milliseconds(static_cast<int>(1000.0f / target_fps));
    
    bool running = true;
    while (running) {
        // Continue running even after episode ends (can restart with R key)
        if (env.done()) {
            // Show game over state
            renderer.render(env);
            
            // Check for restart (R key) or quit
            renderer.poll_events();
#ifdef HAVE_SDL2
            if (renderer.is_key_just_pressed(SDL_SCANCODE_R)) {
                // Restart episode
                env.reset(12345);
            }
#endif
            if (renderer.should_close()) {
                running = false;
                break;
            }
            
            // Small delay to prevent 100% CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            continue;
        }
        auto frame_start = std::chrono::steady_clock::now();
        
        // Poll events
        renderer.poll_events();
        if (renderer.should_close()) {
            running = false;
            break;
        }
        
        // Keyboard input handling - only flap on key press, not while held
        env_flappy::Action action = env_flappy::Action::NO_FLAP;
        
#ifdef HAVE_SDL2
        // Check if SPACE was just pressed (not held) to flap
        if (renderer.is_key_just_pressed(SDL_SCANCODE_SPACE)) {
            action = env_flappy::Action::FLAP;
        }
#endif
        
        // Step environment with the chosen action
        env_flappy::StepResult result = env.step(action);
        
        // Render
        renderer.render(env);
        
        // Frame rate limiting
        auto frame_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
        if (elapsed < frame_time) {
            std::this_thread::sleep_for(frame_time - elapsed);
        }
        
        // Print status every 60 frames
        if (env.steps() % 60 == 0) {
            std::cout << "Step: " << env.steps() 
                      << ", Reward: " << result.reward 
                      << ", Done: " << result.done << std::endl;
        }
    }
    
    std::cout << "Episode ended after " << env.steps() << " steps" << std::endl;
    
    renderer.shutdown();
    return 0;
}
