#include "render_sdl/render_sdl.h"
#include <iostream>
#include <cmath>
#include <cstring>

#ifdef HAVE_SDL2
#include <SDL2/SDL.h>
#endif

namespace render_sdl {

Renderer::Renderer() = default;

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::initialize(const RenderConfig& config) {
#ifdef HAVE_SDL2
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL2 initialization failed: " << SDL_GetError() << std::endl;
        return false;
    }

    window_ = SDL_CreateWindow(
        "FlappyRL",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        config.window_width,
        config.window_height,
        SDL_WINDOW_SHOWN
    );

    if (!window_) {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return false;
    }

    renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer_) {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window_);
        SDL_Quit();
        return false;
    }

    config_ = config;
    
    // Calculate scale based on world height (normalized 0-1) to window
    config_.scale_x = static_cast<float>(config.window_width) * 0.8f;  // Use 80% width
    config_.scale_y = static_cast<float>(config.window_height);  // Full height for world
    
    // Allocate memory for previous keyboard state
    keyboard_state_ = SDL_GetKeyboardState(&num_keys_);
    if (num_keys_ > 0) {
        previous_keyboard_state_ = new Uint8[num_keys_];
        std::memset(previous_keyboard_state_, 0, num_keys_);
    }
    
    initialized_ = true;
    return true;
#else
    std::cerr << "SDL2 not available. Install SDL2 to enable visualization." << std::endl;
    return false;
#endif
}

void Renderer::shutdown() {
#ifdef HAVE_SDL2
    if (previous_keyboard_state_) {
        delete[] previous_keyboard_state_;
        previous_keyboard_state_ = nullptr;
    }
    if (renderer_) {
        SDL_DestroyRenderer(renderer_);
        renderer_ = nullptr;
    }
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
#endif
    initialized_ = false;
}

void Renderer::poll_events() {
#ifdef HAVE_SDL2
    // Save previous keyboard state
    if (previous_keyboard_state_ && keyboard_state_) {
        std::memcpy(previous_keyboard_state_, keyboard_state_, num_keys_);
    }
    
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            // Handle quit in should_close()
        }
    }
    // Update keyboard state
    keyboard_state_ = SDL_GetKeyboardState(nullptr);
#endif
}

bool Renderer::should_close() const {
#ifdef HAVE_SDL2
    SDL_Event e;
    if (SDL_PeepEvents(&e, 1, SDL_PEEKEVENT, SDL_QUIT, SDL_QUIT) > 0) {
        return true;
    }
    // Also check for ESC or Q key
    if (keyboard_state_) {
        if (keyboard_state_[SDL_SCANCODE_ESCAPE] || keyboard_state_[SDL_SCANCODE_Q]) {
            return true;
        }
    }
    return false;
#else
    return false;
#endif
}

bool Renderer::is_key_pressed(int key_code) const {
#ifdef HAVE_SDL2
    if (keyboard_state_) {
        return keyboard_state_[key_code] != 0;
    }
#endif
    return false;
}

bool Renderer::is_key_just_pressed(int key_code) const {
#ifdef HAVE_SDL2
    if (keyboard_state_ && previous_keyboard_state_) {
        // Key is pressed now but wasn't pressed before
        return keyboard_state_[key_code] != 0 && previous_keyboard_state_[key_code] == 0;
    }
#endif
    return false;
}

int Renderer::world_to_screen_x(float world_x) const {
    // Bird is at kBirdX (0.20), center it in view
    float offset_x = 0.20f;  // kBirdX
    float normalized_x = (world_x - offset_x) * 2.0f + 0.5f;  // Center bird
    return static_cast<int>(normalized_x * config_.scale_x);
}

int Renderer::world_to_screen_y(float world_y) const {
    // Flip Y (SDL has origin at top-left, world has origin at bottom)
    float normalized_y = 1.0f - world_y;
    return static_cast<int>(normalized_y * config_.scale_y);
}

void Renderer::render_background() {
#ifdef HAVE_SDL2
    // Sky blue background
    SDL_SetRenderDrawColor(renderer_, 135, 206, 235, 255);
    SDL_RenderClear(renderer_);
    
    // Ground (brown)
    SDL_SetRenderDrawColor(renderer_, 139, 69, 19, 255);
    SDL_Rect ground = {0, world_to_screen_y(0.0f), config_.window_width, 20};
    SDL_RenderFillRect(renderer_, &ground);
    
    // Ceiling
    SDL_SetRenderDrawColor(renderer_, 100, 149, 237, 255);
    SDL_Rect ceiling = {0, 0, config_.window_width, 20};
    SDL_RenderFillRect(renderer_, &ceiling);
#endif
}

void Renderer::render_bird(float y, float vy) {
#ifdef HAVE_SDL2
    const float kBirdX = 0.20f;
    int bird_x = world_to_screen_x(kBirdX);
    int bird_y = world_to_screen_y(y);
    
    // Bird is a yellow circle
    SDL_SetRenderDrawColor(renderer_, 255, 255, 0, 255);
    
    // Draw circle (simple approximation with filled rect)
    int bird_size = 15;
    SDL_Rect bird_rect = {
        bird_x - bird_size / 2,
        bird_y - bird_size / 2,
        bird_size,
        bird_size
    };
    SDL_RenderFillRect(renderer_, &bird_rect);
    
    // Draw a simple "beak" pointing in direction of velocity
    if (std::abs(vy) > 0.1f) {
        SDL_SetRenderDrawColor(renderer_, 255, 165, 0, 255);
        int beak_offset = (vy > 0) ? bird_size / 2 : -bird_size / 2;
        SDL_RenderDrawLine(renderer_, bird_x, bird_y, bird_x + 5, bird_y + beak_offset);
    }
#endif
}

void Renderer::render_pipe(float x, float gap_y, float pipe_width, float pipe_gap, float /* world_height */) {
#ifdef HAVE_SDL2
    int pipe_screen_x = world_to_screen_x(x);
    int pipe_screen_width = static_cast<int>(pipe_width * config_.scale_x);
    
    float gap_top = gap_y + pipe_gap * 0.5f;
    float gap_bottom = gap_y - pipe_gap * 0.5f;
    
    // Top pipe
    SDL_SetRenderDrawColor(renderer_, 34, 139, 34, 255);  // Forest green
    SDL_Rect top_pipe = {
        pipe_screen_x - pipe_screen_width / 2,
        0,
        pipe_screen_width,
        world_to_screen_y(gap_top)
    };
    SDL_RenderFillRect(renderer_, &top_pipe);
    
    // Bottom pipe
    SDL_Rect bottom_pipe = {
        pipe_screen_x - pipe_screen_width / 2,
        world_to_screen_y(gap_bottom),
        pipe_screen_width,
        config_.window_height - world_to_screen_y(gap_bottom)
    };
    SDL_RenderFillRect(renderer_, &bottom_pipe);
    
    // Pipe outline
    SDL_SetRenderDrawColor(renderer_, 0, 100, 0, 255);
    SDL_RenderDrawRect(renderer_, &top_pipe);
    SDL_RenderDrawRect(renderer_, &bottom_pipe);
#endif
}

void Renderer::render(const env_flappy::FlappyEnv& env) {
    if (!initialized_) return;

#ifdef HAVE_SDL2
    render_background();
    
    // Get current observation
    auto obs = env.observe();
    
    // Render bird
    render_bird(obs.y, obs.vy);
    
    // Render current pipe based on observation
    const auto& config = env.config();
    if (obs.dx_to_pipe < 2.0f && obs.dx_to_pipe > -0.5f) {  // Only render if pipe is nearby
        // Bird is at fixed x = 0.20, so pipe x = bird_x + dx_to_pipe
        const float kBirdX = 0.20f;
        float pipe_x = kBirdX + obs.dx_to_pipe;
        float gap_y = obs.y + obs.dy_to_gap;
        render_pipe(pipe_x, gap_y, config.pipe_width, config.pipe_gap, config.world_height);
    }
    
    present();
#endif
}

void Renderer::present() {
#ifdef HAVE_SDL2
    SDL_RenderPresent(renderer_);
#endif
}

} // namespace render_sdl
