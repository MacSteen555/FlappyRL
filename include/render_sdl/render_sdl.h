#ifndef RENDER_SDL_H
#define RENDER_SDL_H

#include "env_flappy/env_flappy.h"
#include <cstdint>

#ifdef HAVE_SDL2
#include <SDL2/SDL.h>
#else
// Forward declarations when SDL2 is not available
typedef void SDL_Window;
typedef void SDL_Renderer;
#endif

namespace render_sdl {

struct RenderConfig {
    int window_width = 800;
    int window_height = 600;
    float scale_x = 1.0f;  // world units to pixels
    float scale_y = 1.0f;
};

class Renderer {
public:
    Renderer();
    ~Renderer();

    bool initialize(const RenderConfig& config = RenderConfig());
    void shutdown();

    void render(const env_flappy::FlappyEnv& env);
    void present();
    
    bool should_close() const;
    void poll_events();
    
    // Check if a key is currently pressed (held down)
    bool is_key_pressed(int key_code) const;
    
    // Check if a key was just pressed this frame (not held)
    bool is_key_just_pressed(int key_code) const;
    
    // Key codes (SDL scancodes)
    static constexpr int KEY_SPACE = 44;
    static constexpr int KEY_ESCAPE = 41;
    static constexpr int KEY_Q = 20;

    bool is_initialized() const { return initialized_; }

private:
    bool initialized_ = false;
    RenderConfig config_;

#ifdef HAVE_SDL2
    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
    const Uint8* keyboard_state_ = nullptr;
    Uint8* previous_keyboard_state_ = nullptr;
    int num_keys_ = 0;
#endif

    void render_bird(float y, float vy);
    void render_pipe(float x, float gap_y, float pipe_width, float pipe_gap, float world_height);
    void render_background();
    
    // Convert world coordinates to screen coordinates
    int world_to_screen_x(float world_x) const;
    int world_to_screen_y(float world_y) const;
};

} // namespace render_sdl

#endif // RENDER_SDL_H
