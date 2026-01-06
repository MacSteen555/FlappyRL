#include "env_flappy/env_flappy.h"
#include <algorithm>
#include <cmath>

namespace env_flappy {

// Helper: Sample gap center from uniform distribution
float FlappyEnv::sample_gap_center() {
    float t = uni01_(rng_);
    return config_.gap_y_min + t * (config_.gap_y_max - config_.gap_y_min);
}

// Helper: Add a new pipe at x_after position
void FlappyEnv::add_pipe(float x_after) {
    Pipe pipe;
    pipe.x = x_after;
    pipe.gap_y = sample_gap_center();
    pipes_.push_back(pipe);
}

// Helper: Check collision with pipes, ground, or ceiling
bool FlappyEnv::check_collision() const {
    // Ground/ceiling collision
    if (y_ <= 0.0f || y_ >= config_.world_height) {
        return true;
    }

    // Pipe collision (only check current pipe)
    if (current_pipe_idx_ < pipes_.size()) {
        
        const Pipe& pipe = pipes_[current_pipe_idx_];
        float pipe_left = pipe.x - config_.pipe_width * 0.5f;
        float pipe_right = pipe.x + config_.pipe_width * 0.5f;

        // Bird is a point at (kBirdX, y_)
        // Check if bird is within pipe's horizontal bounds
        if (kBirdX >= pipe_left && kBirdX <= pipe_right) {
            float gap_top = pipe.gap_y + config_.pipe_gap * 0.5f;
            float gap_bottom = pipe.gap_y - config_.pipe_gap * 0.5f;

            // Collision if bird is outside the gap
            if (y_ <= gap_bottom || y_ >= gap_top) {
                return true;
            }
        }
    }

    return false;
}

// Helper: Check if bird has passed the pipe centerline
bool FlappyEnv::passed_pipe() const {
    if (current_pipe_idx_ >= pipes_.size() || passed_flag_) {
        return false;
    }

    const Pipe& pipe = pipes_[current_pipe_idx_];
    // Passed if bird X is past pipe center
    return kBirdX > pipe.x;
}

// Helper: Compute observation vector [y, vy, dx_to_pipe, dy_to_gap]
Observation FlappyEnv::compute_observation() const {
    Observation obs;
    obs.y = y_;
    obs.vy = vy_;

    if (current_pipe_idx_ < pipes_.size()) {
        const Pipe& p = pipes_[current_pipe_idx_];
        obs.dx_to_pipe = p.x - kBirdX;
        obs.dy_to_gap = p.gap_y - y_;
    } 
    else {
        // Fallback if no pipes (shouldn't happen): "far away"
        obs.dx_to_pipe = 1.0f;
        obs.dy_to_gap = 0.0f;
    }

    return obs;
}

// Reset the environment to start a new episode
Observation FlappyEnv::reset(std::uint64_t seed) {
    // Re-seed RNG for determinism
    rng_.seed(static_cast<std::mt19937::result_type>(seed));

    // Reset bird state
    y_ = 0.5f * config_.world_height;  // use config for determinism
    vy_ = 0.0f;

    // Clear pipes and create initial setup
    pipes_.clear();
    pipes_.reserve(8);  // pre-allocate for fewer allocations
    current_pipe_idx_ = 0;
    passed_flag_ = false;
    done_ = false;
    steps_ = 0;

    // Add first pipe at x = 1.0 (or further to give bird some space)
    add_pipe(1.0f);

    // Keep adding pipes ahead
    while (pipes_.back().x < 3.0f) {
        add_pipe(pipes_.back().x + config_.pipe_spacing);
    }

    return observe();
}

// Get current observation without stepping
Observation FlappyEnv::observe() const {
    return compute_observation();
}

// Step the environment: action -> physics -> scroll/respawn -> collisions -> rewards
StepResult FlappyEnv::step(Action action) {
    StepResult result;
    result.done = false;
    result.reward = config_.r_step;  // Base step reward

    if (done_) {
        result.observation = observe();
        result.done = true;
        return result;
    }

    steps_++;

    // 1) Action -> physics: Apply flap if requested
    if (action == Action::FLAP) {
        vy_ += config_.flap_impulse;
    }

    // 2) Physics: Apply gravity and clamp velocity
    vy_ += config_.gravity * config_.dt;
    if (vy_ < config_.term_vy) {
        vy_ = config_.term_vy;
    }
    if (vy_ > config_.max_vy) {
        vy_ = config_.max_vy;
    }

    // Update bird position
    y_ += vy_ * config_.dt;

    // 3) Scroll pipes left and manage pipe lifecycle
    float scroll_distance = config_.pipe_speed * config_.dt;
    
    // Scroll all pipes left
    for (auto& pipe : pipes_) {
        pipe.x -= scroll_distance;
    }

    // Remove pipes that are fully off-screen on the left
    std::size_t removed_count = 0;
    while (!pipes_.empty() &&
           pipes_[0].x + 0.5f * config_.pipe_width < 0.0f) {
        pipes_.erase(pipes_.begin());
        ++removed_count;
    }
    if (current_pipe_idx_ >= removed_count) {
        current_pipe_idx_ -= removed_count;
    } 
    else {
        current_pipe_idx_ = 0;  // Reset if we removed beyond current
    }

    // Update current pipe index (first pipe ahead of or at bird)
    while (current_pipe_idx_ < pipes_.size() &&
           pipes_[current_pipe_idx_].x + 0.5f * config_.pipe_width < kBirdX) {
        passed_flag_ = false;  // next pipe becomes current; re-arm pass
        if (current_pipe_idx_ + 1 < pipes_.size()) {
            ++current_pipe_idx_;
        } 
        else {
            break;
        }
    }

    // Add new pipes as needed to keep ahead
    float furthest_x = pipes_.empty() ? 0.0f : pipes_.back().x;
    while (furthest_x < 3.0f) {
        add_pipe(furthest_x + config_.pipe_spacing);
        furthest_x = pipes_.back().x;
    }

    // 4) Check collisions
    if (check_collision()) {
        done_ = true;
        result.done = true;
        result.reward = config_.r_death;
    }

    // 5) Check if passed pipe (reward)
    if (!done_ && passed_pipe()) {
        result.reward += config_.r_pass;
        passed_flag_ = true;
    }

    result.observation = observe();
    return result;
}

} // namespace env_flappy
