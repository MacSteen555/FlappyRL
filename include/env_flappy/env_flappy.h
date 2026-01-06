#ifndef ENV_FLAPPY_H
#define ENV_FLAPPY_H

#include <cstdint>
#include <array>
#include <vector>
#include <random>

namespace env_flappy {
    
    enum class Action : uint8_t {
        FLAP = 1,
        NO_FLAP = 0
    };  

    struct Observation {
        float y = 0.0f;
        float vy = 0.0f;
        float dx_to_pipe = 0.0f;
        float dy_to_gap = 0.0f;
    };

    struct StepResult {
        Observation observation;
        float reward = 0.0f;
        bool done = false;
    };

    struct Config {
        float world_height = 1.0f;
        float pipe_width = 0.1f;
        float pipe_gap = 0.25f;
        float pipe_spacing = 0.60f;    // distance from one pipe center to next
        float pipe_speed   = 0.50f;    // scroll speed (units per second)
        float dt           = 1.0f/60.0f;

        // Physics
        float gravity      = -2.0f;    // downward accel (neg)
        float flap_impulse =  0.60f;   // instantaneous vy += impulse (balanced value)
        float term_vy      = -3.0f;    // clamp max downward speed
        float max_vy        =  2.5f;   // clamp max upward speed (increased slightly)

        // Rewards
        float r_pass  =  1.0f;         // crossing pipe centerline
        float r_death = -1.0f;         // on terminal collision
        float r_step  =  0.0f;         // optional shaping per step (e.g., -0.001f)

        // Random gap sampling (center of the gap between pipes [min, max])
        float gap_y_min = 0.30f;
        float gap_y_max = 0.70f;
    };


    class FlappyEnv {
        public:

            explicit FlappyEnv(std::uint64_t seed, const Config& config = Config())
                : config_(config), rng_(static_cast<std::mt19937::result_type>(seed)) { reset(seed); }
            
            Observation reset(std::uint64_t seed);
            StepResult   step(Action action);
            Observation  observe() const;
            
            bool done()  const noexcept { return done_; }
            int  steps() const noexcept { return steps_; }
            const Config& config() const noexcept { return config_; }

#ifdef FLAPPY_ENV_TESTING
            // Test-only hooks (compile only in tests)
            void _set_bird(float y, float vy) { y_ = y; vy_ = vy; }
            void _set_current_pipe(float x, float gap_y) {
                if (current_pipe_idx_ < pipes_.size()) {
                    pipes_[current_pipe_idx_] = {x, gap_y};
                }
            }
#endif

        private:
            // constants
            static constexpr float kBirdX = 0.20f;
            
            // config & rng
            Config config_;
            std::mt19937 rng_;
            std::uniform_real_distribution<float> uni01_{0.0f, 1.0f};
            
            // bird
            float y_  = 0.5f;
            float vy_ = 0.0f;
            
            // pipes
            struct Pipe { float x; float gap_y; };
            std::vector<Pipe> pipes_;
            std::size_t current_pipe_idx_ = 0;
            
            // episode
            bool done_ = false;
            bool passed_flag_ = false; // reset when a new pipe becomes "current"
            int  steps_ = 0;
            
            // helpers
            void         add_pipe(float x_after);
            bool         check_collision() const;
            bool         passed_pipe() const;       // uses kBirdX vs current pipe center
            Observation  compute_observation() const;
            float        sample_gap_center();       // uses rng_ and uni01_
        };

}

#endif // ENV_FLAPPY_H

