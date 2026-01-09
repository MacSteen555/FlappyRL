#ifndef RL_DQN_REPLAY_BUFFER_H
#define RL_DQN_REPLAY_BUFFER_H

#include "env_flappy/env_flappy.h"
#include <vector>
#include <cstdint>
#include <random>
#include <cstddef>

namespace rl_dqn {

// Experience tuple: (state, action, reward, next_state, done)
struct Experience {
    env_flappy::Observation state;
    env_flappy::Action action;
    float reward;
    env_flappy::Observation next_state;
    bool done;
};

// Replay buffer for DQN experience storage
class ReplayBuffer {
public:
    ReplayBuffer(std::size_t capacity, std::uint64_t seed = 12345);
    
    // Add experience to buffer
    void push(const Experience& experience);
    
    // Sample a random batch of experiences
    std::vector<Experience> sample(std::size_t batch_size) const;
    
    // Check if we have enough samples for training
    bool can_sample(std::size_t batch_size) const;
    
    // Get current size
    std::size_t size() const { return experiences_.size(); }
    
    // Get capacity
    std::size_t capacity() const { return capacity_; }
    
    // Clear buffer
    void clear();

private:
    std::vector<Experience> experiences_;
    std::size_t capacity_;
    std::size_t write_index_;  // Current write position for circular buffer
    mutable std::mt19937 rng_;
};

} // namespace rl_dqn

#endif // RL_DQN_REPLAY_BUFFER_H


