#ifndef RL_DQN_DQN_AGENT_H
#define RL_DQN_DQN_AGENT_H

#include "rl_dqn/network.h"
#include "rl_dqn/replay_buffer.h"
#include "rl_dqn/adam.h"
#include "env_flappy/env_flappy.h"
#include <cstdint>
#include <cstddef>

namespace rl_dqn {

// DQN Agent configuration
struct DQNConfig {
    // Network architecture
    std::vector<int> layer_sizes = {4, 128, 128, 2};  // input, hidden, hidden, output
    
    // Training hyperparameters
    float learning_rate = 0.0001f;
    float gamma = 0.99f;  // discount factor
    float epsilon_start = 1.0f;
    float epsilon_end = 0.01f;
    int epsilon_decay_steps = 10000;
    
    // Replay buffer
    std::size_t replay_buffer_size = 10000;
    std::size_t batch_size = 32;
    
    // Training schedule
    int train_frequency = 4;  // train every N steps
    int target_update_frequency = 100;  // update target network every N steps
    
    // Adam optimizer
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_epsilon = 1e-8f;
    
    // Random seed
    std::uint64_t seed = 12345;
};

// DQN Agent
class DQNAgent {
public:
    explicit DQNAgent(const DQNConfig& config = DQNConfig());
    
    // Select action using epsilon-greedy policy
    env_flappy::Action select_action(const env_flappy::Observation& state);
    
    // Store experience in replay buffer
    void store_experience(const env_flappy::Observation& state,
                         env_flappy::Action action,
                         float reward,
                         const env_flappy::Observation& next_state,
                         bool done);
    
    // Train the network on a batch from replay buffer
    float train();
    
    // Update target network (copy weights from main network)
    void update_target_network();
    
    // Get current epsilon (for logging)
    float get_epsilon() const;
    
    // Get Q-values for a state (for debugging)
    std::vector<float> get_q_values(const env_flappy::Observation& state) const;
    
    // Save/load network weights (for model persistence)
    void save_weights(const std::string& filepath) const;
    void load_weights(const std::string& filepath);
    
    // Get training statistics
    int get_training_steps() const { return training_steps_; }
    int get_total_steps() const { return total_steps_; }

private:
    DQNConfig config_;
    
    // Networks
    Network main_network_;
    Network target_network_;
    
    // Replay buffer
    ReplayBuffer replay_buffer_;
    
    // Optimizer
    AdamOptimizer optimizer_;
    
    // Training state
    int total_steps_;
    int training_steps_;
    float current_epsilon_;
    
    // Convert observation to network input
    std::vector<float> observation_to_input(const env_flappy::Observation& obs) const;
    
    // Compute target Q-values for a batch
    std::vector<std::vector<float>> compute_targets(const std::vector<Experience>& batch) const;
};

} // namespace rl_dqn

#endif // RL_DQN_DQN_AGENT_H


