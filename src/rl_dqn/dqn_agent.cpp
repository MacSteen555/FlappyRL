#include "rl_dqn/dqn_agent.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rl_dqn {

DQNAgent::DQNAgent(const DQNConfig& config)
    : config_(config),
      main_network_(config.layer_sizes, config.seed),
      target_network_(config.layer_sizes, config.seed + 1),
      replay_buffer_(config.replay_buffer_size, config.seed + 2),
      optimizer_(config.learning_rate, config.adam_beta1, config.adam_beta2, config.adam_epsilon),
      total_steps_(0),
      training_steps_(0),
      current_epsilon_(config.epsilon_start) {
    
    // Initialize target network with same weights as main network
    update_target_network();
}

std::vector<float> DQNAgent::observation_to_input(const env_flappy::Observation& obs) const {
    return {obs.y, obs.vy, obs.dx_to_pipe, obs.dy_to_gap};
}

env_flappy::Action DQNAgent::select_action(const env_flappy::Observation& state) {
    total_steps_++;
    
    // Update epsilon (linear decay)
    float epsilon_progress = std::min(1.0f, static_cast<float>(total_steps_) / config_.epsilon_decay_steps);
    current_epsilon_ = config_.epsilon_start + (config_.epsilon_end - config_.epsilon_start) * epsilon_progress;
    
    // Epsilon-greedy: random action with probability epsilon
    std::mt19937 rng(static_cast<std::mt19937::result_type>(total_steps_));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    if (dist(rng) < current_epsilon_) {
        // Random action
        return dist(rng) < 0.5f ? env_flappy::Action::NO_FLAP : env_flappy::Action::FLAP;
    } else {
        // Greedy action: choose action with highest Q-value
        std::vector<float> input = observation_to_input(state);
        std::vector<float> q_values = main_network_.forward(input);
        
        // Return action with highest Q-value
        if (q_values[1] > q_values[0]) {  // FLAP > NO_FLAP
            return env_flappy::Action::FLAP;
        } else {
            return env_flappy::Action::NO_FLAP;
        }
    }
}

void DQNAgent::store_experience(const env_flappy::Observation& state,
                                env_flappy::Action action,
                                float reward,
                                const env_flappy::Observation& next_state,
                                bool done) {
    Experience exp;
    exp.state = state;
    exp.action = action;
    exp.reward = reward;
    exp.next_state = next_state;
    exp.done = done;
    
    replay_buffer_.push(exp);
}

std::vector<std::vector<float>> DQNAgent::compute_targets(const std::vector<Experience>& batch) const {
    std::vector<std::vector<float>> targets;
    targets.reserve(batch.size());
    
    for (const auto& exp : batch) {
        std::vector<float> target_q_values(2, 0.0f);
        
        // Compute target Q-value for the action that was taken
        float target_q;
        if (exp.done) {
            // Terminal state: target is just the reward
            target_q = exp.reward;
        } else {
            // Non-terminal: target = reward + gamma * max Q(next_state)
            std::vector<float> next_input = observation_to_input(exp.next_state);
            std::vector<float> next_q_values = target_network_.forward(next_input);
            float max_next_q = std::max(next_q_values[0], next_q_values[1]);
            target_q = exp.reward + config_.gamma * max_next_q;
        }
        
        // Set target for the action that was taken
        int action_idx = (exp.action == env_flappy::Action::FLAP) ? 1 : 0;
        target_q_values[action_idx] = target_q;
        // The other action's target will be set to current predicted value in train() method
        
        targets.push_back(target_q_values);
    }
    
    return targets;
}

float DQNAgent::train() {
    if (!replay_buffer_.can_sample(config_.batch_size)) {
        return 0.0f;  // Not enough experiences yet
    }
    
    // Sample batch
    std::vector<Experience> batch = replay_buffer_.sample(config_.batch_size);
    
    // Compute targets
    std::vector<std::vector<float>> targets = compute_targets(batch);
    
    // Accumulate gradients (will be averaged in optimizer)
    std::vector<std::vector<std::vector<float>>> total_weight_gradients;
    std::vector<std::vector<float>> total_bias_gradients;
    
    // Initialize gradient accumulators
    auto layer_sizes = main_network_.get_layer_sizes();
    total_weight_gradients.resize(layer_sizes.size() - 1);
    total_bias_gradients.resize(layer_sizes.size() - 1);
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        int fan_out = layer_sizes[i + 1];
        int fan_in = layer_sizes[i];
        
        total_weight_gradients[i].resize(fan_out);
        for (int j = 0; j < fan_out; ++j) {
            total_weight_gradients[i][j].resize(fan_in, 0.0f);
        }
        total_bias_gradients[i].resize(fan_out, 0.0f);
    }
    
    float total_loss = 0.0f;
    
    // Process each experience in batch
    for (size_t i = 0; i < batch.size(); ++i) {
        const auto& exp = batch[i];
        
        // Get current Q-values
        std::vector<float> input = observation_to_input(exp.state);
        std::vector<float> predicted_q = main_network_.forward(input);
        
        // Set target for non-taken action to current predicted value (so gradient is zero)
        int action_idx = (exp.action == env_flappy::Action::FLAP) ? 1 : 0;
        int other_action_idx = 1 - action_idx;
        targets[i][other_action_idx] = predicted_q[other_action_idx];
        
        // Compute loss (MSE) for the action that was taken
        float q_predicted = predicted_q[action_idx];
        float q_target = targets[i][action_idx];
        float loss = (q_predicted - q_target) * (q_predicted - q_target);
        total_loss += loss;
        
        // Compute gradients
        std::vector<std::vector<std::vector<float>>> weight_gradients;
        std::vector<std::vector<float>> bias_gradients;
        
        main_network_.backward(input, targets[i], predicted_q, weight_gradients, bias_gradients);
        
        // Accumulate gradients (will average in optimizer)
        for (size_t layer = 0; layer < weight_gradients.size(); ++layer) {
            for (size_t neuron = 0; neuron < weight_gradients[layer].size(); ++neuron) {
                total_bias_gradients[layer][neuron] += bias_gradients[layer][neuron];
                for (size_t weight = 0; weight < weight_gradients[layer][neuron].size(); ++weight) {
                    total_weight_gradients[layer][neuron][weight] += 
                        weight_gradients[layer][neuron][weight];
                }
            }
        }
    }
    
    // Get current weights and biases
    auto weights = main_network_.get_weights();
    auto biases = main_network_.get_biases();
    
    // Apply Adam optimizer update
    optimizer_.update(weights, biases, total_weight_gradients, total_bias_gradients);
    
    // Set updated weights and biases back
    main_network_.set_weights(weights);
    main_network_.set_biases(biases);
    
    float avg_loss = total_loss / batch.size();
    
    training_steps_++;
    return avg_loss;
}

void DQNAgent::update_target_network() {
    auto weights = main_network_.get_weights();
    target_network_.set_weights(weights);
}

float DQNAgent::get_epsilon() const {
    return current_epsilon_;
}

std::vector<float> DQNAgent::get_q_values(const env_flappy::Observation& state) const {
    std::vector<float> input = observation_to_input(state);
    return main_network_.forward(input);
}

void DQNAgent::save_weights(const std::string& filepath) const {
    // TODO: Implement weight serialization
    (void)filepath;  // Suppress unused parameter warning
}

void DQNAgent::load_weights(const std::string& filepath) {
    // TODO: Implement weight deserialization
    (void)filepath;  // Suppress unused parameter warning
}

} // namespace rl_dqn

