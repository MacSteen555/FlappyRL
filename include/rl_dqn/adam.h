#ifndef RL_DQN_ADAM_H
#define RL_DQN_ADAM_H

#include <vector>
#include <cstdint>

namespace rl_dqn {

// Adam optimizer for neural network training
class AdamOptimizer {
public:
    AdamOptimizer(float learning_rate = 0.001f,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float epsilon = 1e-8f);
    
    // Apply Adam update to network weights/biases
    // Modifies weights and biases in place using gradients
    void update(std::vector<std::vector<std::vector<float>>>& weights,
                std::vector<std::vector<float>>& biases,
                const std::vector<std::vector<std::vector<float>>>& weight_gradients,
                const std::vector<std::vector<float>>& bias_gradients);
    
    // Reset optimizer state (useful for new networks)
    void reset();
    
    // Get current step count
    int get_step() const { return step_; }

private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int step_;
    
    // Momentum and velocity for weights and biases
    std::vector<std::vector<std::vector<float>>> m_weights_;  // First moment
    std::vector<std::vector<std::vector<float>>> v_weights_;   // Second moment
    std::vector<std::vector<float>> m_biases_;
    std::vector<std::vector<float>> v_biases_;
    
    // Initialize optimizer state for given network structure
    void initialize_state(const std::vector<std::vector<std::vector<float>>>& weights,
                         const std::vector<std::vector<float>>& biases);
};

} // namespace rl_dqn

#endif // RL_DQN_ADAM_H

