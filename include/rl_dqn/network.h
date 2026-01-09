#ifndef RL_DQN_NETWORK_H
#define RL_DQN_NETWORK_H

#include <vector>
#include <cstdint>
#include <random>

namespace rl_dqn {

// Simple feedforward neural network for DQN
class Network {
public:
    // Constructor: specify layer sizes (e.g., {4, 128, 128, 2})
    explicit Network(const std::vector<int>& layer_sizes, std::uint64_t seed = 12345);

    // Forward pass: compute output from input
    std::vector<float> forward(const std::vector<float>& input) const;

    // Backward pass: compute gradients given loss
    // Returns gradients for each layer's weights and biases
    void backward(const std::vector<float>& input,
                  const std::vector<float>& target_q_values,
                  const std::vector<float>& predicted_q_values,
                  std::vector<std::vector<std::vector<float>>>& weight_gradients,
                  std::vector<std::vector<float>>& bias_gradients) const;

    // Update weights and biases using gradients (called by optimizer)
    void update_weights(const std::vector<std::vector<std::vector<float>>>& weight_gradients,
                        const std::vector<std::vector<float>>& bias_gradients,
                        float learning_rate);

    // Get current weights (for copying to target network)
    std::vector<std::vector<std::vector<float>>> get_weights() const;
    std::vector<std::vector<float>> get_biases() const;

    // Set weights (for target network updates)
    void set_weights(const std::vector<std::vector<std::vector<float>>>& weights);
    void set_biases(const std::vector<std::vector<float>>& biases);

    // Get layer sizes
    const std::vector<int>& get_layer_sizes() const { return layer_sizes_; }

    // Get number of parameters
    int get_num_parameters() const;

private:
    std::vector<int> layer_sizes_;
    std::vector<std::vector<std::vector<float>>> weights_;  // [layer][neuron][weight]
    std::vector<std::vector<float>> biases_;                // [layer][neuron]
    
    mutable std::mt19937 rng_;
    
    // Activation functions
    static float relu(float x);
    static float relu_derivative(float x);
    
    // Weight initialization (Xavier/He)
    void initialize_weights();
    float xavier_init(int fan_in, int fan_out);
    
    // Helper: matrix-vector multiplication
    std::vector<float> matvec_mult(const std::vector<std::vector<float>>& W,
                                    const std::vector<float>& x) const;
};

} // namespace rl_dqn

#endif // RL_DQN_NETWORK_H

