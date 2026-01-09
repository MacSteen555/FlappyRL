#include "rl_dqn/network.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace rl_dqn {

Network::Network(const std::vector<int>& layer_sizes, std::uint64_t seed)
    : layer_sizes_(layer_sizes), rng_(static_cast<std::mt19937::result_type>(seed)) {
    
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("Network needs at least input and output layers");
    }
    
    // Initialize weights and biases
    weights_.resize(layer_sizes.size() - 1);
    biases_.resize(layer_sizes.size() - 1);
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        int fan_in = layer_sizes[i];
        int fan_out = layer_sizes[i + 1];
        
        // Initialize weights
        weights_[i].resize(fan_out);
        for (int j = 0; j < fan_out; ++j) {
            weights_[i][j].resize(fan_in);
            for (int k = 0; k < fan_in; ++k) {
                weights_[i][j][k] = xavier_init(fan_in, fan_out);
            }
        }
        
        // Initialize biases to zero
        biases_[i].resize(fan_out, 0.0f);
    }
}

float Network::relu(float x) {
    return std::max(0.0f, x);
}

float Network::relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

float Network::xavier_init(int fan_in, int fan_out) {
    // Xavier/Glorot initialization
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    return dist(rng_);
}

void Network::initialize_weights() {
    // Already done in constructor, but kept for potential re-initialization
    for (size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
        int fan_in = layer_sizes_[i];
        int fan_out = layer_sizes_[i + 1];
        
        for (int j = 0; j < fan_out; ++j) {
            for (int k = 0; k < fan_in; ++k) {
                weights_[i][j][k] = xavier_init(fan_in, fan_out);
            }
        }
    }
}

std::vector<float> Network::matvec_mult(const std::vector<std::vector<float>>& W,
                                         const std::vector<float>& x) const {
    std::vector<float> result(W.size(), 0.0f);
    for (size_t i = 0; i < W.size(); ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            result[i] += W[i][j] * x[j];
        }
    }
    return result;
}

std::vector<float> Network::forward(const std::vector<float>& input) const {
    if (static_cast<int>(input.size()) != layer_sizes_[0]) {
        throw std::invalid_argument("Input size mismatch");
    }
    
    std::vector<float> activations = input;
    
    // Forward through all layers
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        // Compute: z = W * x + b
        std::vector<float> z = matvec_mult(weights_[layer], activations);
        for (size_t i = 0; i < z.size(); ++i) {
            z[i] += biases_[layer][i];
        }
        
        // Apply activation (ReLU for hidden, linear for output)
        if (layer < weights_.size() - 1) {
            // Hidden layer: ReLU
            for (float& val : z) {
                val = relu(val);
            }
        }
        // Output layer: linear (no activation)
        
        activations = z;
    }
    
    return activations;
}

void Network::backward(const std::vector<float>& input,
                       const std::vector<float>& target_q_values,
                       const std::vector<float>& predicted_q_values,
                       std::vector<std::vector<std::vector<float>>>& weight_gradients,
                       std::vector<std::vector<float>>& bias_gradients) const {
    
    // Initialize gradients
    weight_gradients.resize(weights_.size());
    bias_gradients.resize(biases_.size());
    
    for (size_t i = 0; i < weights_.size(); ++i) {
        weight_gradients[i].resize(weights_[i].size());
        for (size_t j = 0; j < weights_[i].size(); ++j) {
            weight_gradients[i][j].resize(weights_[i][j].size(), 0.0f);
        }
        bias_gradients[i].resize(biases_[i].size(), 0.0f);
    }
    
    // Forward pass to get all activations (both pre and post activation)
    std::vector<std::vector<float>> layer_activations;  // Post-activation (input to next layer)
    std::vector<std::vector<float>> pre_activations;    // Pre-activation (before ReLU)
    std::vector<float> current_activation = input;
    layer_activations.push_back(current_activation);
    
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        std::vector<float> z = matvec_mult(weights_[layer], current_activation);
        for (size_t i = 0; i < z.size(); ++i) {
            z[i] += biases_[layer][i];
        }
        
        // Store pre-activation
        pre_activations.push_back(z);
        
        if (layer < weights_.size() - 1) {
            // Apply ReLU
            for (float& val : z) {
                val = relu(val);
            }
        }
        
        layer_activations.push_back(z);
        current_activation = z;
    }
    
    // Compute output error (MSE derivative)
    std::vector<float> output_error(predicted_q_values.size());
    for (size_t i = 0; i < predicted_q_values.size(); ++i) {
        output_error[i] = predicted_q_values[i] - target_q_values[i];
    }
    
    // Backward pass
    std::vector<float> delta = output_error;
    
    for (int layer = static_cast<int>(weights_.size()) - 1; layer >= 0; --layer) {
        const std::vector<float>& prev_activation = layer_activations[layer];
        
        // Compute gradients for this layer
        for (size_t i = 0; i < weights_[layer].size(); ++i) {
            // Bias gradient
            bias_gradients[layer][i] = delta[i];
            
            // Weight gradients
            for (size_t j = 0; j < weights_[layer][i].size(); ++j) {
                weight_gradients[layer][i][j] = delta[i] * prev_activation[j];
            }
        }
        
        // Propagate error to previous layer (if not input layer)
        if (layer > 0) {
            std::vector<float> prev_delta(prev_activation.size(), 0.0f);
            // pre_activations[layer-1] contains the pre-activation for the previous layer
            const std::vector<float>& prev_pre_activation = pre_activations[layer - 1];
            for (size_t i = 0; i < weights_[layer].size(); ++i) {
                // Apply ReLU derivative using pre-activation value
                // For hidden layers, use ReLU derivative; for output layer, derivative is 1.0
                float relu_deriv = (layer < weights_.size() - 1) ? 
                    relu_derivative(prev_pre_activation[i]) : 1.0f;
                for (size_t j = 0; j < weights_[layer][i].size(); ++j) {
                    prev_delta[j] += weights_[layer][i][j] * delta[i] * relu_deriv;
                }
            }
            delta = prev_delta;
        }
    }
}

void Network::update_weights(const std::vector<std::vector<std::vector<float>>>& weight_gradients,
                             const std::vector<std::vector<float>>& bias_gradients,
                             float learning_rate) {
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        for (size_t i = 0; i < weights_[layer].size(); ++i) {
            // Update bias
            biases_[layer][i] -= learning_rate * bias_gradients[layer][i];
            
            // Update weights
            for (size_t j = 0; j < weights_[layer][i].size(); ++j) {
                weights_[layer][i][j] -= learning_rate * weight_gradients[layer][i][j];
            }
        }
    }
}

std::vector<std::vector<std::vector<float>>> Network::get_weights() const {
    return weights_;
}

std::vector<std::vector<float>> Network::get_biases() const {
    return biases_;
}

void Network::set_weights(const std::vector<std::vector<std::vector<float>>>& weights) {
    if (weights.size() != weights_.size()) {
        throw std::invalid_argument("Weight structure mismatch");
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i].size() != weights_[i].size()) {
            throw std::invalid_argument("Weight structure mismatch");
        }
        for (size_t j = 0; j < weights[i].size(); ++j) {
            if (weights[i][j].size() != weights_[i][j].size()) {
                throw std::invalid_argument("Weight structure mismatch");
            }
        }
    }
    
    weights_ = weights;
}

void Network::set_biases(const std::vector<std::vector<float>>& biases) {
    if (biases.size() != biases_.size()) {
        throw std::invalid_argument("Bias structure mismatch");
    }
    
    for (size_t i = 0; i < biases.size(); ++i) {
        if (biases[i].size() != biases_[i].size()) {
            throw std::invalid_argument("Bias structure mismatch");
        }
    }
    
    biases_ = biases;
}

int Network::get_num_parameters() const {
    int total = 0;
    for (size_t i = 0; i < weights_.size(); ++i) {
        for (size_t j = 0; j < weights_[i].size(); ++j) {
            total += static_cast<int>(weights_[i][j].size());  // weights
        }
        total += static_cast<int>(biases_[i].size());  // biases
    }
    return total;
}

} // namespace rl_dqn

