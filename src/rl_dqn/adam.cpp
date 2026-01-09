#include "rl_dqn/adam.h"
#include <cmath>

namespace rl_dqn {

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), step_(0) {
}

void AdamOptimizer::initialize_state(const std::vector<std::vector<std::vector<float>>>& weights,
                                      const std::vector<std::vector<float>>& biases) {
    // Initialize momentum and velocity to match network structure
    m_weights_.resize(weights.size());
    v_weights_.resize(weights.size());
    m_biases_.resize(biases.size());
    v_biases_.resize(biases.size());
    
    for (size_t i = 0; i < weights.size(); ++i) {
        m_weights_[i].resize(weights[i].size());
        v_weights_[i].resize(weights[i].size());
        for (size_t j = 0; j < weights[i].size(); ++j) {
            m_weights_[i][j].resize(weights[i][j].size(), 0.0f);
            v_weights_[i][j].resize(weights[i][j].size(), 0.0f);
        }
        m_biases_[i].resize(biases[i].size(), 0.0f);
        v_biases_[i].resize(biases[i].size(), 0.0f);
    }
}

void AdamOptimizer::update(std::vector<std::vector<std::vector<float>>>& weights,
                           std::vector<std::vector<float>>& biases,
                           const std::vector<std::vector<std::vector<float>>>& weight_gradients,
                           const std::vector<std::vector<float>>& bias_gradients) {
    
    // Initialize state on first update
    if (step_ == 0) {
        initialize_state(weights, biases);
    }
    
    step_++;
    
    // Bias correction factors
    float bias_correction1 = 1.0f - std::pow(beta1_, step_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_);
    
    // Update weights
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            // Update bias
            float g_bias = bias_gradients[layer][neuron];
            m_biases_[layer][neuron] = beta1_ * m_biases_[layer][neuron] + (1.0f - beta1_) * g_bias;
            v_biases_[layer][neuron] = beta2_ * v_biases_[layer][neuron] + (1.0f - beta2_) * g_bias * g_bias;
            
            float m_hat = m_biases_[layer][neuron] / bias_correction1;
            float v_hat = v_biases_[layer][neuron] / bias_correction2;
            biases[layer][neuron] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            
            // Update weights
            for (size_t weight = 0; weight < weights[layer][neuron].size(); ++weight) {
                float g_weight = weight_gradients[layer][neuron][weight];
                
                // Update moments
                m_weights_[layer][neuron][weight] = beta1_ * m_weights_[layer][neuron][weight] + (1.0f - beta1_) * g_weight;
                v_weights_[layer][neuron][weight] = beta2_ * v_weights_[layer][neuron][weight] + (1.0f - beta2_) * g_weight * g_weight;
                
                // Bias-corrected estimates
                float m_hat = m_weights_[layer][neuron][weight] / bias_correction1;
                float v_hat = v_weights_[layer][neuron][weight] / bias_correction2;
                
                // Update weight
                weights[layer][neuron][weight] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }
}

void AdamOptimizer::reset() {
    step_ = 0;
    m_weights_.clear();
    v_weights_.clear();
    m_biases_.clear();
    v_biases_.clear();
}

} // namespace rl_dqn

