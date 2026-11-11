#include "rl_dqn/dqn_agent.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <cstdint>
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
      current_epsilon_(config.epsilon_start),
      rng_(static_cast<std::mt19937::result_type>(config.seed + 3)) {

    // Initialize target network with same weights as main network
    update_target_network();
}

std::vector<float> DQNAgent::observation_to_input(const env_flappy::Observation& obs) const {
    // Rescale each feature to roughly [-1, 1] so no input dominates early training
    return {obs.y * 2.0f - 1.0f,
            obs.vy / 2.5f,
            obs.dx_to_pipe * 2.0f - 0.5f,
            obs.dy_to_gap * 2.0f};
}

static int argmax2(const std::vector<float>& q) { return q[1] > q[0] ? 1 : 0; }

static env_flappy::Action index_to_action(int idx) {
    return idx == 1 ? env_flappy::Action::FLAP : env_flappy::Action::NO_FLAP;
}

static int action_to_index(env_flappy::Action a) {
    return a == env_flappy::Action::FLAP ? 1 : 0;
}

env_flappy::Action DQNAgent::select_action(const env_flappy::Observation& state) {
    total_steps_++;

    // Update epsilon (linear decay)
    float epsilon_progress = std::min(1.0f, static_cast<float>(total_steps_) / config_.epsilon_decay_steps);
    current_epsilon_ = config_.epsilon_start + (config_.epsilon_end - config_.epsilon_start) * epsilon_progress;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(rng_) < current_epsilon_) {
        return dist(rng_) < config_.explore_flap_prob ? env_flappy::Action::FLAP
                                                       : env_flappy::Action::NO_FLAP;
    }
    return greedy_action(state);
}

env_flappy::Action DQNAgent::greedy_action(const env_flappy::Observation& state) const {
    return index_to_action(argmax2(main_network_.forward(observation_to_input(state))));
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

        float target_q = exp.reward;
        if (!exp.done) {
            std::vector<float> next_input = observation_to_input(exp.next_state);
            std::vector<float> next_target_q = target_network_.forward(next_input);
            float next_q;
            if (config_.double_dqn) {
                // Double DQN: main net picks the action, target net scores it.
                int best = argmax2(main_network_.forward(next_input));
                next_q = next_target_q[best];
            } else {
                next_q = std::max(next_target_q[0], next_target_q[1]);
            }
            target_q += config_.gamma * next_q;
        }

        // Only the taken action's slot matters; the other is filled in train()
        target_q_values[action_to_index(exp.action)] = target_q;
        targets.push_back(target_q_values);
    }

    return targets;
}

float DQNAgent::train() {
    if (!replay_buffer_.can_sample(config_.batch_size)) {
        return 0.0f;  // Not enough experiences yet
    }

    std::vector<Experience> batch = replay_buffer_.sample(config_.batch_size);
    std::vector<std::vector<float>> targets = compute_targets(batch);

    // Gradient accumulators shaped like the network
    auto total_weight_gradients = main_network_.get_weights();
    auto total_bias_gradients = main_network_.get_biases();
    for (auto& layer : total_weight_gradients)
        for (auto& neuron : layer) std::fill(neuron.begin(), neuron.end(), 0.0f);
    for (auto& layer : total_bias_gradients) std::fill(layer.begin(), layer.end(), 0.0f);

    const float delta = config_.huber_delta;
    float total_loss = 0.0f;

    std::vector<std::vector<std::vector<float>>> weight_gradients;
    std::vector<std::vector<float>> bias_gradients;

    for (size_t i = 0; i < batch.size(); ++i) {
        const auto& exp = batch[i];
        std::vector<float> input = observation_to_input(exp.state);
        std::vector<float> predicted_q = main_network_.forward(input);

        int a = action_to_index(exp.action);

        // Non-taken action: target = prediction, so its gradient is zero
        targets[i][1 - a] = predicted_q[1 - a];

        // Huber loss: Network::backward uses (pred - target) as dL/dpred, so we clip the TD error by moving the target to within `delta` of the prediction.
        float td = predicted_q[a] - targets[i][a];
        float abs_td = std::fabs(td);
        total_loss += abs_td <= delta ? 0.5f * td * td : delta * (abs_td - 0.5f * delta);
        targets[i][a] = predicted_q[a] - std::clamp(td, -delta, delta);

        main_network_.backward(input, targets[i], predicted_q, weight_gradients, bias_gradients);

        for (size_t layer = 0; layer < weight_gradients.size(); ++layer) {
            for (size_t neuron = 0; neuron < weight_gradients[layer].size(); ++neuron) {
                total_bias_gradients[layer][neuron] += bias_gradients[layer][neuron];
                for (size_t w = 0; w < weight_gradients[layer][neuron].size(); ++w) {
                    total_weight_gradients[layer][neuron][w] += weight_gradients[layer][neuron][w];
                }
            }
        }
    }

    // Average over the batch (gradient of the mean loss)
    const float inv_n = 1.0f / static_cast<float>(batch.size());
    for (auto& layer : total_weight_gradients)
        for (auto& neuron : layer)
            for (float& g : neuron) g *= inv_n;
    for (auto& layer : total_bias_gradients)
        for (float& g : layer) g *= inv_n;

    auto weights = main_network_.get_weights();
    auto biases = main_network_.get_biases();
    optimizer_.update(weights, biases, total_weight_gradients, total_bias_gradients);
    main_network_.set_weights(weights);
    main_network_.set_biases(biases);

    training_steps_++;
    if (config_.target_update_frequency > 0 &&
        training_steps_ % config_.target_update_frequency == 0) {
        update_target_network();
    }

    return total_loss * inv_n;
}

void DQNAgent::update_target_network() {
    target_network_.set_weights(main_network_.get_weights());
    target_network_.set_biases(main_network_.get_biases());
}

float DQNAgent::get_epsilon() const {
    return current_epsilon_;
}

std::vector<float> DQNAgent::get_q_values(const env_flappy::Observation& state) const {
    return main_network_.forward(observation_to_input(state));
}

std::vector<float> DQNAgent::get_target_q_values(const env_flappy::Observation& state) const {
    return target_network_.forward(observation_to_input(state));
}

// Checkpoint format (little-endian, as written by this machine):
//   char[4]  magic "FRL1"
//   uint32   number of layer sizes N
//   int32[N] layer sizes
//   float[]  for each layer: weights (row-major [out][in]) then biases
namespace {
constexpr char kMagic[4] = {'F', 'R', 'L', '1'};

template <typename T>
void write_pod(std::ostream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
T read_pod(std::istream& in) {
    T v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!in) throw std::runtime_error("Checkpoint is truncated");
    return v;
}
} // namespace

void DQNAgent::save_weights(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open checkpoint for writing: " + filepath);

    const auto& sizes = main_network_.get_layer_sizes();
    out.write(kMagic, sizeof(kMagic));
    write_pod(out, static_cast<std::uint32_t>(sizes.size()));
    for (int n : sizes) write_pod(out, static_cast<std::int32_t>(n));

    const auto weights = main_network_.get_weights();
    const auto biases = main_network_.get_biases();
    for (size_t l = 0; l < weights.size(); ++l) {
        for (const auto& row : weights[l]) {
            out.write(reinterpret_cast<const char*>(row.data()),
                      static_cast<std::streamsize>(row.size() * sizeof(float)));
        }
        out.write(reinterpret_cast<const char*>(biases[l].data()),
                  static_cast<std::streamsize>(biases[l].size() * sizeof(float)));
    }
    if (!out) throw std::runtime_error("Failed writing checkpoint: " + filepath);
}

void DQNAgent::load_weights(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open checkpoint: " + filepath);

    char magic[4] = {};
    in.read(magic, sizeof(magic));
    if (!in || !std::equal(magic, magic + 4, kMagic)) {
        throw std::runtime_error("Not a FlappyRL checkpoint: " + filepath);
    }

    auto n_layers = read_pod<std::uint32_t>(in);
    if (n_layers < 2 || n_layers > 64) throw std::runtime_error("Corrupt checkpoint header");
    std::vector<int> sizes(n_layers);
    for (auto& n : sizes) {
        n = read_pod<std::int32_t>(in);
        if (n <= 0 || n > 1 << 16) throw std::runtime_error("Corrupt checkpoint header");
    }
    if (sizes.front() != 4 || sizes.back() != 2) {
        throw std::runtime_error("Checkpoint network must map 4 inputs to 2 actions");
    }

    // The file's architecture wins over the config's
    Network net(sizes, config_.seed);
    auto weights = net.get_weights();
    auto biases = net.get_biases();
    for (size_t l = 0; l < weights.size(); ++l) {
        for (auto& row : weights[l]) {
            in.read(reinterpret_cast<char*>(row.data()),
                    static_cast<std::streamsize>(row.size() * sizeof(float)));
        }
        in.read(reinterpret_cast<char*>(biases[l].data()),
                static_cast<std::streamsize>(biases[l].size() * sizeof(float)));
    }
    if (!in) throw std::runtime_error("Checkpoint is truncated: " + filepath);
    in.peek();
    if (!in.eof()) throw std::runtime_error("Checkpoint has trailing data: " + filepath);

    net.set_weights(weights);
    net.set_biases(biases);
    config_.layer_sizes = sizes;
    main_network_ = net;
    target_network_ = net;
    optimizer_.reset();  // moment estimates belong to the old parameters
}

} // namespace rl_dqn

