#include <catch2/catch_test_macros.hpp>
#include "rl_dqn/dqn_agent.h"
#include "env_flappy/env_flappy.h"
#include <cmath>

TEST_CASE("DQN Network Forward Pass", "[dqn]") {
    rl_dqn::Network network({4, 8, 2}, 12345);
    
    std::vector<float> input = {0.5f, -0.3f, 0.1f, 0.2f};
    std::vector<float> output = network.forward(input);
    
    REQUIRE(output.size() == 2);
    // Output should be finite
    REQUIRE(std::isfinite(output[0]));
    REQUIRE(std::isfinite(output[1]));
}

TEST_CASE("DQN Network Backward Pass", "[dqn]") {
    rl_dqn::Network network({4, 8, 2}, 12345);
    
    std::vector<float> input = {0.5f, -0.3f, 0.1f, 0.2f};
    std::vector<float> predicted = network.forward(input);
    std::vector<float> target = {0.8f, 0.2f};
    
    std::vector<std::vector<std::vector<float>>> weight_gradients;
    std::vector<std::vector<float>> bias_gradients;
    
    network.backward(input, target, predicted, weight_gradients, bias_gradients);
    
    REQUIRE(weight_gradients.size() == 2);  // 2 layers
    REQUIRE(bias_gradients.size() == 2);
    
    // Gradients should be finite
    for (const auto& layer : weight_gradients) {
        for (const auto& neuron : layer) {
            for (float grad : neuron) {
                REQUIRE(std::isfinite(grad));
            }
        }
    }
}

TEST_CASE("DQN Agent Basic Functionality", "[dqn]") {
    rl_dqn::DQNConfig config;
    config.layer_sizes = {4, 8, 2};
    config.batch_size = 4;
    config.replay_buffer_size = 100;
    
    rl_dqn::DQNAgent agent(config);
    
    // Test action selection
    env_flappy::Observation obs;
    obs.y = 0.5f;
    obs.vy = 0.1f;
    obs.dx_to_pipe = 1.0f;
    obs.dy_to_gap = 0.2f;
    
    env_flappy::Action action = agent.select_action(obs);
    bool is_valid_action = (action == env_flappy::Action::NO_FLAP || action == env_flappy::Action::FLAP);
    REQUIRE(is_valid_action);
    
    // Test Q-values
    std::vector<float> q_values = agent.get_q_values(obs);
    REQUIRE(q_values.size() == 2);
    REQUIRE(std::isfinite(q_values[0]));
    REQUIRE(std::isfinite(q_values[1]));
}

TEST_CASE("DQN Agent Training", "[dqn]") {
    rl_dqn::DQNConfig config;
    config.layer_sizes = {4, 8, 2};
    config.batch_size = 4;
    config.replay_buffer_size = 100;
    config.learning_rate = 0.001f;
    
    rl_dqn::DQNAgent agent(config);
    
    // Create some experiences
    env_flappy::Observation state1, state2, state3, state4;
    state1.y = 0.5f; state1.vy = 0.1f; state1.dx_to_pipe = 1.0f; state1.dy_to_gap = 0.2f;
    state2.y = 0.4f; state2.vy = 0.2f; state2.dx_to_pipe = 0.8f; state2.dy_to_gap = 0.1f;
    state3.y = 0.6f; state3.vy = -0.1f; state3.dx_to_pipe = 0.6f; state3.dy_to_gap = 0.3f;
    state4.y = 0.3f; state4.vy = 0.3f; state4.dx_to_pipe = 0.4f; state4.dy_to_gap = 0.0f;
    
    // Store experiences
    agent.store_experience(state1, env_flappy::Action::NO_FLAP, 0.1f, state2, false);
    agent.store_experience(state2, env_flappy::Action::FLAP, 0.2f, state3, false);
    agent.store_experience(state3, env_flappy::Action::NO_FLAP, -1.0f, state4, true);
    agent.store_experience(state4, env_flappy::Action::FLAP, 0.5f, state1, false);
    
    // Try to train (should return 0.0 if not enough samples)
    (void)agent.train();  // Suppress unused variable warning
    
    // Add more experiences to reach batch size
    for (int i = 0; i < 10; ++i) {
        agent.store_experience(state1, env_flappy::Action::NO_FLAP, 0.1f, state2, false);
    }
    
    // Now should be able to train
    float loss2 = agent.train();
    REQUIRE(loss2 >= 0.0f);
    REQUIRE(std::isfinite(loss2));
}

TEST_CASE("Replay Buffer", "[dqn]") {
    rl_dqn::ReplayBuffer buffer(10, 12345);
    
    env_flappy::Observation state, next_state;
    state.y = 0.5f; state.vy = 0.1f; state.dx_to_pipe = 1.0f; state.dy_to_gap = 0.2f;
    next_state.y = 0.4f; next_state.vy = 0.2f; next_state.dx_to_pipe = 0.8f; next_state.dy_to_gap = 0.1f;
    
    rl_dqn::Experience exp;
    exp.state = state;
    exp.action = env_flappy::Action::FLAP;
    exp.reward = 0.5f;
    exp.next_state = next_state;
    exp.done = false;
    
    // Push experiences
    for (int i = 0; i < 15; ++i) {
        buffer.push(exp);
    }
    
    REQUIRE(buffer.size() == 10);  // Should be capped at capacity
    REQUIRE(buffer.can_sample(5));
    
    // Sample batch
    std::vector<rl_dqn::Experience> batch = buffer.sample(5);
    REQUIRE(batch.size() == 5);
}

