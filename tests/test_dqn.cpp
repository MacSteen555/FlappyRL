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


namespace {
env_flappy::Observation make_obs(float y, float vy, float dx, float dy) {
    env_flappy::Observation o;
    o.y = y; o.vy = vy; o.dx_to_pipe = dx; o.dy_to_gap = dy;
    return o;
}
} // namespace

TEST_CASE("Target network sync copies weights AND biases", "[dqn]") {
    rl_dqn::DQNConfig config;
    config.layer_sizes = {4, 16, 2};
    config.batch_size = 8;
    config.learning_rate = 1e-2f;
    config.target_update_frequency = 0;  // manual sync only

    rl_dqn::DQNAgent agent(config);
    auto s = make_obs(0.5f, 0.0f, 0.5f, 0.1f);
    for (int i = 0; i < 64; ++i) {
        agent.store_experience(s, env_flappy::Action::FLAP, 5.0f, s, true);
    }
    for (int i = 0; i < 50; ++i) (void)agent.train();  // biases now non-zero

    agent.update_target_network();
    auto q = agent.get_q_values(s);
    auto tq = agent.get_target_q_values(s);
    REQUIRE(q[0] == tq[0]);
    REQUIRE(q[1] == tq[1]);
}

TEST_CASE("Agent learns terminal reward and bootstrapped value", "[dqn]") {
    // State A --FLAP--> terminal, reward -1    => Q(A, FLAP)    -> -1
    // State B --NO_FLAP--> B, reward 1, gamma  => Q(B, NO_FLAP) -> 1 / (1 - gamma)
    rl_dqn::DQNConfig config;
    config.layer_sizes = {4, 32, 32, 2};
    config.batch_size = 32;
    config.replay_buffer_size = 1000;
    config.learning_rate = 1e-3f;
    config.gamma = 0.5f;
    config.target_update_frequency = 50;

    rl_dqn::DQNAgent agent(config);
    auto a = make_obs(0.2f, -1.0f, 0.8f, 0.3f);
    auto b = make_obs(0.7f, 1.0f, 0.1f, -0.2f);
    for (int i = 0; i < 200; ++i) {
        agent.store_experience(a, env_flappy::Action::FLAP, -1.0f, a, true);
        agent.store_experience(b, env_flappy::Action::NO_FLAP, 1.0f, b, false);
    }
    for (int i = 0; i < 3000; ++i) (void)agent.train();

    auto qa = agent.get_q_values(a);
    auto qb = agent.get_q_values(b);
    INFO("Q(A,FLAP)=" << qa[1] << " Q(B,NO_FLAP)=" << qb[0]);
    REQUIRE(std::fabs(qa[1] - (-1.0f)) < 0.1f);
    REQUIRE(std::fabs(qb[0] - 2.0f) < 0.2f);
}
