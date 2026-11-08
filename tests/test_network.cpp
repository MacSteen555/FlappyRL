#include <catch2/catch_test_macros.hpp>
#include "rl_dqn/network.h"
#include <cmath>
#include <vector>

namespace {

// Loss used by Network::backward: L = 0.5 * sum((pred - target)^2),
// so dL/dpred = pred - target.
float loss(const rl_dqn::Network& net, const std::vector<float>& x,
           const std::vector<float>& target) {
    std::vector<float> y = net.forward(x);
    float l = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) {
        float d = y[i] - target[i];
        l += 0.5f * d * d;
    }
    return l;
}

bool close(float analytic, float numeric) {
    float scale = std::max({1.0f, std::fabs(analytic), std::fabs(numeric)});
    return std::fabs(analytic - numeric) / scale < 2e-2f;
}

} // namespace

TEST_CASE("Network backward matches finite-difference gradients", "[network]") {
    // Two hidden layers of different widths so every backprop path is exercised.
    rl_dqn::Network net({3, 6, 5, 2}, 7);
    const std::vector<float> x = {0.4f, -0.7f, 0.25f};
    const std::vector<float> target = {1.5f, -0.5f};

    std::vector<std::vector<std::vector<float>>> wg;
    std::vector<std::vector<float>> bg;
    net.backward(x, target, net.forward(x), wg, bg);

    const float eps = 1e-3f;
    int mismatches = 0;

    auto weights = net.get_weights();
    for (size_t l = 0; l < weights.size(); ++l) {
        for (size_t i = 0; i < weights[l].size(); ++i) {
            for (size_t j = 0; j < weights[l][i].size(); ++j) {
                auto w = weights;
                w[l][i][j] += eps;
                net.set_weights(w);
                float lp = loss(net, x, target);
                w[l][i][j] -= 2 * eps;
                net.set_weights(w);
                float lm = loss(net, x, target);
                net.set_weights(weights);

                float numeric = (lp - lm) / (2 * eps);
                if (!close(wg[l][i][j], numeric)) {
                    ++mismatches;
                    UNSCOPED_INFO("weight[" << l << "][" << i << "][" << j << "] analytic="
                                  << wg[l][i][j] << " numeric=" << numeric);
                }
            }
        }
    }

    auto biases = net.get_biases();
    for (size_t l = 0; l < biases.size(); ++l) {
        for (size_t i = 0; i < biases[l].size(); ++i) {
            auto b = biases;
            b[l][i] += eps;
            net.set_biases(b);
            float lp = loss(net, x, target);
            b[l][i] -= 2 * eps;
            net.set_biases(b);
            float lm = loss(net, x, target);
            net.set_biases(biases);

            float numeric = (lp - lm) / (2 * eps);
            if (!close(bg[l][i], numeric)) {
                ++mismatches;
                UNSCOPED_INFO("bias[" << l << "][" << i << "] analytic=" << bg[l][i]
                              << " numeric=" << numeric);
            }
        }
    }

    REQUIRE(mismatches == 0);
}

TEST_CASE("Network can fit a simple regression target", "[network]") {
    // Plain SGD on y = [x0 + x1, x0 - x1]; loss must drop substantially.
    rl_dqn::Network net({2, 16, 16, 2}, 3);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    auto eval = [&] {
        std::mt19937 r(99);
        float total = 0.0f;
        for (int n = 0; n < 200; ++n) {
            std::vector<float> x = {u(r), u(r)};
            total += loss(net, x, {x[0] + x[1], x[0] - x[1]});
        }
        return total / 200.0f;
    };

    float before = eval();
    for (int it = 0; it < 3000; ++it) {
        std::vector<float> x = {u(rng), u(rng)};
        std::vector<float> t = {x[0] + x[1], x[0] - x[1]};
        std::vector<std::vector<std::vector<float>>> wg;
        std::vector<std::vector<float>> bg;
        net.backward(x, t, net.forward(x), wg, bg);
        net.update_weights(wg, bg, 0.01f);
    }
    float after = eval();

    INFO("loss before=" << before << " after=" << after);
    REQUIRE(after < 0.05f * before);
}
