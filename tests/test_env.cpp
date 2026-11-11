#include <catch2/catch_test_macros.hpp>
#include "env_flappy/env_flappy.h"

#include <vector>

using env_flappy::Action;
using env_flappy::FlappyEnv;

namespace {

constexpr float kBirdX = FlappyEnv::kBirdX;

// Simple open-loop policy: flap every 18 frames (roughly a hover).
Action hover(int t) { return t % 18 == 0 ? Action::FLAP : Action::NO_FLAP; }

bool same(const env_flappy::Observation& a, const env_flappy::Observation& b) {
    return a.y == b.y && a.vy == b.vy && a.dx_to_pipe == b.dx_to_pipe && a.dy_to_gap == b.dy_to_gap;
}

} // namespace

TEST_CASE("Same seed and actions give identical trajectories", "[env]") {
    FlappyEnv a(7), b(7);
    for (int t = 0; t < 300; ++t) {
        auto ra = a.step(hover(t));
        auto rb = b.step(hover(t));
        REQUIRE(same(ra.observation, rb.observation));
        REQUIRE(ra.reward == rb.reward);
        REQUIRE(ra.done == rb.done);
        if (ra.done) break;
    }
}

TEST_CASE("reset(seed) reproduces a fresh env; different seeds differ", "[env]") {
    FlappyEnv played(7);
    for (int t = 0; t < 100; ++t) played.step(hover(t));
    played.reset(7);

    FlappyEnv fresh(7);
    REQUIRE(played.pipes().size() == fresh.pipes().size());
    for (size_t i = 0; i < fresh.pipes().size(); ++i) {
        REQUIRE(played.pipes()[i].x == fresh.pipes()[i].x);
        REQUIRE(played.pipes()[i].gap_y == fresh.pipes()[i].gap_y);
    }
    REQUIRE(same(played.observe(), fresh.observe()));
    REQUIRE(played.steps() == 0);
    REQUIRE_FALSE(played.done());

    FlappyEnv other(8);
    REQUIRE(other.pipes()[0].gap_y != fresh.pipes()[0].gap_y);
}

TEST_CASE("Hitting the floor or ceiling ends the episode with r_death", "[env]") {
    FlappyEnv env(1);
    const float r_death = env.config().r_death;

    SECTION("floor") {
        env._set_bird(0.01f, -1.0f);
        auto r = env.step(Action::NO_FLAP);
        REQUIRE(r.done);
        REQUIRE(r.reward == r_death);
    }
    SECTION("ceiling") {
        env._set_bird(0.99f, 1.0f);
        auto r = env.step(Action::NO_FLAP);
        REQUIRE(r.done);
        REQUIRE(r.reward == r_death);
    }
}

TEST_CASE("Hitting a pipe ends the episode", "[env]") {
    FlappyEnv env(1);
    // Pipe overlapping the bird horizontally, gap far above the bird
    env._set_current_pipe(kBirdX + 0.01f, 0.9f);
    env._set_bird(0.5f, 0.0f);
    auto r = env.step(Action::NO_FLAP);
    REQUIRE(r.done);
    REQUIRE(r.reward == env.config().r_death);
    REQUIRE(env.done());
}

TEST_CASE("Flying through the gap pays r_pass exactly once", "[env]") {
    FlappyEnv env(1);
    // Pipe center just ahead of the bird, bird centered in the gap
    env._set_current_pipe(kBirdX + 0.005f, 0.5f);
    env._set_bird(0.5f, 0.0f);

    float total = 0.0f;
    int pass_steps = 0;
    for (int t = 0; t < 30; ++t) {  // well before the next pipe (~72 steps away)
        auto r = env.step(Action::NO_FLAP);
        REQUIRE_FALSE(r.done);
        if (r.reward > 0.0f) ++pass_steps;
        total += r.reward;
    }
    REQUIRE(pass_steps == 1);
    REQUIRE(total == env.config().r_pass);
}

TEST_CASE("Stepping after done is a no-op", "[env]") {
    FlappyEnv env(1);
    env._set_bird(0.01f, -1.0f);
    REQUIRE(env.step(Action::NO_FLAP).done);

    auto before = env.observe();
    int steps = env.steps();
    auto r = env.step(Action::FLAP);
    REQUIRE(r.done);
    REQUIRE(same(r.observation, before));
    REQUIRE(env.steps() == steps);
}

TEST_CASE("Pipes keep spawning ahead of the bird", "[env]") {
    FlappyEnv env(3);
    // Keep the bird alive regardless of pipes: run long enough to scroll
    // several pipes past, re-placing the bird in each current gap.
    for (int t = 0; t < 1000; ++t) {
        auto obs = env.observe();
        env._set_bird(obs.y + obs.dy_to_gap, 0.0f);
        auto r = env.step(Action::NO_FLAP);
        REQUIRE_FALSE(r.done);
        REQUIRE(env.pipes().size() >= 2);
        REQUIRE(env.pipes().back().x >= 3.0f - env.config().pipe_spacing);
        // Observation always refers to a pipe not yet fully behind the bird
        REQUIRE(r.observation.dx_to_pipe > -env.config().pipe_width);
    }
}
