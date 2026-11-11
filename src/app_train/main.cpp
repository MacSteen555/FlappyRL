#include "env_flappy/env_flappy.h"
#include "rl_dqn/dqn_agent.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

struct TrainOptions {
    long long total_steps = 300000;    // environment steps
    long long warmup_steps = 5000;     // random-policy steps before learning starts
    int eval_every = 25000;            // env steps between evaluations
    int eval_episodes = 10;
    int max_episode_steps = 20000;     // truncate very long episodes (~5.5 min of play)
    std::uint64_t seed = 1;
    std::string out = "models/flappy_dqn.bin";  // best checkpoint; final goes to <out>.last
};

void print_usage() {
    std::cout << "Usage: app_train [--steps N] [--warmup N] [--eval-every N]\n"
                 "                 [--eval-episodes N] [--seed N] [--out PATH]\n";
}

bool parse_args(int argc, char** argv, TrainOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        auto next = [&](long long& out) {
            if (i + 1 >= argc) return false;
            out = std::atoll(argv[++i]);
            return true;
        };
        long long v = 0;
        if (!std::strcmp(argv[i], "--steps") && next(v)) opt.total_steps = v;
        else if (!std::strcmp(argv[i], "--warmup") && next(v)) opt.warmup_steps = v;
        else if (!std::strcmp(argv[i], "--eval-every") && next(v)) opt.eval_every = static_cast<int>(v);
        else if (!std::strcmp(argv[i], "--eval-episodes") && next(v)) opt.eval_episodes = static_cast<int>(v);
        else if (!std::strcmp(argv[i], "--seed") && next(v)) opt.seed = static_cast<std::uint64_t>(v);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) opt.out = argv[++i];
        else { print_usage(); return false; }
    }
    return true;
}

// Number of pipes passed in one greedy episode on a fixed seed.
int run_greedy_episode(const rl_dqn::DQNAgent& agent, std::uint64_t seed, int max_steps) {
    env_flappy::FlappyEnv env(seed);
    env_flappy::Observation obs = env.reset(seed);
    int pipes = 0;
    for (int t = 0; t < max_steps; ++t) {
        auto r = env.step(agent.greedy_action(obs));
        if (r.reward > 0.0f) ++pipes;
        obs = r.observation;
        if (r.done) break;
    }
    return pipes;
}

} // namespace

int main(int argc, char** argv) {
    TrainOptions opt;
    if (!parse_args(argc, argv, opt)) return 1;

    rl_dqn::DQNConfig cfg;
    cfg.layer_sizes = {4, 64, 64, 2};
    cfg.learning_rate = 5e-4f;
    cfg.gamma = 0.99f;
    cfg.epsilon_start = 1.0f;
    cfg.epsilon_end = 0.01f;
    cfg.epsilon_decay_steps = static_cast<int>(opt.total_steps / 3);
    cfg.explore_flap_prob = 0.1f;
    cfg.replay_buffer_size = 100000;
    cfg.batch_size = 64;
    cfg.train_frequency = 4;
    cfg.target_update_frequency = 500;  // in train() calls = 2000 env steps
    cfg.seed = opt.seed;

    rl_dqn::DQNAgent agent(cfg);

    std::cout << "FlappyRL - DQN training\n"
              << "  steps=" << opt.total_steps << " warmup=" << opt.warmup_steps
              << " seed=" << opt.seed << "\n"
              << "  checkpoint=" << opt.out << "\n\n";

    std::filesystem::path out_path(opt.out);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    // Each training episode gets a fresh seed
    std::uint64_t episode_seed = opt.seed * 1000003ULL;
    const std::uint64_t eval_seed_base = 0xE7A1000000ULL;

    env_flappy::FlappyEnv env(episode_seed);
    env_flappy::Observation obs = env.reset(episode_seed);

    int episode = 0;
    int ep_steps = 0;
    int ep_pipes = 0;
    double recent_pipes = 0.0;   // exponential moving average of pipes/episode
    double recent_loss = 0.0;
    double best_eval = -1.0;

    auto t0 = std::chrono::steady_clock::now();

    for (long long step = 1; step <= opt.total_steps; ++step) {
        env_flappy::Action action = agent.select_action(obs);
        env_flappy::StepResult r = env.step(action);
        ++ep_steps;
        if (r.reward > 0.0f) ++ep_pipes;

        // Truncation (time limit) is not a real terminal: keep bootstrapping.
        bool truncated = !r.done && ep_steps >= opt.max_episode_steps;
        agent.store_experience(obs, action, r.reward, r.observation, r.done);
        obs = r.observation;

        if (step > opt.warmup_steps && step % cfg.train_frequency == 0) {
            float loss = agent.train();
            recent_loss = 0.99 * recent_loss + 0.01 * loss;
        }

        if (r.done || truncated) {
            ++episode;
            recent_pipes = episode == 1 ? ep_pipes : 0.98 * recent_pipes + 0.02 * ep_pipes;
            ++episode_seed;
            obs = env.reset(episode_seed);
            ep_steps = 0;
            ep_pipes = 0;
        }

        if (step % opt.eval_every == 0 || step == opt.total_steps) {
            int total = 0, best = 0;
            for (int e = 0; e < opt.eval_episodes; ++e) {
                int p = run_greedy_episode(agent, eval_seed_base + e, opt.max_episode_steps);
                total += p;
                best = std::max(best, p);
            }
            double mean = static_cast<double>(total) / opt.eval_episodes;
            // DQN performance oscillates, so keep the best policy seen 
            bool saved = false;
            if (mean >= best_eval) {
                best_eval = mean;
                agent.save_weights(opt.out);
                saved = true;
            }

            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << std::fixed << std::setprecision(3)
                      << "step " << std::setw(8) << step
                      << " | ep " << std::setw(5) << episode
                      << " | eps " << agent.get_epsilon()
                      << " | train pipes(avg) " << std::setprecision(2) << std::setw(6) << recent_pipes
                      << " | loss " << std::setprecision(4) << recent_loss
                      << " | EVAL pipes mean " << std::setprecision(1) << std::setw(6) << mean
                      << " max " << std::setw(4) << best
                      << " | " << std::setprecision(0) << secs << "s"
                      << (saved ? "  [saved best]" : "") << std::endl;
        }
    }

    agent.save_weights(opt.out + ".last");
    std::cout << "\nBest eval mean " << std::setprecision(1) << best_eval
              << " pipes -> " << opt.out << "\n"
              << "Final weights -> " << opt.out << ".last\n"
              << "Watch it play: app_play --agent " << opt.out << std::endl;

    return 0;
}
