#include "rl_dqn/replay_buffer.h"
#include <algorithm>
#include <stdexcept>

namespace rl_dqn {

ReplayBuffer::ReplayBuffer(std::size_t capacity, std::uint64_t seed)
    : capacity_(capacity), write_index_(0), rng_(static_cast<std::mt19937::result_type>(seed)) {
    experiences_.reserve(capacity);
}

void ReplayBuffer::push(const Experience& experience) {
    if (experiences_.size() < capacity_) {
        experiences_.push_back(experience);
    } else {
        // Circular buffer: replace oldest experience
        experiences_[write_index_] = experience;
        write_index_ = (write_index_ + 1) % capacity_;
    }
}

std::vector<Experience> ReplayBuffer::sample(std::size_t batch_size) const {
    if (experiences_.size() < batch_size) {
        throw std::runtime_error("Not enough experiences in buffer");
    }
    
    std::vector<Experience> batch;
    batch.reserve(batch_size);
    
    // Uniform sampling with replacement (standard for DQN)
    std::uniform_int_distribution<std::size_t> pick(0, experiences_.size() - 1);
    for (std::size_t i = 0; i < batch_size; ++i) {
        batch.push_back(experiences_[pick(rng_)]);
    }
    
    return batch;
}

bool ReplayBuffer::can_sample(std::size_t batch_size) const {
    return experiences_.size() >= batch_size;
}

void ReplayBuffer::clear() {
    experiences_.clear();
    write_index_ = 0;
}

} // namespace rl_dqn

