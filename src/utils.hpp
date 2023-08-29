#include "common.hpp"
#include <chrono>

class Timer {
 public:
    Timer(): m_start(std::chrono::steady_clock::time_point::min()){}

    void clear() {
        m_start = std::chrono::steady_clock::time_point::min();
    }

    bool isStarted() const {
        return (m_start != std::chrono::steady_clock::time_point::min());
    }

    void start() {
        m_start = std::chrono::steady_clock::now();
    }

    std::int64_t stop() const {
        if (!this->isStarted()) {
            throw std::runtime_error("timer has not been started");
        }

        const std::chrono::steady_clock::duration diff = std::chrono::steady_clock::now() - m_start;
        return std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    }

 private:
    std::chrono::steady_clock::time_point m_start;
};