//
// Created by alex on 11/3/24.
//

#include "Timer.h"

namespace Bcg {
    Timer::Timer() : m_start(Clock::now()), delta(0) {}

    Timer &Timer::start() {
        m_start = Clock::now();
        return *this;
    }

    Timer &Timer::stop() {
        m_end = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start);
        delta = duration.count() * 1e-9f;
        return *this;
    }

    Timer &Timer::update() {
        stop();
        m_start = m_end;
        return *this;
    }
}