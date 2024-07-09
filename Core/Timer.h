//
// Created by alex on 09.07.24.
//

#ifndef ENGINE24_TIMER_H
#define ENGINE24_TIMER_H

#include <chrono>

namespace Bcg {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<float>;

    struct Timer {
        Timer() : m_start(Clock::now()), delta(0) {}

        void start() {
            m_start = Clock::now();
        }

        void stop() {
            m_end = Clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start);
            delta = duration.count() * 1e-9f;
        }

        void update() {
            stop();
            m_start = m_end;
        }

        TimePoint m_start;
        TimePoint m_end;
        float delta;
    };

    using FrameTimer = Timer;
}

#endif //ENGINE24_TIMER_H
