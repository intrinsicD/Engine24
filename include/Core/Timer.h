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
        Timer();

        Timer &start();

        Timer &stop();

        Timer &update();

        TimePoint m_start;
        TimePoint m_end;
        float delta;
    };

    class TimeTicker{
    public:
        TimeTicker() {
            reset();
        }

        void reset() {
            m_last_time_point = Clock::now();
        }

        double tick() {
            const TimePoint current_time_point = Clock::now();
            const Duration time_span = std::chrono::duration_cast<Duration>(current_time_point - m_last_time_point);
            m_last_time_point = current_time_point;
            return time_span.count();
        }

    private:
        TimePoint m_last_time_point;
    };
}

#endif //ENGINE24_TIMER_H
