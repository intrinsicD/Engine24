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
}

#endif //ENGINE24_TIMER_H
