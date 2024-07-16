//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_FRAMETIMER_H
#define ENGINE24_FRAMETIMER_H

#include "Timer.h"

namespace Bcg {
    struct FrameTimer {
        Timer timer;
        float avg_frame_time = 0;
        float fps = 0;
        size_t frame_counter = 0;
    };
}
#endif //ENGINE24_FRAMETIMER_H
