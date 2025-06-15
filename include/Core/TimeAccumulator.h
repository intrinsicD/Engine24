//
// Created by alex on 6/15/25.
//

#ifndef TIMEACCUMULATOR_H
#define TIMEACCUMULATOR_H

namespace Bcg {
    class TimeAccumulator {
    public:
        TimeAccumulator() = default;

        void add(double delta_time) {
            m_accumulator += delta_time;
        }

        bool has_step(double fixed_time_step) const {
            return m_accumulator >= fixed_time_step;
        }

        void consume_step(double fixed_time_step) {
            m_accumulator -= fixed_time_step;
        }

        double get_alpha(double fixed_time_step) const {
            return m_accumulator / fixed_time_step;
        }

    private:
        double m_accumulator = 0.0;
    };
}

#endif //TIMEACCUMULATOR_H
