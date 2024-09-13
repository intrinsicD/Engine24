//
// Created by alex on 13.09.24.
//

#ifndef ENGINE24_VIEWPARAMETERS_H
#define ENGINE24_VIEWPARAMETERS_H

#include "MatVec.h"

namespace Bcg {
    template<typename T>
    class ViewParameters {
    public:
        ViewParameters(const Vector<T, 3> &eye, const Vector<T, 3> &center, const Vector<T, 3> &up) :
                m_eye(eye), m_center(center), m_up(up.normalized()), m_dirty(true) {}

        ViewParameters() : ViewParameters(Vector<T, 3>{0.0, 0.0, 3.0},
                                          Vector<T, 3>{0.0, 0.0, 0.0},
                                          Vector<T, 3>{0.0, 1.0, 0.0}) {}

        const Vector<T, 3> &eye() const {
            return m_eye;
        }

        const Vector<T, 3> &center() const {
            return m_center;
        }

        const Vector<T, 3> &up() const {
            return m_up;
        }

        Vector<T, 3> &eye() {
            m_dirty = true;
            return m_eye;
        }

        Vector<T, 3> &center() {
            m_dirty = true;
            return m_center;
        }

        Vector<T, 3> &up() {
            m_dirty = true;
            return m_up;
        }

        void set_eye(const Vector<T, 3> &eye) {
            m_eye = eye;
            m_dirty = true;
        }

        void set_center(const Vector<T, 3> &center) {
            m_center = center;
            m_dirty = true;
        }

        void set_up(const Vector<T, 3> &up) {
            m_up = up;
            m_dirty = true;
        }

        Vector<T, 3> front() const {
            return (center() - eye()).normalized();
        }

        Vector<T, 3> compute_right(const Vector<T, 3> &front) const {
            return cross(front, up()).normalized();
        }

        Vector<T, 3> right() const {
            return compute_right(front());
        }

        T distance_to_center() const {
            return (center() - eye()).norm();
        }

        bool is_dirty() const {
            return m_dirty;
        }

        void mark_clean() {
            m_dirty = false;
        }

    private:
        Vector<T, 3> m_eye;
        Vector<T, 3> m_center;
        Vector<T, 3> m_up;
        bool m_dirty = false;
    };
}
#endif //ENGINE24_VIEWPARAMETERS_H
