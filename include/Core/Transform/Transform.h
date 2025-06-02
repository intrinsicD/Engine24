//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_TRANSFORM_H
#define ENGINE24_TRANSFORM_H

#include "glm/gtc/type_ptr.hpp"

#define GLM_ENABLE_EXPERIMENTAL

#include "glm/gtx/matrix_decompose.hpp"
#include "glm/gtx/quaternion.hpp"
#include "StringTraits.h"
#include "GlmToEigen.h"

#include <numbers>

namespace Bcg {
    struct Transform {
        static Transform Identity() {
            Transform t;
            t.m_local = glm::mat4(1.0f);
            t.cached_parent_world = glm::mat4(1.0f);
            return t;
        }

        const glm::mat4 &local() const {
            return m_local;
        }

        void set_local(const glm::mat4 &m) {
            m_local = m;
            dirty = true;
        }

        glm::mat4 world() const {
            return cached_parent_world * m_local;
        }

        void set_parent_world(const glm::mat4 &parent_world) {
            cached_parent_world = parent_world;
            dirty = true;
        }

        const glm::mat4 &get_cached_parent_world() const {
            return cached_parent_world;
        }

        bool dirty = false;
    private:
        glm::mat4 m_local = glm::mat4(1.0f);
        glm::mat4 cached_parent_world = glm::mat4(1.0f);
    };

    struct TransformParameters {
        glm::vec3 scale;
        glm::vec3 angle_axis;
        glm::vec3 position;
    };



//! OpenGL matrix for translation by vector t
    template<typename T, glm::qualifier Q = glm::defaultp>
    glm::mat<4, 4, T, Q> translation_matrix(const glm::vec<3, T, Q> &t) {
        glm::mat<4, 4, T, Q> m(T(0.0));
        m[0][0] = m[1][1] = m[2][2] = m[3][3] = T(1.0);
        m[3][0] = t[0];
        m[3][1] = t[1];
        m[3][2] = t[2];
        return m;
    }

//! OpenGL matrix for scaling x/y/z by s
    template<typename T, glm::qualifier Q = glm::defaultp>
    glm::mat<4, 4, T, Q> scaling_matrix(const T s) {
        glm::mat<4, 4, T, Q> m(T(0.0));
        m[0][0] = m[1][1] = m[2][2] = s;
        m[3][3] = T(1.0);
        return m;
    }

//! OpenGL matrix for scaling x/y/z by the components of s
    template<typename T, glm::qualifier Q = glm::defaultp>
    glm::mat<4, 4, T, Q> scaling_matrix(const glm::vec<3, T, Q> s) {
        glm::mat<4, 4, T, Q> m(T(0.0));
        m[0][0] = s[0];
        m[1][1] = s[1];
        m[2][2] = s[2];
        m[3][3] = T(1.0);

        return m;
    }

//! OpenGL matrix for rotation around x-axis by given angle (in degrees)
    template<typename T, glm::qualifier Q = glm::defaultp>
    glm::mat<4, 4, T, Q> rotation_matrix_x(T angle) {
        T ca = cos(angle * T(std::numbers::pi / 180.0));
        T sa = sin(angle * T(std::numbers::pi / 180.0));

        glm::mat<4, 4, T, Q> m(T(0.0));
        m[0][0] = T(1.0);
        m[1][1] = ca;
        m[2][1] = -sa;
        m[2][2] = ca;
        m[1][2] = sa;
        m[3][3] = T(1.0);

        return m;
    }

//! OpenGL matrix for rotation around y-axis by given angle (in degrees)
    template<typename T, glm::qualifier Q = glm::defaultp>
    glm::mat<4, 4, T, Q> rotation_matrix_y(T angle) {
        T ca = cos(angle * T(std::numbers::pi / T(180.0)));
        T sa = sin(angle * T(std::numbers::pi / T(180.0)));

        glm::mat<4, 4, T, Q> m(T(0.0));
        m[0][0] = ca;
        m[2][0] = sa;
        m[1][1] = T(1.0);
        m[0][2] = -sa;
        m[2][2] = ca;
        m[3][3] = T(1.0);

        return m;
    }

//! OpenGL matrix for rotation around z-axis by given angle (in degrees)
    template<typename T, glm::qualifier Q = glm::defaultp>
    glm::mat<4, 4, T, Q> rotation_matrix_z(T angle) {
        T ca = cos(angle * T(std::numbers::pi / 180.0));
        T sa = sin(angle * T(std::numbers::pi / 180.0));

        glm::mat<4, 4, T, Q> m(T(0.0));
        m[0][0] = ca;
        m[1][0] = -sa;
        m[0][1] = sa;
        m[1][1] = ca;
        m[2][2] = 1.0;
        m[3][3] = 1.0;

        return m;
    }

//! OpenGL matrix for rotation around given axis by given angle (in degrees)
    template<typename T, glm::qualifier Q = glm::defaultp>
    glm::mat<4, 4, T, Q> rotation_matrix(const glm::vec<3, T, Q> &axis, T angle) {
        glm::mat<4, 4, T> m(T(0.0));
        T a = angle * T(std::numbers::pi / 180.0);
        T c = cosf(a);
        T s = sinf(a);
        T one_m_c = T(1) - c;
        glm::vec<3, T, Q> ax = glm::normalize(axis);

        m[0][0] = ax[0] * ax[0] * one_m_c + c;
        m[1][0] = ax[0] * ax[1] * one_m_c - ax[2] * s;
        m[2][0] = ax[0] * ax[2] * one_m_c + ax[1] * s;

        m[0][1] = ax[1] * ax[0] * one_m_c + ax[2] * s;
        m[1][1] = ax[1] * ax[1] * one_m_c + c;
        m[2][1] = ax[1] * ax[2] * one_m_c - ax[0] * s;

        m[0][2] = ax[2] * ax[0] * one_m_c - ax[1] * s;
        m[1][2] = ax[2] * ax[1] * one_m_c + ax[0] * s;
        m[2][2] = ax[2] * ax[2] * one_m_c + c;

        m[3][3] = T(1.0);

        return m;
    }

//! OpenGL matrix for rotation specified by unit quaternion
    template<typename T, glm::qualifier Q = glm::defaultp>
    glm::mat<4, 4, T, Q> rotation_matrix(const glm::vec<4, T, Q> &quat) {
        glm::mat<4, 4, T, Q> m(T(0.0));
        T s1(1);
        T s2(2);

        m[0][0] = s1 - s2 * quat[1] * quat[1] - s2 * quat[2] * quat[2];
        m[0][1] = s2 * quat[0] * quat[1] + s2 * quat[3] * quat[2];
        m[0][2] = s2 * quat[0] * quat[2] - s2 * quat[3] * quat[1];

        m[1][0] = s2 * quat[0] * quat[1] - s2 * quat[3] * quat[2];
        m[1][1] = s1 - s2 * quat[0] * quat[0] - s2 * quat[2] * quat[2];
        m[1][2] = s2 * quat[1] * quat[2] + s2 * quat[3] * quat[0];

        m[2][0] = s2 * quat[0] * quat[2] + s2 * quat[3] * quat[1];
        m[2][1] = s2 * quat[1] * quat[2] - s2 * quat[3] * quat[0];
        m[2][2] = s1 - s2 * quat[0] * quat[0] - s2 * quat[1] * quat[1];

        m[3][3] = T(1.0);

        return m;
    }

    template<typename T, glm::qualifier Q = glm::defaultp>
    glm::mat<3, 3, T, Q> linear(const glm::mat<4, 4, T, Q> &m) {
        return glm::mat<3, 3, T, Q>(m);
    }
}

#endif //ENGINE24_TRANSFORM_H
