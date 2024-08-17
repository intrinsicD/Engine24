//
// Created by alex on 17.08.24.
//

#include "mat2.cuh"

namespace Bcg {
    __device__ __host__  mat2::mat2() : col0(), col1() {

    }

    __device__ __host__  mat2::mat2(vec2 col0, vec2 col1) : col0(col0), col1(col1) {

    }

    __device__ __host__  mat2 mat2::identity() {
        return {{1, 0},
                {0, 1}};
    }

    __device__ __host__ mat2 mat2::constant(float c) {
        return {{c, c},
                {c, c}};
    }

    __device__ __host__  mat2 mat2::reflect_x() {
        return {{1, 0},
                {0, -1}};
    }

    __device__ __host__  mat2 mat2::reflect_y() {
        return {{-1, 0},
                {0,  1}};
    }

    __device__ __host__  mat2 mat2::rot(float angle) {
        float c = cosf(angle);
        float s = sinf(angle);
        return {{c, -s},
                {s, c}};
    }

    __device__ __host__  mat2 mat2::project(float angle) {
        float c = cosf(angle);
        float s = sinf(angle);
        float cs = c * s;
        return {{c * c, cs},
                {cs,    s * s}};
    }

    __device__ __host__  mat2 mat2::shear_x(float s) {
        return {{1, s},
                {0, 1}};
    }

    __device__ __host__  mat2 mat2::shear_y(float s) {
        return {{1, 0},
                {s, 1}};
    }

    __device__ __host__  mat2 mat2::operator-() const {
        return {-col0, -col1};
    }

    __device__ __host__  vec2 mat2::operator[](int i) const {
        return *(&col0 + i);
    }

    __device__ __host__  vec2 &mat2::operator[](int i) {
        return *(&col0 + i);
    }

    __device__ __host__ const float &mat2::operator()(int r, int c) const {
        return (*this)[c][r];
    }

    __device__ __host__ float &mat2::operator()(int r, int c) {
        return (*this)[c][r];
    }

    __device__ __host__  mat2 mat2::operator+(const mat2 &b) const {
        return {col0 + b.col0, col1 + b.col1};
    }

    __device__ __host__  mat2 mat2::operator-(const mat2 &b) const {
        return {col0 - b.col0, col1 - b.col1};
    }

    __device__ __host__  mat2 mat2::operator*(const mat2 &b) const {
        return {col0 * b.col0.x + col1 * b.col0.y,
                col0 * b.col1.x + col1 * b.col1.y};
    }

    __device__ __host__  mat2 mat2::operator+(float b) const {
        return {col0 + b, col1 + b};
    }

    __device__ __host__  mat2 mat2::operator-(float b) const {
        return {col0 - b, col1 - b};
    }

    __device__ __host__  mat2 mat2::operator*(float b) const {
        return {col0 * b, col1 * b};
    }

    __device__ __host__  mat2 mat2::operator/(float b) const {
        return {col0 / b, col1 / b};
    }

    __device__ __host__  vec2 mat2::operator*(const vec2 &v) const {
        return {col0 * v.x + col1 * v.y};
    }

    __device__ __host__  mat2 mat2::transpose() const {
        return {{col0.x, col1.x},
                {col0.y, col1.y}};
    }

    __device__ __host__  float mat2::determinant() const {
        return mat2_determinant(col0.x, col0.y, col1.x, col1.y);
    }

    __device__ __host__  mat2 mat2::inverse() const {
        return transpose() / determinant();
    }

    __device__ __host__  mat2 mat2::adjoint() const {
        return {{col1.y,  -col0.y},
                {-col1.x, col0.x}};
    }

    __device__ __host__  mat2 mat2::cofactor() const {
        return adjoint();
    }

    __device__ __host__  mat2 operator+(float a, const mat2 &b) {
        return {a + b.col0, a + b.col1};
    }

    __device__ __host__  mat2 operator-(float a, const mat2 &b) {
        return {a - b.col0, a - b.col1};
    }

    __device__ __host__  mat2 operator*(float a, const mat2 &b) {
        return {a * b.col0, a * b.col1};
    }

    __device__ __host__  mat2 operator/(float a, const mat2 &b) {
        return {a / b.col0, a / b.col1};
    }

    __device__ __host__ float mat2_determinant(float a, float b, float c, float d) {
        return a * d - b * c;
    }
}