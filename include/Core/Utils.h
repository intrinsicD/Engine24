//
// Created by alex on 04.07.24.
//

#ifndef ENGINE24_UTILS_H
#define ENGINE24_UTILS_H

#include <utility>
#include <cstddef>
#include <vector>
#include <string>
#include <algorithm>

namespace Bcg {
    // Ranges object to support Python-like iteration. Use with `range()`.
    template<typename T>
    struct RangeHelper {
        struct Iterator {
            T pos;
            T step;

            Iterator &operator++() {
                pos += step;
                return *this;
            }

            bool operator!=(const Iterator &other) const { return pos != other.pos; }

            T operator*() const { return pos; }
        };

        RangeHelper(T min, T max, T step = 1) : begin_(min), end_(max), step_(step) {}

        T begin_, end_, step_;

        Iterator begin() const { return {begin_, step_}; }

        Iterator end() const { return {end_, step_}; }
    };

    // Python `range()` equivalent. Construct an object to iterate over a sequence.
    template<typename T>
    inline auto Range(T min, T max, T step = 1) {
        return RangeHelper<T>{min, max, step};
    }

    // Enumerate object to support Python-like enumeration. Use with `enumerate()`.
    template<typename Iterable>
    struct EnumerateHelper {
        struct Iterator {
            using IterType = decltype(std::begin(std::declval<Iterable &>()));
            IterType iter;
            size_t pos;

            Iterator &operator++() {
                ++iter;
                ++pos;
                return *this;
            }

            bool operator!=(const Iterator &other) const { return iter != other.iter; }

            auto operator*() const { return std::make_pair(pos, *iter); }
        };

        Iterable &data;

        Iterator begin() { return {std::begin(data), 0}; }

        Iterator end() { return {std::end(data), std::size(data)}; }
    };

    template<typename Iterable>
    inline auto Enumerate(Iterable &iterable) {
        return EnumerateHelper<Iterable>{iterable};
    }

    template<typename First, typename Second>
    inline std::vector<std::pair<First, Second>> Zip(const std::vector<First> &first,
                                                     const std::vector<Second> &second) {

        size_t min_size = std::min(first.size(), second.size());
        std::vector<std::pair<First, Second>> container(min_size);

        for (size_t i = 0; i < min_size; ++i) {
            container[i] = std::make_pair(first[i], second[i]);
        }
        return container;
    }

    template<typename First, typename Second>
    inline void Unzip(const std::vector<std::pair<First, Second>> &container,
                      std::vector<First> *first = nullptr,
                      std::vector<Second> *second = nullptr) {
        if (first != nullptr) {
            first->clear();
            first->reserve(container.size());
        }

        if (second != nullptr) {
            second->clear();
            second->reserve(container.size());
        }

        if (first == nullptr && second == nullptr) return;

        for (const auto &item: container) {
            if (first != nullptr) {
                first->emplace_back(item.first);
            }
            if (second != nullptr) {
                second->emplace_back(item.second);
            }
        }
    }

    template<typename First, typename Second>
    bool Descending(const std::pair<First, Second> &lhs,
                    const std::pair<First, Second> &rhs) {
        return lhs.first > rhs.first;
    }

    template<typename First, typename Second>
    bool Ascending(const std::pair<First, Second> &lhs,
                   const std::pair<First, Second> &rhs) {
        return lhs.first < rhs.first;
    }

    template<typename First, typename Second>
    inline void SortByFirst(std::vector<First> &first, std::vector<Second> &second, bool descending = false) {
        auto container = Zip(first, second);
        std::sort(container.begin(), container.end(),
                  (descending ? Descending<First, Second> : Ascending<First, Second>));
        Unzip(container, &first, &second);
    }

    std::string ReadTextFile(const std::string &filename);

    std::vector<float> ParseNumbers(const std::string &s, unsigned int &num_lines, const char *skip_chars = nullptr, const char *delimiters = " \t\r;,");

}

#endif //ENGINE24_UTILS_H
