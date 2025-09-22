#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>

namespace Bcg {
    template<typename T>
    class BoundedHeap {
    public:
        explicit BoundedHeap(size_t max_size) : max_size_(max_size) {
            // Pre-allocate memory to avoid reallocations during the "filling" phase.
            data_.reserve(max_size);
        }

        // Main logic: add an item to the heap.
        void push(const T &item) {
            if (data_.size() < max_size_) {
                // Phase 1: Heap is not full yet. Just add the element.
                data_.push_back(item);
                std::push_heap(data_.begin(), data_.end());
            } else if (item < data_.front()) {
                // Phase 2: Heap is full.
                // If the new item is smaller than the largest item in the heap,
                // replace the largest with the new item.

                // 1. Move the largest element to the end of the vector.
                std::pop_heap(data_.begin(), data_.end());
                // 2. Overwrite it with the new, smaller item.
                data_.back() = item;
                // 3. Restore the heap property.
                std::push_heap(data_.begin(), data_.end());
            }
        }

        // Get the largest element in the heap.
        const T &top() const {
            return data_.front();
        }

        // Get the current size of the heap.
        size_t size() const {
            return data_.size();
        }

        // The heap itself is not sorted. This extracts the elements and sorts them.
        std::vector<T> get_sorted_data() {
            std::sort(data_.begin(), data_.end());
            return data_;
        }

    private:
        size_t max_size_;
        std::vector<T> data_;
    };
}
