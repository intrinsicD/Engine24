//
// Created by alex on 08.08.24.
//

#ifndef ENGINE24_HEAP_CUH
#define ENGINE24_HEAP_CUH

#include <thrust/swap.h>

namespace lbvh{
    struct HeapElement {
        float distance;
        int index;

        __device__ __host__
        bool operator<(const HeapElement& other) const {
            return distance < other.distance; // For min-heap
        }
    };

    inline __device__ void push_heap(HeapElement* heap, int& heap_size, HeapElement elem, int k) {
        if (heap_size < k) {
            heap[heap_size++] = elem;
            int i = heap_size - 1;
            while (i > 0 && heap[(i - 1) / 2] < heap[i]) {
                thrust::swap(heap[(i - 1) / 2], heap[i]);
                i = (i - 1) / 2;
            }
        } else if (elem < heap[0]) {
            heap[0] = elem;
            int i = 0;
            while (true) {
                int left = 2 * i + 1;
                int right = 2 * i + 2;
                int largest = i;
                if (left < heap_size && heap[left] < heap[largest]) largest = left;
                if (right < heap_size && heap[right] < heap[largest]) largest = right;
                if (largest == i) break;
                thrust::swap(heap[i], heap[largest]);
                i = largest;
            }
        }
    }
}

#endif //ENGINE24_HEAP_CUH
