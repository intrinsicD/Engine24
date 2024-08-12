//
// Created by alex on 27.07.24.
//

#ifndef ENGINE24_RESOURCEPOOL_H
#define ENGINE24_RESOURCEPOOL_H

#include "ResourceHandle.h"
#include <vector>
#include <queue>

namespace Bcg {
    template<typename T>
    class ResourcePool {
    public:
        explicit ResourcePool(unsigned int capacity = 100) : capacity_(capacity),
                                                             resources_(capacity),
                                                             deleted_(capacity, true) {
            for (unsigned int i = 0; i < capacity; ++i) {
                freeList_.push(i);
            }
        }

        void reserve(unsigned int capacity) {
            resources_.reserve(capacity);
            deleted_.reserve(capacity);
            capacity_ = capacity;
            for (unsigned int i = 0; i < capacity; ++i) {
                freeList_.push(i);
            }
        }

        void resize(unsigned int capacity) {
            resources_.resize(capacity);
            deleted_.resize(capacity, true);
            capacity_ = capacity;
            for (unsigned int i = 0; i < capacity; ++i) {
                freeList_.push(i);
            }
        }

        ResourceHandle<T, ResourcePool<T>> create() {
            return emplace({});
        }

        ResourceHandle<T, ResourcePool<T>> emplace(const T &resource) {
            if (freeList_.empty()) {
                resources_.push_back(resource);
                deleted_.push_back(false);
                return ResourceHandle<T, ResourcePool<T>>(resources_.size() - 1, this);
            } else {
                unsigned int index = freeList_.front();
                freeList_.pop();
                resources_[index] = resource;
                deleted_[index] = false;
                return ResourceHandle<T, ResourcePool<T>>(index, this);
            }
        }

        ResourceHandle<T, ResourcePool<T>> emplace(T &&resource) {
            if (freeList_.empty()) {
                resources_.emplace_back(std::move(resource));
                deleted_.push_back(false);
                return ResourceHandle<T, ResourcePool<T>>(resources_.size() - 1, this);
            } else {
                unsigned int index = freeList_.front();
                freeList_.pop();
                resources_[index] = std::move(resource);
                deleted_[index] = false;
                return ResourceHandle<T, ResourcePool<T>>(index, this);
            }
        }

        void remove(ResourceHandle<T, ResourcePool<T>> handle) {
            if (handle.index_ < resources_.size()) {
                freeList_.push(handle.index_);
                deleted_[handle.index_] = true;
            }
        }

        void clear() {
            freeList_ = std::queue<unsigned int>();
            for (unsigned int i = 0; i < capacity_; ++i) {
                freeList_.push(i);
                deleted_[i] = true;
            }
        }

        const T &operator[](unsigned int index) const {
            return resources_[index];
        }

        T &operator[](unsigned int index) {
            return resources_[index];
        }

        class Iterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = ResourceHandle<T, ResourcePool<T>>;
            using pointer = value_type*;
            using reference = value_type&;

            Iterator(unsigned int index, ResourcePool* pool)
                    : handle_(index, pool) {}

            value_type operator*() const { return handle_; }
            value_type* operator->() { return &handle_; }

            // Prefix increment
            Iterator& operator++() {
                ++handle_.index_;
                return *this;
            }

            // Postfix increment
            Iterator operator++(int) {
                Iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            friend bool operator==(const Iterator& a, const Iterator& b) {
                return a.handle_ == b.handle_;
            }

            friend bool operator!=(const Iterator& a, const Iterator& b) {
                return !(a == b);
            }

        private:
            ResourceHandle<T, ResourcePool<T>> handle_;
        };

        Iterator begin() {
            return Iterator(0, this);
        }

        Iterator end() {
            return Iterator(resources_.size(), this);
        }
    private:
        friend class ResourceHandle<T, ResourcePool<T>>;

        unsigned int capacity_;
        std::vector<T> resources_;
        std::vector<bool> deleted_;
        std::queue<unsigned int> freeList_;
    };
}

#endif //ENGINE24_RESOURCEPOOL_H
