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
        explicit ResourcePool(unsigned int capacity = 100) : m_capacity(capacity),
                                                             m_resources(capacity),
                                                             m_deleted(capacity, true) {
            for (unsigned int i = 0; i < capacity; ++i) {
                m_free_list.push(i);
            }
        }

        void reserve(unsigned int capacity) {
            m_resources.reserve(capacity);
            m_deleted.reserve(capacity);
            m_capacity = capacity;
            for (unsigned int i = 0; i < capacity; ++i) {
                m_free_list.push(i);
            }
        }

        void resize(unsigned int capacity) {
            m_resources.resize(capacity);
            m_deleted.resize(capacity, true);
            m_capacity = capacity;
            for (unsigned int i = 0; i < capacity; ++i) {
                m_free_list.push(i);
            }
        }

        inline ResourceHandle<T, ResourcePool<T>> create() {
            return emplace({});
        }

        ResourceHandle<T, ResourcePool<T>> emplace(const T &resource) {
            if (m_free_list.empty()) {
                m_resources.push_back(resource);
                m_ref_counts.push_back(0);
                m_deleted.push_back(false);
                return ResourceHandle<T, ResourcePool<T>>(m_resources.size() - 1, this);
            } else {
                unsigned int index = m_free_list.front();
                m_free_list.pop();
                m_resources[index] = resource;
                m_ref_counts[index] = 0;
                m_deleted[index] = false;
                return ResourceHandle<T, ResourcePool<T>>(index, this);
            }
        }

        ResourceHandle<T, ResourcePool<T>> emplace(T &&resource) {
            if (m_free_list.empty()) {
                m_resources.emplace_back(std::move(resource));
                m_ref_counts.push_back(0);
                m_deleted.push_back(false);
                return ResourceHandle<T, ResourcePool<T>>(m_resources.size() - 1, this);
            } else {
                unsigned int index = m_free_list.front();
                m_free_list.pop();
                m_resources[index] = std::move(resource);
                m_ref_counts[index] = 0;
                m_deleted[index] = false;
                return ResourceHandle<T, ResourcePool<T>>(index, this);
            }
        }

        void remove(ResourceHandle<T, ResourcePool<T>> handle) {
            if (handle.index_ < m_resources.size()) {
                m_free_list.push(handle.index_);
                m_deleted[handle.index_] = true;
            }
        }

        void clear() {
            m_free_list = std::queue<unsigned int>();
            for (unsigned int i = 0; i < m_capacity; ++i) {
                m_free_list.push(i);
                m_deleted[i] = true;
            }
        }

        inline const T &operator[](const ResourceHandle<T, ResourcePool<T>> &handle) const {
            return m_resources[handle.index];
        }

        inline T &operator[](ResourceHandle<T, ResourcePool<T>> &handle) {
            return m_resources[handle.index];
        }

        inline const T &operator[](unsigned int index) const {
            return m_resources[index];
        }

        inline T &operator[](unsigned int index) {
            return m_resources[index];
        }

        class Iterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = ResourceHandle<T, ResourcePool<T>>;
            using pointer = value_type *;
            using reference = value_type &;

            Iterator(unsigned int index, ResourcePool *pool)
                    : handle_(index, pool) {}

            inline value_type operator*() const { return handle_; }

            inline value_type *operator->() { return &handle_; }

            // Prefix increment
            Iterator &operator++() {
                while (handle_.index_ < handle_.pool_->m_resources.size() && handle_.pool_->m_deleted[handle_.index_]) {
                    ++handle_.index_;
                }
                return *this;
            }

            Iterator &operator--() {
                while (handle_.index_ > 0 && handle_.pool_->m_deleted[handle_.index_]) {
                    --handle_.index_;
                }
                return *this;
            }

            friend bool operator==(const Iterator &a, const Iterator &b) {
                return a.handle_ == b.handle_;
            }

            friend bool operator!=(const Iterator &a, const Iterator &b) {
                return !(a == b);
            }

        private:
            ResourceHandle<T, ResourcePool<T>> handle_;
        };

        inline Iterator begin() {
            return Iterator(0, this);
        }

        inline Iterator end() {
            return Iterator(m_resources.size(), this);
        }

        inline Iterator begin() const {
            return Iterator(0, this);
        }

        inline Iterator end() const {
            return Iterator(m_resources.size(), this);
        }

        unsigned int capacity() const {
            return m_capacity;
        }

        unsigned int size() const {
            return m_resources.size();
        }

        unsigned int size_active() const {
            return m_resources.size() - m_free_list.size();
        }

        unsigned int size_free() const {
            return m_free_list.size();
        }

    private:
        friend class ResourceHandle<T, ResourcePool<T>>;

        unsigned int m_capacity;
        std::vector<T> m_resources;
        std::vector<int> m_ref_counts;
        std::vector<bool> m_deleted;
        std::queue<unsigned int> m_free_list;
    };
}

#endif //ENGINE24_RESOURCEPOOL_H
