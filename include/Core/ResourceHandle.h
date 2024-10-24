//
// Created by alex on 27.07.24.
//

#ifndef ENGINE24_RESOURCEHANDLE_H
#define ENGINE24_RESOURCEHANDLE_H

namespace Bcg {
    template<typename T, class Pool>
    class ResourceHandle {
    public:
        ResourceHandle() : m_index(-1), m_pool(nullptr) {}

        ResourceHandle(unsigned int index, Pool *pool) : m_index(index), m_pool(pool) {
            if(is_valid()){
                ++m_pool->m_ref_counts[m_index];
            }
        }

        ~ResourceHandle() {
            if(is_valid()){
                --m_pool->m_ref_counts[m_index];
            }
        }

        inline unsigned int index() const {
            return m_index;
        }

        inline operator T &() {
            return m_pool->resources_[m_index];
        }

        inline operator const T &() const {
            return m_pool->resources_[m_index];
        }

        inline T &get() {
            return m_pool->resources_[m_index];
        }

        inline const T &get() const {
            return m_pool->resources_[m_index];
        }

        inline T &operator*() {
            return m_pool->resources_[m_index];
        }

        inline const T &operator*() const {
            return m_pool->resources_[m_index];
        }

        inline T *operator->() {
            return &m_pool->resources_[m_index];
        }

        inline const T *operator->() const {
            return &m_pool->resources_[m_index];
        }

        inline  bool is_deleted() const {
            return m_pool->deleted_[m_index];
        }

        inline  bool is_valid() const {
            return m_index != -1 && m_pool != nullptr && m_index < m_pool->resources_.size();
        }

        inline operator bool() const {
            return is_valid() && !is_deleted();
        }

        inline bool operator==(const ResourceHandle<T, Pool> &other) const {
            return m_index == other.m_index && m_pool == other.m_pool;
        }

        inline bool operator!=(const ResourceHandle<T, Pool> &other) const {
            return !operator==(other);
        }

    private:
        unsigned int m_index;
        Pool *m_pool;
    };
}

#endif //ENGINE24_RESOURCEHANDLE_H
