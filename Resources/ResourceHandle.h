//
// Created by alex on 27.07.24.
//

#ifndef ENGINE24_RESOURCEHANDLE_H
#define ENGINE24_RESOURCEHANDLE_H

namespace Bcg {
    template<typename T, class Pool>
    class ResourceHandle {
    public:
        ResourceHandle() : index_(-1), pool_(nullptr) {}

        ResourceHandle(unsigned int index, Pool *pool) : index_(index), pool_(pool) {}

        [[nodiscard]] unsigned int index() const {
            return index_;
        }

        T &get() const {
            return const_cast<T&>(pool_->resources_[index_]);
        }

        T &operator*() const {
            return const_cast<T&>(pool_->resources_[index_]);
        }

        T *operator->() const {
            return &const_cast<T&>(pool_->resources_[index_]);
        }

        [[nodiscard]] bool is_deleted() const {
            return pool_->deleted_[index_];
        }

        [[nodiscard]] bool is_valid() const {
            return index_ != -1 && pool_ != nullptr && index_ < pool_->resources_.size();
        }

        explicit operator bool() const {
            return is_valid() && !is_deleted();
        }

        bool operator==(const ResourceHandle<T, Pool> &other) const {
            return index_ == other.index_ && pool_ == other.pool_;
        }

        bool operator!=(const ResourceHandle<T, Pool> &other) const {
            return !operator==(other);
        }

        // Increment operator (prefix)
        ResourceHandle &operator++() {
            ++index_;
            return *this;
        }

        // Increment operator (postfix)
        ResourceHandle operator++(int) {
            ResourceHandle tmp = *this;
            ++(*this);
            return tmp;
        }

        // Decrement operator (prefix)
        ResourceHandle &operator--() {
            --index_;
            return *this;
        }

        // Decrement operator (postfix)
        ResourceHandle operator--(int) {
            ResourceHandle tmp = *this;
            --(*this);
            return tmp;
        }

        unsigned int index_;
        Pool *pool_;
    };
}

#endif //ENGINE24_RESOURCEHANDLE_H
