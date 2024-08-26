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

        inline unsigned int index() const {
            return index_;
        }

        inline operator T &() {
            return pool_->resources_[index_];
        }

        inline operator const T &() const {
            return pool_->resources_[index_];
        }

        inline T &get() {
            return pool_->resources_[index_];
        }

        inline const T &get() const {
            return pool_->resources_[index_];
        }

        inline T &operator*() {
            return pool_->resources_[index_];
        }

        inline const T &operator*() const {
            return pool_->resources_[index_];
        }

        inline T *operator->() {
            return &pool_->resources_[index_];
        }

        inline const T *operator->() const {
            return &pool_->resources_[index_];
        }

        inline  bool is_deleted() const {
            return pool_->deleted_[index_];
        }

        inline  bool is_valid() const {
            return index_ != -1 && pool_ != nullptr && index_ < pool_->resources_.size();
        }

        inline operator bool() const {
            return is_valid() && !is_deleted();
        }

        inline bool operator==(const ResourceHandle<T, Pool> &other) const {
            return index_ == other.index_ && pool_ == other.pool_;
        }

        inline bool operator!=(const ResourceHandle<T, Pool> &other) const {
            return !operator==(other);
        }

    private:
        unsigned int index_;
        Pool *pool_;
    };
}

#endif //ENGINE24_RESOURCEHANDLE_H
