// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Copyright 2001-2005 by Computer Graphics Group, RWTH Aachen
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cassert>
#include <string>
#include <utility>
#include <vector>
#include "Exceptions.h"
#include "MatVec.h"
#include "GlmToEigen.h"
#include "GlmStringTraits.h"
#include "DimTraits.h"

namespace Bcg {
    class BasePropertyArray {
    public:
        //! Destructor.
        virtual ~BasePropertyArray() = default;

        //! Reserve memory for n elements.
        virtual void reserve(size_t n) = 0;

        //! Resize storage to hold n elements.
        virtual void resize(size_t n) = 0;

        //! Free unused memory.
        virtual void free_memory() = 0;

        //! Extend the number of elements by one.
        virtual void push_back() = 0;

        //! Let two elements swap their storage place.
        virtual void swap(size_t i0, size_t i1) = 0;

        //! Return a deep copy of self.
        [[nodiscard]] virtual BasePropertyArray *clone() const = 0;

        //! Return the name of the property
        [[nodiscard]] virtual const std::string &name() const = 0;

        [[nodiscard]] virtual std::string element_string(size_t i) const = 0;

        [[nodiscard]] virtual size_t size() const = 0;

        [[nodiscard]] virtual size_t dims() const = 0;
    };

    template<typename T, int N>
    std::ostream &operator<<(std::ostream &os, const Vector<T, N> &vec) {
        Eigen::Vector<T, N> v = MapConst(vec);
        os << v.transpose();
        return os;
    }

    template<class T>
    class PropertyArray final : public BasePropertyArray {
    public:
        using ValueType = T;
        using VectorType = std::vector<ValueType>;
        using reference = typename VectorType::reference;
        using const_reference = typename VectorType::const_reference;

        explicit PropertyArray(std::string name, T t = T())
            : name_(std::move(name)), value_(std::move(t)) {
        }

        void reserve(size_t n) override { data_.reserve(n); }

        void resize(size_t n) override { data_.resize(n, value_); }

        void push_back() override { data_.push_back(value_); }

        void free_memory() override { data_.shrink_to_fit(); }

        void swap(size_t i0, size_t i1) override {
            T d(data_[i0]);
            data_[i0] = data_[i1];
            data_[i1] = d;
        }

        [[nodiscard]] BasePropertyArray *clone() const override {
            auto *p = new PropertyArray<T>(name_, value_);
            p->data_ = data_;
            return p;
        }

        //! Get pointer to array (does not work for T==bool)
        [[nodiscard]] const T *data() const { return &data_[0]; }

        //! Get reference to the underlying vector
        std::vector<T> &vector() { return data_; }

        const std::vector<T> &vector() const { return data_; }

        //! Access the i'th element. No range check is performed!
        reference operator[](size_t idx) {
            assert(idx < data_.size());
            return data_[idx];
        }

        //! Const access to the i'th element. No range check is performed!
        const_reference operator[](size_t idx) const {
            assert(idx < data_.size());
            return data_[idx];
        }

        //! Return the name of the property
        [[nodiscard]] const std::string &name() const override { return name_; }

        [[nodiscard]] std::string element_string(size_t i) const override {
            return StringTraits<T>::ToString(data_[i]);
        }

        [[nodiscard]] size_t size() const override {
            return data_.size();
        }

        [[nodiscard]] size_t dims() const override {
            return DimTraits<T>::GetDims(value_);
        }

    private:
        std::string name_;
        VectorType data_;
        ValueType value_;
    };

    // specialization for bool properties
    template<>
    inline const bool *PropertyArray<bool>::data() const {
        assert(false);
        return nullptr;
    }

    template<class T>
    class Property {
    public:
        using reference = typename PropertyArray<T>::reference;
        using const_reference = typename PropertyArray<T>::const_reference;

        friend class PropertyContainer;

        friend class SurfaceMesh;

        friend class Graph;

        friend class PointCloud;

        explicit Property(PropertyArray<T> *p = nullptr) : parray_(p) {
        }

        void reset() { parray_ = nullptr; }

        explicit operator bool() const { return parray_ != nullptr; }

        [[nodiscard]] const std::string &name() const { return parray_->name(); }

        reference operator[](size_t i) {
            assert(parray_ != nullptr);
            return (*parray_)[i];
        }

        const_reference operator[](size_t i) const {
            assert(parray_ != nullptr);
            return (*parray_)[i];
        }

        const T *data() const {
            assert(parray_ != nullptr);
            return parray_->data();
        }

        std::vector<T> &vector() {
            assert(parray_ != nullptr);
            return parray_->vector();
        }

        const std::vector<T> &vector() const {
            assert(parray_ != nullptr);
            return parray_->vector();
        }

        [[nodiscard]] const BasePropertyArray *base() const {
            return parray_;
        }

    private:
        PropertyArray<T> &array() {
            assert(parray_ != nullptr);
            return *parray_;
        }

        const PropertyArray<T> &array() const {
            assert(parray_ != nullptr);
            return *parray_;
        }

        PropertyArray<T> *parray_;
    };

    class PropertyContainer {
    public:
        // default constructor
        PropertyContainer() = default;

        // destructor (deletes all property arrays)
        virtual ~PropertyContainer() { clear(); }

        // copy constructor: performs deep copy of property arrays
        PropertyContainer(const PropertyContainer &rhs) { operator=(rhs); }

        // assignment: performs deep copy of property arrays
        PropertyContainer &operator=(const PropertyContainer &rhs) {
            if (this != &rhs) {
                clear();
                parrays_.resize(rhs.n_properties());
                size_ = rhs.size();
                for (size_t i = 0; i < parrays_.size(); ++i) {
                    parrays_[i] = rhs.parrays_[i]->clone();
                }
            }
            return *this;
        }

        [[nodiscard]] bool empty() const { return size_ == 0; }

        // returns the current size of the property arrays
        [[nodiscard]] size_t size() const { return size_; }

        // returns the number of property arrays
        [[nodiscard]] size_t n_properties() const { return parrays_.size(); }

        // returns a vector of all property names
        [[nodiscard]] std::vector<std::string> properties(std::initializer_list<int> filter_dims = {}) const {
            //TODO figure out filtering by type, float, int, other custom types ...
            std::vector<std::string> names;
            names.reserve(parrays_.size());
            for (const auto *array: parrays_) {
                if (filter_dims.size() > 0) {
                    for (const auto &dim: filter_dims) {
                        if (array->dims() == dim) {
                            names.emplace_back(array->name());
                        }
                    }
                } else {
                    names.emplace_back(array->name());
                }
            }
            return names;
        }

        // add a property with name \p name and default value \p t
        template<class T>
        Property<T> add(const std::string &name, const T t = T()) {
            // throw exception if a property with this name already exists
            for (const auto *parray: parrays_) {
                if (parray->name() == name) {
                    const auto msg = "[PropertyContainer] A property with name \"" +
                                     name + "\" already exists.\n";
                    throw InvalidInputException(msg);
                }
            }

            // otherwise add the property
            auto *p = new PropertyArray<T>(name, t);
            p->resize(size_);
            parrays_.push_back(p);
            return Property<T>(p);
        }

        // do we have a property with a given name?
        [[nodiscard]] bool exists(const std::string &name) const {
            for (auto parray: parrays_) {
                if (parray->name() == name) {
                    return true;
                }
            }
            return false;
        }

        // get a property by its name. returns invalid property if it does not exist.
        template<class T>
        Property<T> get(const std::string &name) const {
            for (auto parray: parrays_) {
                if (parray->name() == name) {
                    return Property<T>(dynamic_cast<PropertyArray<T> *>(parray));
                }
            }
            return Property<T>();
        }

        [[nodiscard]] BasePropertyArray *get_base(const std::string &name) const {
            for (auto parray: parrays_) {
                if (parray->name() == name) {
                    return parray;
                }
            }
            return nullptr;
        }

        // returns a property if it exists, otherwise it creates it first.
        template<class T>
        Property<T> get_or_add(const std::string &name, const T t = T()) {
            Property<T> p = get<T>(name);
            if (!p) {
                p = add<T>(name, t);
            }
            return p;
        }

        // delete a property
        template<class T>
        void remove(Property<T> &h) {
            const auto end = parrays_.end();
            for (auto it = parrays_.begin(); it != end; ++it) {
                if (*it == h.parray_) {
                    delete *it;
                    parrays_.erase(it);
                    h.reset();
                    break;
                }
            }
        }

        // delete all properties
        void clear() {
            for (auto &parray: parrays_) {
                delete parray;
            }
            parrays_.clear();
            size_ = 0;
        }

        // reserve memory for n entries in all arrays
        void reserve(size_t n) const {
            for (auto parray: parrays_) {
                parray->reserve(n);
            }
        }

        // resize all arrays to size n
        void resize(size_t n) {
            for (auto &parray: parrays_) {
                parray->resize(n);
            }
            size_ = n;
        }

        // free unused space in all arrays
        void free_memory() const {
            for (auto parray: parrays_) {
                parray->free_memory();
            }
        }

        // add a new element to each vector
        void push_back() {
            for (auto &parray: parrays_) {
                parray->push_back();
            }
            ++size_;
        }

        // swap elements i0 and i1 in all arrays
        void swap(size_t i0, size_t i1) const {
            for (auto parray: parrays_) {
                parray->swap(i0, i1);
            }
        }

        [[nodiscard]] const std::vector<BasePropertyArray *> &get_parray() const {
            return parrays_;
        }

    private:
        std::vector<BasePropertyArray *> parrays_;
        size_t size_{0};
    };
} // namespace pmp
