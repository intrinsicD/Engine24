// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Copyright 2001-2005 by Computer Graphics Group, RWTH Aachen
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cassert>
#include <string>
#include <utility>
#include <vector>
#include <memory>

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
        [[nodiscard]] virtual std::shared_ptr<BasePropertyArray> clone() const = 0;

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
            : m_name(std::move(name)), m_value(std::move(t)) {
        }

        void reserve(size_t n) override { m_data.reserve(n); }

        void resize(size_t n) override { m_data.resize(n, m_value); }

        void push_back() override { m_data.push_back(m_value); }

        void free_memory() override { m_data.shrink_to_fit(); }

        void swap(size_t i0, size_t i1) override {
            T d(m_data[i0]);
            m_data[i0] = m_data[i1];
            m_data[i1] = d;
        }

        [[nodiscard]] std::shared_ptr<BasePropertyArray> clone() const override {
            auto sptr = std::make_shared<PropertyArray<T> >(m_name, m_value);
            sptr->m_data = m_data;
            return sptr;
        }

        //! Get pointer to array (does not work for T==bool)
        [[nodiscard]] const T *data() const { return m_data.data(); }

        //! Get reference to the underlying vector
        std::vector<T> &vector() { return m_data; }

        const std::vector<T> &vector() const { return m_data; }

        //! Access the i'th element. No range check is performed!
        reference operator[](size_t idx) {
            assert(idx < m_data.size());
            return m_data[idx];
        }

        //! Const access to the i'th element. No range check is performed!
        const_reference operator[](size_t idx) const {
            assert(idx < m_data.size());
            return m_data[idx];
        }

        //! Return the name of the property
        [[nodiscard]] const std::string &name() const override { return m_name; }

        [[nodiscard]] std::string element_string(size_t i) const override {
            return StringTraits<T>::ToString(m_data[i]);
        }

        [[nodiscard]] size_t size() const override {
            return m_data.size();
        }

        [[nodiscard]] size_t dims() const override {
            return DimTraits<T>::GetDims(m_value);
        }

    private:
        std::string m_name;
        VectorType m_data;
        ValueType m_value;
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

        explicit Property(std::shared_ptr<PropertyArray<T> > p = nullptr) : m_parray(p) {
        }

        void reset() { m_parray = nullptr; }

        explicit operator bool() const { return m_parray != nullptr; }

        [[nodiscard]] const std::string &name() const { return m_parray->name(); }

        reference operator[](size_t i) {
            assert(m_parray != nullptr);
            return (*m_parray)[i];
        }

        const_reference operator[](size_t i) const {
            assert(m_parray != nullptr);
            return (*m_parray)[i];
        }

        const T *data() const {
            assert(m_parray != nullptr);
            return m_parray->data();
        }

        std::vector<T> &vector() {
            assert(m_parray != nullptr);
            return m_parray->vector();
        }

        const std::vector<T> &vector() const {
            assert(m_parray != nullptr);
            return m_parray->vector();
        }

        [[nodiscard]] const BasePropertyArray *base() const {
            return m_parray;
        }

    private:
        PropertyArray<T> &array() {
            assert(m_parray != nullptr);
            return *m_parray;
        }

        const PropertyArray<T> &array() const {
            assert(m_parray != nullptr);
            return *m_parray;
        }

        std::shared_ptr<PropertyArray<T> > m_parray;
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
                m_parrays.resize(rhs.n_properties());
                size_ = rhs.size();
                for (size_t i = 0; i < m_parrays.size(); ++i) {
                    m_parrays[i] = rhs.m_parrays[i]->clone();
                }
            }
            return *this;
        }

        [[nodiscard]] bool empty() const { return size_ == 0; }

        // returns the current size of the property arrays
        [[nodiscard]] size_t size() const { return size_; }

        // returns the number of property arrays
        [[nodiscard]] size_t n_properties() const { return m_parrays.size(); }

        // returns a vector of all property names
        [[nodiscard]] std::vector<std::string> properties(std::initializer_list<int> filter_dims = {}) const {
            //TODO figure out filtering by type, float, int, other custom types ...
            std::vector<std::string> names;
            names.reserve(m_parrays.size());
            for (const auto &array: m_parrays) {
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
            for (const auto &parray: m_parrays) {
                if (parray->name() == name) {
                    const auto msg = "[PropertyContainer] A property with name \"" +
                                     name + "\" already exists.\n";
                    throw InvalidInputException(msg);
                }
            }

            // Correct way:
            // 1. Create the specific typed pointer first.
            auto sptr = std::make_shared<PropertyArray<T> >(name, t);
            // 2. Resize it.
            sptr->resize(size_);
            // 3. Add it to the type-erased vector (implicit up-cast).
            m_parrays.push_back(sptr);
            // 4. Return a correctly constructed Property<T> that shares ownership.
            return Property<T>(sptr);
        }

        // do we have a property with a given name?
        [[nodiscard]] bool exists(const std::string &name) const {
            for (auto &parray: m_parrays) {
                if (parray->name() == name) {
                    return true;
                }
            }
            return false;
        }

        // get a property by its name. returns invalid property if it does not exist.
        template<class T>
        Property<T> get(const std::string &name) const {
            for (auto &parray: m_parrays) {
                if (parray->name() == name) {
                    // The dynamic_pointer_cast is correct.
                    auto casted_ptr = std::dynamic_pointer_cast<PropertyArray<T> >(parray);
                    // Pass the resulting shared_ptr to the constructor.
                    if (casted_ptr) return Property<T>(casted_ptr);
                    return Property<T>();
                }
            }
            return Property<T>(); // Returns a handle with a nullptr
        }

        [[nodiscard]] BasePropertyArray *get_base(const std::string &name) const {
            for (auto &parray: m_parrays) {
                if (parray->name() == name) {
                    return parray.get();
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
            const auto end = m_parrays.end();
            for (auto it = m_parrays.begin(); it != end; ++it) {
                if (it->get() == h.m_parray.get()) {
                    it->reset();
                    m_parrays.erase(it);
                    h.reset();
                    break;
                }
            }
        }

        // delete all properties
        void clear() {
            m_parrays.clear();
            size_ = 0;
        }

        // reserve memory for n entries in all arrays
        void reserve(size_t n) {
            for (auto &parray: m_parrays) {
                parray->reserve(n);
            }
        }

        // resize all arrays to size n
        void resize(size_t n) {
            for (auto &parray: m_parrays) {
                parray->resize(n);
            }
            size_ = n;
        }

        // free unused space in all arrays
        void free_memory() {
            for (auto &parray: m_parrays) {
                parray->free_memory();
            }
        }

        // add a new element to each vector
        void push_back() {
            for (auto &parray: m_parrays) {
                parray->push_back();
            }
            ++size_;
        }

        // swap elements i0 and i1 in all arrays
        void swap(size_t i0, size_t i1) const {
            for (auto &parray: m_parrays) {
                parray->swap(i0, i1);
            }
        }

        [[nodiscard]] const std::vector<std::shared_ptr<BasePropertyArray> > &get_parray() const {
            return m_parrays;
        }

    private:
        std::vector<std::shared_ptr<BasePropertyArray> > m_parrays;
        size_t size_{0};
    };
} // namespace pmp
