//
// Created by alex on 29.07.24.
//

#ifndef ENGINE24_GEOMETRYCOMMON_H
#define ENGINE24_GEOMETRYCOMMON_H

namespace Bcg {
    class Handle {
    public:
        //! default constructor with invalid index
        explicit Handle(IndexType idx = BCG_MAX_INDEX) : idx_(idx) {}

        //! Get the underlying index of this handle
        inline IndexType idx() const { return idx_; }

        //! reset handle to be invalid (index=PMP_MAX_INDEX.)
        inline void reset() { idx_ = BCG_MAX_INDEX; }

        //! \return whether the handle is valid, i.e., the index is not equal to PMP_MAX_INDEX.
        inline bool is_valid() const { return idx_ != BCG_MAX_INDEX; }

        //! are two handles equal?
        auto operator<=>(const Handle &rhs) const = default;

        IndexType idx_;
    };

    class Vertex : public Handle {
        using Handle::Handle;
    };

    class Halfedge : public Handle {
        using Handle::Handle;
    };

    class Edge : public Handle {
        using Handle::Handle;
    };

    class Face : public Handle {
        using Handle::Handle;
    };

    inline std::ostream &operator<<(std::ostream &os, Vertex v) {
        return (os << 'v' << v.idx());
    }

    inline std::ostream &operator<<(std::ostream &os, Halfedge h) {
        return (os << 'h' << h.idx());
    }

    inline std::ostream &operator<<(std::ostream &os, Edge e) {
        return (os << 'e' << e.idx());
    }

    inline std::ostream &operator<<(std::ostream &os, Face f) {
        return (os << 'f' << f.idx());
    }

    template<class T>
    class VertexProperty : public Property<T> {
    public:
        explicit VertexProperty() = default;

        explicit VertexProperty(Property<T> p) : Property<T>(p) {}

        typename Property<T>::reference operator[](Vertex v) {
            return Property<T>::operator[](v.idx());
        }

        typename Property<T>::const_reference operator[](Vertex v) const {
            return Property<T>::operator[](v.idx());
        }
    };

    template<class T>
    class HalfedgeProperty : public Property<T> {
    public:
        explicit HalfedgeProperty() = default;

        explicit HalfedgeProperty(Property<T> p) : Property<T>(p) {}

        typename Property<T>::reference operator[](Halfedge h) {
            return Property<T>::operator[](h.idx());
        }

        typename Property<T>::const_reference operator[](Halfedge h) const {
            return Property<T>::operator[](h.idx());
        }
    };

    template<class T>
    class EdgeProperty : public Property<T> {
    public:
        explicit EdgeProperty() = default;

        explicit EdgeProperty(Property<T> p) : Property<T>(p) {}

        typename Property<T>::reference operator[](Edge e) {
            return Property<T>::operator[](e.idx());
        }

        typename Property<T>::const_reference operator[](Edge e) const {
            return Property<T>::operator[](e.idx());
        }
    };

    template<class T>
    class FaceProperty : public Property<T> {
    public:
        explicit FaceProperty() = default;

        explicit FaceProperty(Property<T> p) : Property<T>(p) {}

        typename Property<T>::reference operator[](Face f) {
            return Property<T>::operator[](f.idx());
        }

        typename Property<T>::const_reference operator[](Face f) const {
            return Property<T>::operator[](f.idx());
        }
    };

    template<class DataContainer, typename HandleType>
    class Iterator {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = HandleType;
        using reference = HandleType &;
        using pointer = HandleType *;
        using iterator_category = std::bidirectional_iterator_tag;

        Iterator(HandleType handle = HandleType(), const DataContainer *m = nullptr) : handle_(handle), data_(m) {
            if (data_ && data_->has_garbage()) {
                while (data_->is_valid(handle_) && data_->is_deleted(handle_)) { ++handle_.idx_; }
            }
        }

        HandleType operator*() const { return handle_; }

        auto operator<=>(const Iterator &rhs) const = default;

        Iterator &operator++() {
            ++handle_.idx_;
            assert(data_);
            while (data_->has_garbage() && data_->is_valid(handle_) && data_->is_deleted(handle_)) {
                ++handle_.idx_;
            }
            return *this;
        }

        Iterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        //! pre-decrement iterator
        Iterator &operator--() {
            --handle_.idx_;
            assert(data_);
            while (data_->has_garbage() && data_->is_valid(handle_) && data_->is_deleted(handle_)) {
                --handle_.idx_;
            }
            return *this;
        }

        //! post-decrement iterator
        Iterator operator--(int) {
            auto tmp = *this;
            --(*this);
            return tmp;
        }

    private:
        HandleType handle_;
        const DataContainer *data_;
    };

    template<class DataContainer, typename HandleType>
    class HandleContainer {
    public:
        HandleContainer(Iterator<DataContainer, HandleType> begin, Iterator<DataContainer, HandleType> end)
                : begin_(begin), end_(end) {
        }

        Iterator<DataContainer, HandleType> begin() const { return begin_; }

        Iterator<DataContainer, HandleType> end() const { return end_; }

    private:
        Iterator<DataContainer, HandleType> begin_;
        Iterator<DataContainer, HandleType> end_;
    };

    template<class DataContainer>
    class VertexAroundVertexCirculatorBase {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = Vertex;
        using reference = Vertex &;
        using pointer = Vertex *;
        using iterator_category = std::bidirectional_iterator_tag;

        //! default constructor
        VertexAroundVertexCirculatorBase(const DataContainer *data_ = nullptr,
                                     Vertex v = Vertex())
                : data__(data_) {
            if (data__)
                halfedge_ = data__->halfedge(v);
        }

        //! are two circulators equal?
        bool operator==(const VertexAroundVertexCirculatorBase &rhs) const {
            assert(data__);
            assert(data__ == rhs.data__);
            return (is_active_ && (halfedge_ == rhs.halfedge_));
        }

        //! are two circulators different?
        bool operator!=(const VertexAroundVertexCirculatorBase &rhs) const {
            return !operator==(rhs);
        }

        //! pre-increment (rotate counter-clockwise)
        VertexAroundVertexCirculatorBase &operator++() {
            assert(data__);
            halfedge_ = data__->ccw_rotated_halfedge(halfedge_);
            is_active_ = true;
            return *this;
        }

        //! post-increment (rotate counter-clockwise)
        VertexAroundVertexCirculatorBase operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        //! pre-decrement (rotate clockwise)
        VertexAroundVertexCirculatorBase &operator--() {
            assert(data__);
            halfedge_ = data__->cw_rotated_halfedge(halfedge_);
            return *this;
        }

        //! post-decrement (rotate clockwise)
        VertexAroundVertexCirculatorBase operator--(int) {
            auto tmp = *this;
            --(*this);
            return tmp;
        }

        //! get the vertex the circulator refers to
        Vertex operator*() const {
            assert(data__);
            return data__->to_vertex(halfedge_);
        }

        //! cast to bool: true if vertex is not isolated
        operator bool() const { return halfedge_.is_valid(); }

        //! \return the current halfedge
        Halfedge halfedge() const { return halfedge_; }

        // helper for C++11 range-based for-loops
        VertexAroundVertexCirculatorBase &begin() {
            is_active_ = !halfedge_.is_valid();
            return *this;
        }

        // helper for C++11 range-based for-loops
        VertexAroundVertexCirculatorBase &end() {
            is_active_ = true;
            return *this;
        }

    private:
        const DataContainer *data__;
        Halfedge halfedge_;
        bool is_active_{true}; // helper for C++11 range-based for-loops
    };

    //! this class circulates through all outgoing halfedges of a vertex.
    //! it also acts as a container-concept for C++11 range-based for loops.
    //! \sa VertexAroundVertexCirculator, halfedges(Vertex)
    template<class DataContainer>
    class HalfedgeAroundVertexCirculatorBase {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = Halfedge;
        using reference = Halfedge &;
        using pointer = Halfedge *;
        using iterator_category = std::bidirectional_iterator_tag;

        //! default constructor
        HalfedgeAroundVertexCirculatorBase(const DataContainer *data_ = nullptr,
                                       Vertex v = Vertex())
                : data__(data_) {
            if (data__)
                halfedge_ = data__->halfedge(v);
        }

        //! are two circulators equal?
        bool operator==(const HalfedgeAroundVertexCirculatorBase &rhs) const {
            assert(data__);
            assert(data__ == rhs.data__);
            return (is_active_ && (halfedge_ == rhs.halfedge_));
        }

        //! are two circulators different?
        bool operator!=(const HalfedgeAroundVertexCirculatorBase &rhs) const {
            return !operator==(rhs);
        }

        //! pre-increment (rotate counter-clockwise)
        HalfedgeAroundVertexCirculatorBase &operator++() {
            assert(data__);
            halfedge_ = data__->ccw_rotated_halfedge(halfedge_);
            is_active_ = true;
            return *this;
        }

        //! post-increment (rotate counter-clockwise)
        HalfedgeAroundVertexCirculatorBase operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        //! pre-decrement (rotate clockwise)
        HalfedgeAroundVertexCirculatorBase &operator--() {
            assert(data__);
            halfedge_ = data__->cw_rotated_halfedge(halfedge_);
            return *this;
        }

        //! post-decrement (rotate clockwise)
        HalfedgeAroundVertexCirculatorBase operator--(int) {
            auto tmp = *this;
            --(*this);
            return tmp;
        }

        //! get the halfedge the circulator refers to
        Halfedge operator*() const { return halfedge_; }

        //! cast to bool: true if vertex is not isolated
        operator bool() const { return halfedge_.is_valid(); }

        // helper for C++11 range-based for-loops
        HalfedgeAroundVertexCirculatorBase &begin() {
            is_active_ = !halfedge_.is_valid();
            return *this;
        }

        // helper for C++11 range-based for-loops
        HalfedgeAroundVertexCirculatorBase &end() {
            is_active_ = true;
            return *this;
        }

    private:
        const DataContainer *data__;
        Halfedge halfedge_;
        bool is_active_{true}; // helper for C++11 range-based for-loops
    };

    //! this class circulates through all edges incident to a vertex.
    //! it also acts as a container-concept for C++11 range-based for loops.
    //! \sa VertexAroundVertexCirculator, edges(Vertex)
    template<class DataContainer>
    class EdgeAroundVertexCirculatorBase {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = Edge;
        using reference = Edge &;
        using pointer = Edge *;
        using iterator_category = std::bidirectional_iterator_tag;

        //! default constructor
        EdgeAroundVertexCirculatorBase(const DataContainer *data_ = nullptr,
                                   Vertex v = Vertex())
                : data__(data_) {
            if (data__)
                halfedge_ = data__->halfedge(v);
        }

        //! are two circulators equal?
        bool operator==(const EdgeAroundVertexCirculatorBase &rhs) const {
            assert(data__);
            assert(data__ == rhs.data__);
            return (is_active_ && (halfedge_ == rhs.halfedge_));
        }

        //! are two circulators different?
        bool operator!=(const EdgeAroundVertexCirculatorBase &rhs) const {
            return !operator==(rhs);
        }

        //! pre-increment (rotate counter-clockwise)
        EdgeAroundVertexCirculatorBase &operator++() {
            assert(data__);
            halfedge_ = data__->ccw_rotated_halfedge(halfedge_);
            is_active_ = true;
            return *this;
        }

        //! post-increment (rotate counter-clockwise)
        EdgeAroundVertexCirculatorBase operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        //! pre-decrement (rotate clockwise)
        EdgeAroundVertexCirculatorBase &operator--() {
            assert(data__);
            halfedge_ = data__->cw_rotated_halfedge(halfedge_);
            return *this;
        }

        //! post-decrement (rotate clockwise)
        EdgeAroundVertexCirculatorBase operator--(int) {
            auto tmp = *this;
            --(*this);
            return tmp;
        }

        //! get the halfedge the circulator refers to
        Edge operator*() const { return data__->edge(halfedge_); }

        //! cast to bool: true if vertex is not isolated
        operator bool() const { return halfedge_.is_valid(); }

        // helper for C++11 range-based for-loops
        EdgeAroundVertexCirculatorBase &begin() {
            is_active_ = !halfedge_.is_valid();
            return *this;
        }

        // helper for C++11 range-based for-loops
        EdgeAroundVertexCirculatorBase &end() {
            is_active_ = true;
            return *this;
        }

    private:
        const DataContainer *data__;
        Halfedge halfedge_;
        bool is_active_{true}; // helper for C++11 range-based for-loops
    };

    //! this class circulates through all incident faces of a vertex.
    //! it also acts as a container-concept for C++11 range-based for loops.
    //! \sa VertexAroundVertexCirculator, HalfedgeAroundVertexCirculator, faces(Vertex)
    template<class DataContainer>
    class FaceAroundVertexCirculatorBase {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = Face;
        using reference = Face &;
        using pointer = Face *;
        using iterator_category = std::bidirectional_iterator_tag;

        //! construct with data_ and vertex (vertex should not be isolated!)
        FaceAroundVertexCirculatorBase(const DataContainer *m = nullptr,
                                   Vertex v = Vertex())
                : data__(m) {
            if (data__) {
                halfedge_ = data__->halfedge(v);
                if (halfedge_.is_valid() && data__->is_boundary(halfedge_))
                    operator++();
            }
        }

        //! are two circulators equal?
        bool operator==(const FaceAroundVertexCirculatorBase &rhs) const {
            assert(data__);
            assert(data__ == rhs.data__);
            return (is_active_ && (halfedge_ == rhs.halfedge_));
        }

        //! are two circulators different?
        bool operator!=(const FaceAroundVertexCirculatorBase &rhs) const {
            return !operator==(rhs);
        }

        //! pre-increment (rotates counter-clockwise)
        FaceAroundVertexCirculatorBase &operator++() {
            assert(data__ && halfedge_.is_valid());
            do {
                halfedge_ = data__->ccw_rotated_halfedge(halfedge_);
            } while (data__->is_boundary(halfedge_));
            is_active_ = true;
            return *this;
        }

        //! post-increment (rotate counter-clockwise)
        FaceAroundVertexCirculatorBase operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        //! pre-decrement (rotate clockwise)
        FaceAroundVertexCirculatorBase &operator--() {
            assert(data__ && halfedge_.is_valid());
            do
                halfedge_ = data__->cw_rotated_halfedge(halfedge_);
            while (data__->is_boundary(halfedge_));
            return *this;
        }

        //! post-decrement (rotate clockwise)
        FaceAroundVertexCirculatorBase operator--(int) {
            auto tmp = *this;
            --(*this);
            return tmp;
        }

        //! get the face the circulator refers to
        Face operator*() const {
            assert(data__ && halfedge_.is_valid());
            return data__->face(halfedge_);
        }

        //! cast to bool: true if vertex is not isolated
        operator bool() const { return halfedge_.is_valid(); }

        // helper for C++11 range-based for-loops
        FaceAroundVertexCirculatorBase &begin() {
            is_active_ = !halfedge_.is_valid();
            return *this;
        }

        // helper for C++11 range-based for-loops
        FaceAroundVertexCirculatorBase &end() {
            is_active_ = true;
            return *this;
        }

    private:
        const DataContainer *data__;
        Halfedge halfedge_;
        bool is_active_{true}; // helper for C++11 range-based for-loops
    };

    //! this class circulates through the vertices of a face.
    //! it also acts as a container-concept for C++11 range-based for loops.
    //! \sa HalfedgeAroundFaceCirculator, vertices(Face)
    template<class DataContainer>
    class VertexAroundFaceCirculatorBase {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = Vertex;
        using reference = Vertex &;
        using pointer = Vertex *;
        using iterator_category = std::bidirectional_iterator_tag;

        //! default constructor
        VertexAroundFaceCirculatorBase(const DataContainer *m = nullptr,
                                   Face f = Face())
                : data__(m) {
            if (data__)
                halfedge_ = data__->halfedge(f);
        }

        //! are two circulators equal?
        bool operator==(const VertexAroundFaceCirculatorBase &rhs) const {
            assert(data__);
            assert(data__ == rhs.data__);
            return (is_active_ && (halfedge_ == rhs.halfedge_));
        }

        //! are two circulators different?
        bool operator!=(const VertexAroundFaceCirculatorBase &rhs) const {
            return !operator==(rhs);
        }

        //! pre-increment (rotates counter-clockwise)
        VertexAroundFaceCirculatorBase &operator++() {
            assert(data__ && halfedge_.is_valid());
            halfedge_ = data__->next_halfedge(halfedge_);
            is_active_ = true;
            return *this;
        }

        //! post-increment (rotate counter-clockwise)
        VertexAroundFaceCirculatorBase operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        //! pre-decrement (rotates clockwise)
        VertexAroundFaceCirculatorBase &operator--() {
            assert(data__ && halfedge_.is_valid());
            halfedge_ = data__->prev_halfedge(halfedge_);
            return *this;
        }

        //! post-decrement (rotate clockwise)
        VertexAroundFaceCirculatorBase operator--(int) {
            auto tmp = *this;
            --(*this);
            return tmp;
        }

        //! get the vertex the circulator refers to
        Vertex operator*() const {
            assert(data__ && halfedge_.is_valid());
            return data__->to_vertex(halfedge_);
        }

        // helper for C++11 range-based for-loops
        VertexAroundFaceCirculatorBase &begin() {
            is_active_ = false;
            return *this;
        }

        // helper for C++11 range-based for-loops
        VertexAroundFaceCirculatorBase &end() {
            is_active_ = true;
            return *this;
        }

    private:
        const DataContainer *data__;
        Halfedge halfedge_;
        bool is_active_{true}; // helper for C++11 range-based for-loops
    };

    //! this class circulates through all halfedges of a face.
    //! it also acts as a container-concept for C++11 range-based for loops.
    //! \sa VertexAroundFaceCirculator, halfedges(Face)
    template<class DataContainer>
    class HalfedgeAroundFaceCirculatorBase {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = Halfedge;
        using reference = Halfedge &;
        using pointer = Halfedge *;
        using iterator_category = std::bidirectional_iterator_tag;

        //! default constructor
        HalfedgeAroundFaceCirculatorBase(const DataContainer *m = nullptr,
                                     Face f = Face())
                : data__(m) {
            if (data__)
                halfedge_ = data__->halfedge(f);
        }

        //! are two circulators equal?
        bool operator==(const HalfedgeAroundFaceCirculatorBase &rhs) const {
            assert(data__);
            assert(data__ == rhs.data__);
            return (is_active_ && (halfedge_ == rhs.halfedge_));
        }

        //! are two circulators different?
        bool operator!=(const HalfedgeAroundFaceCirculatorBase &rhs) const {
            return !operator==(rhs);
        }

        //! pre-increment (rotates counter-clockwise)
        HalfedgeAroundFaceCirculatorBase &operator++() {
            assert(data__ && halfedge_.is_valid());
            halfedge_ = data__->next_halfedge(halfedge_);
            is_active_ = true;
            return *this;
        }

        //! post-increment (rotate counter-clockwise)
        HalfedgeAroundFaceCirculatorBase operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        //! pre-decrement (rotates clockwise)
        HalfedgeAroundFaceCirculatorBase &operator--() {
            assert(data__ && halfedge_.is_valid());
            halfedge_ = data__->prev_halfedge(halfedge_);
            return *this;
        }

        //! post-decrement (rotate clockwise)
        HalfedgeAroundFaceCirculatorBase operator--(int) {
            auto tmp = *this;
            --(*this);
            return tmp;
        }

        //! get the halfedge the circulator refers to
        Halfedge operator*() const { return halfedge_; }

        // helper for C++11 range-based for-loops
        HalfedgeAroundFaceCirculatorBase &begin() {
            is_active_ = false;
            return *this;
        }

        // helper for C++11 range-based for-loops
        HalfedgeAroundFaceCirculatorBase &end() {
            is_active_ = true;
            return *this;
        }

    private:
        const DataContainer *data__;
        Halfedge halfedge_;
        bool is_active_{true}; // helper for C++11 range-based for-loops
    };
}

#endif //ENGINE24_GEOMETRYCOMMON_H
