//
// Created by alex on 29.07.24.
//

#ifndef ENGINE24_POINTCLOUD_H
#define ENGINE24_POINTCLOUD_H

#include "Properties.h"
#include "Types.h"
#include "GeometryCommon.h"

namespace Bcg {
    class PointCloud {
    public:
        using VertexIterator = Iterator<PointCloud, Vertex>;
        using VertexContainer = HandleContainer<PointCloud, Vertex>;

        PointCloud();

        //! destructor
        virtual ~PointCloud();

        //! copy constructor: copies \p rhs to \p *this. performs a deep copy of all
        //! properties.
        PointCloud(const PointCloud &rhs) { operator=(rhs); }

        //! assign \p rhs to \p *this. performs a deep copy of all properties.
        PointCloud &operator=(const PointCloud &rhs);

        //! assign \p rhs to \p *this. does not copy custom properties.
        PointCloud &assign(const PointCloud &rhs);

        //! add a new vertex with position \p p
        Vertex add_vertex(const PointType &p);

        //! \return number of (deleted and valid) vertices in the mesh
        [[nodiscard]] size_t vertices_size() const { return vprops_.size(); }

        //! \return number of vertices in the mesh
        [[nodiscard]] size_t n_vertices() const { return vertices_size() - deleted_vertices_; }

        //! \return true if the mesh is empty, i.e., has no vertices
        [[nodiscard]] bool is_empty() const { return n_vertices() == 0; }

        //! clear mesh: remove all vertices, edges, faces
        virtual void clear();

        //! remove unused memory from vectors
        void free_memory();

        //! reserve memory (mainly used in file readers)
        void reserve(size_t nvertices, size_t nedges, size_t nfaces);

        //! remove deleted elements
        void garbage_collection();

        //! \return whether vertex \p v is deleted
        //! \sa garbage_collection()
        [[nodiscard]] bool is_deleted(Vertex v) const { return vdeleted_[v]; }

        //! \return whether vertex \p v is valid.
        [[nodiscard]] bool is_valid(Vertex v) const { return (v.idx() < vertices_size()); }

        //!@}
        //! \name Property handling
        //!@{

        //! add a vertex property of type \p T with name \p name and default
        //! value \p t. fails if a property named \p name exists already,
        //! since the name has to be unique. in this case it returns an
        //! invalid property
        template<class T>
        VertexProperty<T> add_vertex_property(const std::string &name,
                                              const T t = T()) {
            return VertexProperty<T>(vprops_.add<T>(name, t));
        }

        //! get the vertex property named \p name of type \p T. returns an
        //! invalid VertexProperty if the property does not exist or if the
        //! type does not match.
        template<class T>
        VertexProperty<T> get_vertex_property(const std::string &name) const {
            return VertexProperty<T>(vprops_.get<T>(name));
        }

        //! if a vertex property of type \p T with name \p name exists, it is
        //! returned. otherwise this property is added (with default value \c
        //! t)
        template<class T>
        VertexProperty<T> vertex_property(const std::string &name, const T t = T()) {
            return VertexProperty<T>(vprops_.get_or_add<T>(name, t));
        }

        //! remove the vertex property \p p
        template<class T>
        void remove_vertex_property(VertexProperty<T> &p) {
            vprops_.remove(p);
        }

        //! does the mesh have a vertex property with name \p name?
        [[nodiscard]] bool has_vertex_property(const std::string &name) const {
            return vprops_.exists(name);
        }

        //! \return the names of all vertex properties
        [[nodiscard]] std::vector<std::string> vertex_properties() const {
            return vprops_.properties();
        }

        //!@}
        //! \name Iterators and circulators
        //!@{

        //! \return start iterator for vertices
        [[nodiscard]] VertexIterator vertices_begin() const {
            return {Vertex(0), this};
        }

        //! \return end iterator for vertices
        [[nodiscard]] VertexIterator vertices_end() const {
            return {Vertex(static_cast<IndexType>(vertices_size())),
                                  this};
        }

        //! \return vertex container for C++11 range-based for-loops
        [[nodiscard]] VertexContainer vertices() const {
            return {vertices_begin(), vertices_end()};
        }

        //! position of a vertex (read only)
        [[nodiscard]] const PointType &position(Vertex v) const { return vpoint_[v]; }

        //! position of a vertex
        PointType &position(Vertex v) { return vpoint_[v]; }

        //! \return vector of point positions
        std::vector<PointType> &positions() { return vpoint_.vector(); }

        Vertex new_vertex() {
            if (vertices_size() == BCG_MAX_INDEX - 1) {
                auto what =
                        "SurfaceMesh: cannot allocate vertex, max. index reached";
                throw AllocationException(what);
            }
            vprops_.push_back();
            return Vertex(static_cast<IndexType>(vertices_size()) - 1);
        }

        // are there any deleted entities?
        [[nodiscard]] bool has_garbage() const { return has_garbage_; }

        PropertyContainer vprops_;

        // point coordinates
        VertexProperty<PointType> vpoint_;

        // markers for deleted entities
        VertexProperty<bool> vdeleted_;

        // numbers of deleted entities
        IndexType deleted_vertices_{0};

        bool has_garbage_{false};
    };
}

#endif //ENGINE24_POINTCLOUD_H
