//
// Created by alex on 18.07.24.
//

#include "Intersections.h"

namespace Bcg {

    //------------------------------------------------------------------------------------------------------------------
    // AABB
    //------------------------------------------------------------------------------------------------------------------

    bool Intersect(const AABB &aabb, const Vector<float, 3> &point) {
        return (point.x() >= aabb.min.x() && point.x() <= aabb.max.x() &&
                point.y() >= aabb.min.y() && point.y() <= aabb.max.y() &&
                point.z() >= aabb.min.z() && point.z() <= aabb.max.z());
    }

    bool Intersect(const AABB &aabb, const AABB &other) {
        return (aabb.min.x() <= other.max.x() && aabb.max.x() >= other.min.x() &&
                aabb.min.y() <= other.max.y() && aabb.max.y() >= other.min.y() &&
                aabb.min.z() <= other.max.z() && aabb.max.z() >= other.min.z());
    }

    bool Intersect(const AABB &aabb, const Line &line) {
        Vector<float, 3> dir = line.end - line.start;
        Vector<float, 3> invDir = 1.0f / dir.array();

        Vector<float, 3> t0 = (aabb.min - line.start).cwiseProduct(invDir);
        Vector<float, 3> t1 = (aabb.max - line.start).cwiseProduct(invDir);

        Vector<float, 3> tmin = t0.cwiseMin(t1);
        Vector<float, 3> tmax = t0.cwiseMax(t1);

        float tminMax = tmin.maxCoeff();
        float tmaxMin = tmax.minCoeff();

        return tmaxMin >= tminMax && tmaxMin >= 0.0f;
    }

    bool Intersect(const AABB &aabb, const OBB &obb) {
        // Compute the AABB center and half extents
        Vector<float, 3> aabbHalfExtents = aabb.half_extent();
        Matrix<float, 3, 3> obbRot = obb.orientation.matrix();

        // Compute the translation vector
        Vector<float, 3> t = obb.center - aabb.center();
        t = t.transpose() * obbRot;

        // Compute the absolute orientation matrix
        Eigen::Matrix3f absOrientation = obbRot.cwiseAbs();

        // Test the AABB axes
        for (int i = 0; i < 3; ++i) {
            if (std::abs(t(i)) > (aabbHalfExtents(i) + obb.half_extent.dot(absOrientation.col(i)))) {
                return false;
            }
        }

        // Test the OBB axes
        for (int i = 0; i < 3; ++i) {
            if (std::abs(t.dot(obbRot.col(i))) > (obb.half_extent(i) + aabbHalfExtents.dot(absOrientation.row(i)))) {
                return false;
            }
        }

        // Test the cross products of the axes
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float ra = aabbHalfExtents((i + 1) % 3) * absOrientation((i + 2) % 3, j) +
                           aabbHalfExtents((i + 2) % 3) * absOrientation((i + 1) % 3, j);
                float rb = obb.half_extent((j + 1) % 3) * absOrientation(i, (j + 2) % 3) +
                           obb.half_extent((j + 2) % 3) * absOrientation(i, (j + 1) % 3);
                if (std::abs(t((i + 1) % 3) * obbRot((i + 2) % 3, j) - t((i + 2) % 3) * obbRot((i + 1) % 3, j)) >
                    (ra + rb)) {
                    return false;
                }
            }
        }

        return true;
    }

    bool Intersect(const AABB &aabb, const Plane &plane) {
        Vector<float, 3> positiveVertex = aabb.min;
        Vector<float, 3> negativeVertex = aabb.max;

        if (plane.normal.x() >= 0) {
            positiveVertex.x() = aabb.max.x();
            negativeVertex.x() = aabb.min.x();
        }
        if (plane.normal.y() >= 0) {
            positiveVertex.y() = aabb.max.y();
            negativeVertex.y() = aabb.min.y();
        }
        if (plane.normal.z() >= 0) {
            positiveVertex.z() = aabb.max.z();
            negativeVertex.z() = aabb.min.z();
        }

        if (plane.normal.dot(positiveVertex) + plane.d < 0) {
            return false;
        }

        if (plane.normal.dot(negativeVertex) + plane.d <= 0) {
            return true;
        }

        return false;
    }

    bool Intersect(const AABB &aabb, const Sphere &sphere) {
        float sqDist = 0.0f;

        for (int i = 0; i < 3; ++i) {
            float v = sphere.center[i];
            if (v < aabb.min[i]) sqDist += (aabb.min[i] - v) * (aabb.min[i] - v);
            if (v > aabb.max[i]) sqDist += (v - aabb.max[i]) * (v - aabb.max[i]);
        }

        return sqDist <= sphere.radius * sphere.radius;
    }

    bool CheckSeparatingAxis(const Vector<float, 3> &axis, const AABB &aabb, const Triangle &triangle) {
        // Project AABB vertices onto the axis
        float minAABB = std::numeric_limits<float>::max();
        float maxAABB = std::numeric_limits<float>::lowest();

        for (int i = 0; i < 8; ++i) {
            Vector<float, 3> corner = aabb.min;
            if (i & 1) corner.x() = aabb.max.x();
            if (i & 2) corner.y() = aabb.max.y();
            if (i & 4) corner.z() = aabb.max.z();
            float projection = corner.dot(axis);
            minAABB = std::min(minAABB, projection);
            maxAABB = std::max(maxAABB, projection);
        }

        // Project triangle vertices onto the axis
        Vector<float, 3> triangleVertices[] = {triangle.vertices.u, triangle.vertices.v, triangle.vertices.w};
        float minTriangle = std::numeric_limits<float>::max();
        float maxTriangle = std::numeric_limits<float>::lowest();
        for (const auto &vertex: triangleVertices) {
            float projection = vertex.dot(axis);
            minTriangle = std::min(minTriangle, projection);
            maxTriangle = std::max(maxTriangle, projection);
        }

        // Check for overlap
        return maxAABB >= minTriangle && maxTriangle >= minAABB;
    }

    bool Intersect(const AABB &aabb, const Triangle &triangle) {
        // Check the AABB face normals
        Vector<float, 3> faceNormals[] = {
                Vector<float, 3>(1, 0, 0),
                Vector<float, 3>(0, 1, 0),
                Vector<float, 3>(0, 0, 1)
        };

        for (const auto &normal: faceNormals) {
            if (!CheckSeparatingAxis(normal, aabb, triangle)) {
                return false;
            }
        }

        // Check the triangle normal
        Vector<float, 3> vu = VU(triangle);
        Vector<float, 3> wu = WU(triangle);
        Vector<float, 3> triangleNormal = vu.cross(wu).normalized();
        if (!CheckSeparatingAxis(triangleNormal, aabb, triangle)) {
            return false;
        }

        // Check cross products of edges
        Vector<float, 3> edges[] = {vu, WV(triangle), -wu};

        for (const auto &edge: edges) {
            for (const auto &normal: faceNormals) {
                Vector<float, 3> axis = edge.cross(normal).normalized();
                if (!CheckSeparatingAxis(axis, aabb, triangle)) {
                    return false;
                }
            }
        }

        return true;
    }

    //------------------------------------------------------------------------------------------------------------------
    // Frustum
    //------------------------------------------------------------------------------------------------------------------


    bool Intersect(const Frustum &frustum, const Vector<float, 3> &point) {
        for (int i = 0; i < 6; i++) {
            if (frustum.planes.array[i].normal.dot(point) + frustum.planes.array[i].d < 0) {
                return false;
            }
        }
        return true;
    }

    bool Intersect(const Frustum &frustum, const AABB &aabb) {
        for (int i = 0; i < 6; i++) {
            Vector<float, 3> positiveVertex = aabb.min;
            auto &plane = frustum.planes.array[i];

            if (plane.normal.x() >= 0) { positiveVertex.x() = aabb.max.x(); }
            if (plane.normal.y() >= 0) { positiveVertex.y() = aabb.max.y(); }
            if (plane.normal.z() >= 0) { positiveVertex.z() = aabb.max.z(); }

            // Check if the positive vertex is outside the plane
            if (plane.normal.dot(positiveVertex) + plane.d < 0) {
                return false;
            }
        }
        return true;
    }

    std::pair<float, float> ProjectOntoAxis(const Vector<float, 3> &axis, const std::vector<Vector<float, 3>> &points) {
        float min = std::numeric_limits<float>::infinity();
        float max = -std::numeric_limits<float>::infinity();

        for (const auto &vertex: points) {
            float projection = vertex.dot(axis);
            min = std::min(min, projection);
            max = std::max(max, projection);
        }

        return {min, max};
    }

    bool CheckSeparatingAxis(const Vector<float, 3> &axis, const Frustum &frustum,
                             const std::vector<Vector<float, 3>> &points) {
        // Project frustum vertices onto the axis
        auto [frustumMin, frustumMax] = ProjectOntoAxis(axis, frustum.vertices.vector);
        // Project triangle vertices onto the axis
        auto [otherMin, otherMax] = ProjectOntoAxis(axis, points);
        // Check for overlap
        return frustumMax < otherMin || otherMax < frustumMin;
    }

    bool Intersect(const Frustum &frustum, const Frustum &other) {
        // Check the normals of each of the planes of both frustums
        for (const auto &plane: frustum.planes.array) {
            if (CheckSeparatingAxis(plane.normal, frustum, other.vertices.vector)) {
                return false;
            }
        }

        for (const auto &plane: other.planes.array) {
            if (CheckSeparatingAxis(plane.normal, frustum, other.vertices.vector)) {
                return false;
            }
        }

        // Check the cross products of the edges of the frustums
        std::vector<Vector<float, 3>> frustumEdges = frustum.Edges();
        std::vector<Vector<float, 3>> otherEdges = other.Edges();

        for (const auto &edge1: frustumEdges) {
            for (const auto &edge2: otherEdges) {
                Vector<float, 3> axis = edge1.cross(edge2);
                if (axis.norm() > 1e-6 && CheckSeparatingAxis(axis, frustum, other.vertices.vector)) {
                    return false;
                }
            }
        }

        return true;
    }

    bool Intersect(const Frustum &frustum, const Line &line) {
        Vector<float, 3> d = line.vec_to_end();
        for (const auto &plane: frustum.planes.array) {
            float denom = plane.normal.dot(d);

            // Check if line is parallel to the plane
            if (std::abs(denom) > 1e-6) {
                float t = -(plane.normal.dot(line.start) + plane.d) / denom;

                // Check if the intersection point is within the line segment
                if (t >= 0.0f && t <= 1.0f) {
                    Vector<float, 3> intersection = line.start + t * d;

                    // Check if the intersection point is inside the frustum
                    bool inside = true;
                    for (const auto &testPlane: frustum.planes.array) {
                        if (testPlane.normal.dot(intersection) + testPlane.d < 0) {
                            inside = false;
                            break;
                        }
                    }

                    if (inside) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    bool Intersect(const Frustum &frustum, const OBB &obb) {
        // Check if OBB is on the negative side of any frustum plane
        for (const auto &plane: frustum.planes.array) {
            Vector<float, 3> absNormal = (obb.orientation.matrix() * plane.normal).cwiseAbs();
            float radius = absNormal.dot(obb.half_extent);
            float distance = plane.normal.dot(obb.center) + plane.d;

            if (distance < -radius) {
                return false;
            }
        }

        // Check if any of the OBB vertices are inside the frustum
        for (int i = 0; i < 8; ++i) {
            Vector<float, 3> vertex = obb.center +
                                      obb.orientation.matrix() * Vector<float, 3>(
                                              (i & 1 ? 1 : -1) * obb.half_extent.x(),
                                              (i & 2 ? 1 : -1) * obb.half_extent.y(),
                                              (i & 4 ? 1 : -1) * obb.half_extent.z());

            bool inside = true;
            for (const auto &plane: frustum.planes.array) {
                if (plane.normal.dot(vertex) + plane.d < 0) {
                    inside = false;
                    break;
                }
            }

            if (inside) {
                return true;
            }
        }

        return false;
    }

    bool Intersect(const Frustum &frustum, const Plane &plane) {
        bool positive = false;
        bool negative = false;

        for (const auto &vertex: frustum.vertices.array) {
            float distance = plane.normal.dot(vertex) + plane.d;

            if (distance > 0) {
                positive = true;
            } else if (distance < 0) {
                negative = true;
            }

            // If both sides are found, the frustum intersects the plane
            if (positive && negative) {
                return true;
            }
        }

        // If all vertices are on one side, no intersection
        return false;
    }

    bool Intersect(const Frustum &frustum, const Sphere &sphere) {
        for (const auto &plane: frustum.planes.array) {
            float distance = plane.normal.dot(sphere.center) + plane.d;

            if (distance < -sphere.radius) {
                return false;
            }
        }

        return true;
    }

    bool Intersect(const Frustum &frustum, const Triangle &triangle) {
        // Check frustum plane normals
        for (const auto &plane: frustum.planes.array) {
            if (CheckSeparatingAxis(plane.normal, frustum, triangle.vertices.vector)) {
                return false;
            }
        }

        // Check triangle normal
        Vector<float, 3> triangleNormal = Normal(triangle);
        if (CheckSeparatingAxis(triangleNormal, frustum, triangle.vertices.vector)) {
            return false;
        }

        // Check cross products of frustum edges and triangle edges
        std::vector<Vector<float, 3>> frustumEdges = frustum.Edges();
        std::vector<Vector<float, 3>> triangleEdges = triangle.Edges();

        for (const auto &frustumEdge: frustumEdges) {
            for (const auto &triangleEdge: triangleEdges) {
                Vector<float, 3> axis = frustumEdge.cross(triangleEdge);
                if (axis.squaredNorm() > 1e-6) { // Check for degenerate axis
                    if (CheckSeparatingAxis(axis.normalized(), frustum, triangle.vertices.vector)) {
                        return false;
                    }
                }
            }
        }

        return true;
    }
}