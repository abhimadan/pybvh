#pragma once

#include <limits>
#include <cmath>
#include <vector>
#include <iostream>

#include "box.h"
#include "vector.h"

namespace pybvh {

struct BVH {
  Box bounds;
  Box center_bounds;
  int parent_idx;
  int left_idx, right_idx;
  int begin, end;

  double num_leaves;

  bool isLeaf() const { return end - begin == 1; }
  bool hasLeftChild() const { return left_idx >= 0; }
  bool hasRightChild() const { return right_idx >= 0; }

  int leafIdx(const int* indices) const {
    assert(isLeaf());
    assert(indices != nullptr);
    return indices[begin];
  }

  // Vertices
  Vector getLeafVertex(const int* indices, const Vector* vertices) const {
    assert(isLeaf());
    assert(indices != nullptr);
    assert(vertices != nullptr);
    return vertices[leafIdx(indices)];
  }

  // Faces
  Vector getLeafCorner(int corner_idx, const int* indices,
                       const Vector* vertices,
                       const IndexVector3* faces) const {
    assert(isLeaf());
    assert(indices != nullptr);
    assert(vertices != nullptr);
    assert(faces != nullptr);
    assert(0 <= corner_idx && corner_idx < 3);
    int idx = faces[leafIdx(indices)](corner_idx);
    return vertices[idx];
  }
  Vector getLeafBarycenter(const int* indices, const Vector* vertices,
                           const IndexVector3* faces) const {
    assert(isLeaf());
    assert(indices != nullptr);
    assert(vertices != nullptr);
    assert(faces != nullptr);
    IndexVector3 fidx = faces[leafIdx(indices)];
    return (vertices[fidx(0)] + vertices[fidx(1)] + vertices[fidx(2)]) / 3;
  }

  // Edges (code is basically the same as getLeafCorner aside from the
  // endpoint_idx assertion)
  Vector getLeafEndpoint(int endpoint_idx, const int* indices,
                         const Vector* vertices,
                         const IndexVector2* edges) const {
    assert(isLeaf());
    assert(indices != nullptr);
    assert(vertices != nullptr);
    assert(edges != nullptr);
    assert(0 <= endpoint_idx && endpoint_idx < 2);
    int idx = edges[leafIdx(indices)](endpoint_idx);
    return vertices[idx];
  }
  Vector getLeafBarycenter(const int* indices, const Vector* vertices,
                           const IndexVector2* edges) const {
    assert(isLeaf());
    assert(indices != nullptr);
    assert(vertices != nullptr);
    assert(edges != nullptr);
    IndexVector2 eidx = edges[leafIdx(indices)];
    return (vertices[eidx(0)] + vertices[eidx(1)]) / 2;
  }
};

// These functions take raw pointers to reference data because they don't own
// this memory - for example, it may have come from python.
void buildBVHPoints(const Vector* Vptr, std::vector<int>& indices, int begin,
                    int end, std::vector<BVH>& nodes, int parent_idx = -1);
void buildBVHTriangles(const Vector* Vptr, const IndexVector3* Fptr,
                       std::vector<int>& indices, int begin, int end,
                       std::vector<BVH>& nodes, int parent_idx = -1);
void buildBVHEdges(const Vector* Vptr, const IndexVector2* Eptr,
                   std::vector<int>& indices, int begin, int end,
                   std::vector<BVH>& nodes, int parent_idx = -1);

// This struct only contains pointers so it can be generically used to
// reference, e.g., GPU data.
struct BVHTree {
  const BVH* nodes;
  int num_nodes;
  const int* indices;
  const Vector* Vptr;
  const IndexVector3* Fptr;
  const IndexVector2* Eptr;

  BVHTree()
      : nodes(nullptr),
        num_nodes(0),
        indices(nullptr),
        Vptr(nullptr),
        Fptr(nullptr),
        Eptr(nullptr) {}

  BVHTree(const std::vector<BVH>& nodes, const std::vector<int>& indices,
          const Vector* Vptr, const IndexVector3* Fptr,
          const IndexVector2* Eptr)
      : BVHTree(nodes.data(), nodes.size(), indices.data(), Vptr, Fptr, Eptr) {}

  BVHTree(const BVH* nodes, int num_nodes, const int* indices,
          const Vector* Vptr, const IndexVector3* Fptr,
          const IndexVector2* Eptr)
      : nodes(nodes),
        num_nodes(num_nodes),
        indices(indices),
        Vptr(Vptr),
        Fptr(Fptr),
        Eptr(Eptr) {}

  bool isValid() const { return nodes != nullptr; }
};

} // namespace pybvh
