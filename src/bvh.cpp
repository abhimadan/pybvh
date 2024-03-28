#include "bvh.h"

#include <algorithm>
#include <iostream>
#include <array>

namespace pybvh {

void fillInternalNode(BVH* node, const std::vector<BVH>& nodes) {
  const BVH* left_node = &nodes[node->left_idx];
  const BVH* right_node = &nodes[node->right_idx];

  node->bounds.expand(left_node->bounds);
  node->bounds.expand(right_node->bounds);
  node->center_bounds.expand(left_node->center_bounds);
  node->center_bounds.expand(right_node->center_bounds);
}

void fillPointLeafNode(BVH* node, const Vector* Vptr, const int* indices_buffer) {
  node->left_idx = -1;
  node->right_idx = -1;

  node->bounds.expand(node->getLeafVertex(indices_buffer, Vptr));
  node->center_bounds.expand(node->getLeafVertex(indices_buffer, Vptr));
}

void fillTriLeafNode(BVH* node, const Vector* Vptr, const IndexVector3* Fptr,
                     const int* indices_buffer) {
  node->left_idx = -1;
  node->right_idx = -1;

  int idx = node->leafIdx(indices_buffer);
  Vector v0 = node->getLeafCorner(0, indices_buffer, Vptr, Fptr);
  Vector v1 = node->getLeafCorner(1, indices_buffer, Vptr, Fptr);
  Vector v2 = node->getLeafCorner(2, indices_buffer, Vptr, Fptr);
  Vector barycenter = (v0 + v1 + v2) / 3;
  node->bounds.expand(v0);
  node->bounds.expand(v1);
  node->bounds.expand(v2);
  node->center_bounds.expand(barycenter);
}

void fillEdgeLeafNode(BVH* node, const Vector* Vptr, const IndexVector2* Eptr,
                      const int* indices_buffer) {
  node->left_idx = -1;
  node->right_idx = -1;

  int idx = node->leafIdx(indices_buffer);
  Vector v0 = node->getLeafEndpoint(0, indices_buffer, Vptr, Eptr);
  Vector v1 = node->getLeafEndpoint(1, indices_buffer, Vptr, Eptr);
  Vector barycenter = (v0 + v1) / 2;
  node->bounds.expand(v0);
  node->bounds.expand(v1);
  node->center_bounds.expand(barycenter);
}

void buildBVHPoints(const Vector* Vptr, std::vector<int>& indices, int begin,
                    int end, std::vector<BVH>& nodes, int parent_idx) {
  if (end <= begin) {
    return;
  }
  int node_idx = nodes.size();
  nodes.emplace_back();
  BVH* node = &nodes[node_idx];
  node->begin = begin;
  node->end = end;
  node->num_leaves = end-begin;
  node->parent_idx = parent_idx;
  for (int i = begin; i < end; i++) {
    int idx = indices[i];
    node->bounds.expand(Vptr[idx]);
    node->center_bounds.expand(Vptr[idx]);
  }
  if (node->isLeaf()) { // just one point
    fillPointLeafNode(node, Vptr, indices.data());
    return;
  }

  int dim_idx;
  node->center_bounds.diagonal().maxCoeff(&dim_idx);
  double midpoint = node->center_bounds.center()(dim_idx);
  Box left_box, right_box;
  left_box = right_box = node->center_bounds;
  right_box.lower(dim_idx) = left_box.upper(dim_idx) = midpoint;

  const int* split_ptr =
      std::partition(&indices[begin], &indices[end],
                     [&](int idx) { return left_box.contains(Vptr[idx]); });
  int split = split_ptr - &indices[0];

  if (split-begin == 0 || end-split == 0) {
    // Fallback to split interval in half - should only happen if lots of
    // centroids are in the same location (I guess the real solution is to
    // expand the notion of a leaf node to include multiple primitives...)
    split = (begin+end)/2;
  }
  assert(split-begin != 0 && end-split != 0);

  node->left_idx = nodes.size();
  buildBVHPoints(Vptr, indices, begin, split, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  node->right_idx = nodes.size();
  buildBVHPoints(Vptr, indices, split, end, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  // This might cause a segfault but I think it should be ok, since buildBVH
  // should always produce at least a leaf node
  fillInternalNode(node, nodes);
}

void buildBVHTriangles(const Vector* Vptr, const IndexVector3* Fptr,
                       std::vector<int>& indices, int begin, int end,
                       std::vector<BVH>& nodes, int parent_idx) {
  if (end <= begin) {
    return;
  }
  int node_idx = nodes.size();
  nodes.emplace_back();
  BVH* node = &nodes[node_idx];
  node->begin = begin;
  node->end = end;
  node->num_leaves = end-begin;
  node->parent_idx = parent_idx;
  for (int i = begin; i < end; i++) {
    IndexVector3 fidx = Fptr[indices[i]];
    Vector corners[3];
    for (int j = 0; j < 3; j++) {
      corners[j] = Vptr[fidx(j)];
      node->bounds.expand(corners[j]);
    }
    Vector barycenter = (corners[0] + corners[1] + corners[2]) / 3;
    node->center_bounds.expand(barycenter);
  }
  if (node->isLeaf()) {
    fillTriLeafNode(node, Vptr, Fptr, indices.data());
    return;
  }

  int dim_idx;
  node->center_bounds.diagonal().maxCoeff(&dim_idx);
  double midpoint = node->center_bounds.center()(dim_idx);
  Box left_box, right_box;
  left_box = right_box = node->center_bounds;
  right_box.lower(dim_idx) = left_box.upper(dim_idx) = midpoint;

  const int* split_idx =
      std::partition(&indices[begin], &indices[end], [&](int idx) {
        IndexVector3 fidx = Fptr[idx];
        Vector v0 = Vptr[fidx(0)];
        Vector v1 = Vptr[fidx(1)];
        Vector v2 = Vptr[fidx(2)];
        Vector barycenter = (v0 + v1 + v2) / 3;
        return left_box.contains(barycenter);
      });
  int split = split_idx - &indices[0];

  if (split-begin == 0 || end-split == 0) {
    // Fallback to split interval in half - should only happen if lots of
    // centroids are in the same location (I guess the real solution is to
    // expand the notion of a leaf node to include multiple primitives...)
    split = (begin+end)/2;
  }
  assert(split-begin != 0 && end-split != 0);

  node->left_idx = nodes.size();
  buildBVHTriangles(Vptr, Fptr, indices, begin, split, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  node->right_idx = nodes.size();
  buildBVHTriangles(Vptr, Fptr, indices, split, end, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  fillInternalNode(node, nodes);
}

void buildBVHEdges(const Vector* Vptr, const IndexVector2* Eptr,
                   std::vector<int>& indices, int begin, int end,
                   std::vector<BVH>& nodes, int parent_idx) {
  if (end <= begin) {
    return;
  }
  int node_idx = nodes.size();
  nodes.emplace_back();
  BVH* node = &nodes[node_idx];
  node->begin = begin;
  node->end = end;
  node->num_leaves = end-begin;
  node->parent_idx = parent_idx;
  for (int i = begin; i < end; i++) {
    IndexVector2 eidx = Eptr[indices[i]];
    Vector corners[2];
    for (int j = 0; j < 2; j++) {
      corners[j] = Vptr[eidx(j)];
      node->bounds.expand(corners[j]);
    }
    Vector barycenter = (corners[0] + corners[1]) / 2;
    node->center_bounds.expand(barycenter);
  }
  if (node->isLeaf()) {
    fillEdgeLeafNode(node, Vptr, Eptr, indices.data());
    return;
  }

  int dim_idx;
  node->center_bounds.diagonal().maxCoeff(&dim_idx);
  double midpoint = node->center_bounds.center()(dim_idx);
  Box left_box, right_box;
  left_box = right_box = node->center_bounds;
  right_box.lower(dim_idx) = left_box.upper(dim_idx) = midpoint;

  const int* split_idx = std::partition(&indices[begin], &indices[end],
      [&](int idx) {
        IndexVector2 eidx = Eptr[idx];
        Vector v0 = Vptr[eidx(0)];
        Vector v1 = Vptr[eidx(1)];
        Vector barycenter = (v0 + v1) / 2;
        return left_box.contains(barycenter);
      });
  int split = split_idx - &indices[0];

  if (split-begin == 0 || end-split == 0) {
    // Fallback to split interval in half - should only happen if lots of
    // centroids are in the same location (I guess the real solution is to
    // expand the notion of a leaf node to include multiple primitives...)
    split = (begin+end)/2;
  }
  assert(split-begin != 0 && end-split != 0);

  node->left_idx = nodes.size();
  buildBVHEdges(Vptr, Eptr, indices, begin, split, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  node->right_idx = nodes.size();
  buildBVHEdges(Vptr, Eptr, indices, split, end, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  fillInternalNode(node, nodes);
}

} // namespace pybvh
