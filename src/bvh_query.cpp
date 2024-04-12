#include "bvh_query.h"

#include <algorithm>
#include <iostream>

#include "closest_point.h"
#include "ray_intersection.h"

namespace pybvh {

DistResult minDistStack(int node_idx, const Vector& q, const BVHTree& tree,
                        double max_radius) {
  if (node_idx < 0 || node_idx >= tree.num_nodes) {
    return DistResult(max_radius);
  }

  const BVH* bvh = &tree.nodes[node_idx];

  if (bvh->isLeaf()) {
    Vector p;
    double u = -1;
    double v = -1;
    if (tree.Fptr != nullptr) {
      Vector v0 = bvh->getLeafCorner(0, tree.indices, tree.Vptr, tree.Fptr);
      Vector v1 = bvh->getLeafCorner(1, tree.indices, tree.Vptr, tree.Fptr);
      Vector v2 = bvh->getLeafCorner(2, tree.indices, tree.Vptr, tree.Fptr);
      p = closestPointOnTriangle(v0, v1, v2, q, u, v);
    } else if (tree.Eptr != nullptr) {
      Vector v0 = bvh->getLeafEndpoint(0, tree.indices, tree.Vptr, tree.Eptr);
      Vector v1 = bvh->getLeafEndpoint(1, tree.indices, tree.Vptr, tree.Eptr);
      p = closestPointOnEdge(v0, v1, q, u);
    } else {
      p = bvh->getLeafVertex(tree.indices, tree.Vptr);
    }
    DistResult result =
        DistResult(p, (q - p).squaredNorm(), u, v, bvh->leafIdx(tree.indices));
    return result.dist < max_radius ? result : DistResult(max_radius);
  }

  double box_dist = bvh->bounds.squaredDist(q);
  if (box_dist > max_radius) {
    return DistResult(max_radius);
  }

  const BVH* left_node =
      bvh->hasLeftChild() ? &tree.nodes[bvh->left_idx] : nullptr;
  const BVH* right_node =
      bvh->hasRightChild() ? &tree.nodes[bvh->right_idx] : nullptr;
  double left_box_dist = left_node == nullptr ? 0 : left_node->bounds.squaredDist(q);
  double right_box_dist = right_node == nullptr ? 0 : right_node->bounds.squaredDist(q);

  DistResult child_result(max_radius);
  if (left_box_dist < right_box_dist) {
    child_result = minDistStack(bvh->left_idx, q, tree, max_radius);

    DistResult far_result =
        minDistStack(bvh->right_idx, q, tree, child_result.dist);
    if (child_result.dist > far_result.dist) {
      child_result = far_result;
    }
  } else {
    child_result = minDistStack(bvh->right_idx, q, tree, max_radius);

    DistResult far_result =
        minDistStack(bvh->left_idx, q, tree, child_result.dist);
    if (child_result.dist > far_result.dist) {
      child_result = far_result;
    }
  }

  return child_result;
}

DistResult minDist(const Vector& q, const BVHTree& tree, double max_radius) {
  return minDistStack(0, q, tree, max_radius);
}

void pushOntoPQueue(KNNResult& results, int k, double max_radius,
                    const DistResult& result) {
  if (results.size() < k && result.dist < max_radius) {
    results.push(result);
  } else if (result.dist < results.top().dist) {
    results.pop();
    results.push(result);
  }
}

void knnStack(int node_idx, const Vector& q, const int k, const BVHTree& tree,
              double max_radius, KNNResult& results) {
  if (node_idx < 0 || node_idx >= tree.num_nodes) {
    return;
  }

  const BVH* bvh = &tree.nodes[node_idx];

  if (bvh->isLeaf()) {
    Vector p;
    double u = -1;
    double v = -1;
    if (tree.Fptr != nullptr) {
      Vector v0 = bvh->getLeafCorner(0, tree.indices, tree.Vptr, tree.Fptr);
      Vector v1 = bvh->getLeafCorner(1, tree.indices, tree.Vptr, tree.Fptr);
      Vector v2 = bvh->getLeafCorner(2, tree.indices, tree.Vptr, tree.Fptr);
      p = closestPointOnTriangle(v0, v1, v2, q, u, v);
    } else if (tree.Eptr != nullptr) {
      Vector v0 = bvh->getLeafEndpoint(0, tree.indices, tree.Vptr, tree.Eptr);
      Vector v1 = bvh->getLeafEndpoint(1, tree.indices, tree.Vptr, tree.Eptr);
      p = closestPointOnEdge(v0, v1, q, u);
    } else {
      p = bvh->getLeafVertex(tree.indices, tree.Vptr);
    }
    DistResult leaf_result =
        DistResult(p, (q - p).squaredNorm(), u, v, bvh->leafIdx(tree.indices));
    pushOntoPQueue(results, k, max_radius, leaf_result);
    return;
  }

  double box_dist = bvh->bounds.squaredDist(q);
  if (box_dist > max_radius) {
    return;
  }

  const BVH* left_node =
      bvh->hasLeftChild() ? &tree.nodes[bvh->left_idx] : nullptr;
  const BVH* right_node =
      bvh->hasRightChild() ? &tree.nodes[bvh->right_idx] : nullptr;
  double left_box_dist = left_node == nullptr ? 0 : left_node->bounds.squaredDist(q);
  double right_box_dist = right_node == nullptr ? 0 : right_node->bounds.squaredDist(q);

  if (left_box_dist < right_box_dist) {
    knnStack(bvh->left_idx, q, k, tree, max_radius, results);
    knnStack(bvh->right_idx, q, k, tree, max_radius, results);
  } else {
    knnStack(bvh->left_idx, q, k, tree, max_radius, results);
    knnStack(bvh->right_idx, q, k, tree, max_radius, results);
  }
}

KNNResult knn(const Vector& q, int k, const BVHTree& tree, double max_radius) {
  auto comp = [](const DistResult& r0, const DistResult& r1) {
    return r0.dist < r1.dist;
  };
  KNNResult results(comp);
  knnStack(0, q, k, tree, max_radius, results);

  return results;
}

void radiusSearchStack(int node_idx, const Vector& q, const double radius,
                       const BVHTree& tree, RadiusResult& results) {
  if (node_idx < 0 || node_idx >= tree.num_nodes) {
    return;
  }

  const BVH* bvh = &tree.nodes[node_idx];

  if (bvh->isLeaf()) {
    Vector p;
    double u = -1;
    double v = -1;
    if (tree.Fptr != nullptr) {
      Vector v0 = bvh->getLeafCorner(0, tree.indices, tree.Vptr, tree.Fptr);
      Vector v1 = bvh->getLeafCorner(1, tree.indices, tree.Vptr, tree.Fptr);
      Vector v2 = bvh->getLeafCorner(2, tree.indices, tree.Vptr, tree.Fptr);
      p = closestPointOnTriangle(v0, v1, v2, q, u, v);
    } else if (tree.Eptr != nullptr) {
      Vector v0 = bvh->getLeafEndpoint(0, tree.indices, tree.Vptr, tree.Eptr);
      Vector v1 = bvh->getLeafEndpoint(1, tree.indices, tree.Vptr, tree.Eptr);
      p = closestPointOnEdge(v0, v1, q, u);
    } else {
      p = bvh->getLeafVertex(tree.indices, tree.Vptr);
    }
    DistResult leaf_result =
        DistResult(p, (q - p).squaredNorm(), u, v, bvh->leafIdx(tree.indices));
    if (leaf_result.dist <= radius) {
      results.push_back(leaf_result);
    }
    return;
  }

  double box_dist = bvh->bounds.squaredDist(q);
  if (box_dist > radius) {
    return;
  }

  const BVH* left_node =
      bvh->hasLeftChild() ? &tree.nodes[bvh->left_idx] : nullptr;
  const BVH* right_node =
      bvh->hasRightChild() ? &tree.nodes[bvh->right_idx] : nullptr;
  double left_box_dist = left_node == nullptr ? 0 : left_node->bounds.squaredDist(q);
  double right_box_dist = right_node == nullptr ? 0 : right_node->bounds.squaredDist(q);

  if (left_box_dist < right_box_dist) {
    radiusSearchStack(bvh->left_idx, q, radius, tree, results);
    radiusSearchStack(bvh->right_idx, q, radius, tree, results);
  } else {
    radiusSearchStack(bvh->left_idx, q, radius, tree, results);
    radiusSearchStack(bvh->right_idx, q, radius, tree, results);
  }
}

RadiusResult radiusSearch(const Vector& q, double radius, const BVHTree& tree) {
  RadiusResult results;
  radiusSearchStack(0, q, radius, tree, results);
  return results;
}

HitResult closestRayIntersectionStack(int node_idx, const Vector& origin,
                                      const Vector& direction, double max_t,
                                      const BVHTree& tree) {
  if (node_idx < 0 || node_idx >= tree.num_nodes) {
    return HitResult();
  }

  const BVH* bvh = &tree.nodes[node_idx];

  if (bvh->isLeaf()) {
    // Already know Fptr is non-null
    Vector v0 = bvh->getLeafCorner(0, tree.indices, tree.Vptr, tree.Fptr);
    Vector v1 = bvh->getLeafCorner(1, tree.indices, tree.Vptr, tree.Fptr);
    Vector v2 = bvh->getLeafCorner(2, tree.indices, tree.Vptr, tree.Fptr);
    Vector intersection;
    double u, v, t;
    bool hit = rayTriangleIntersection(v0, v1, v2, origin, direction, u, v, t,
                                       intersection);
    if (!hit || t > max_t) {
      return HitResult();
    }
    return HitResult(intersection, t, u, v, bvh->leafIdx(tree.indices));
  }

  const BVH* left_node =
      bvh->hasLeftChild() ? &tree.nodes[bvh->left_idx] : nullptr;
  const BVH* right_node =
      bvh->hasRightChild() ? &tree.nodes[bvh->right_idx] : nullptr;
  double left_t, right_t;
  bool hit_left = left_node == nullptr ? false
                                       : left_node->bounds.rayIntersection(
                                             origin, direction, left_t);
  bool hit_right = right_node == nullptr ? false
                                         : right_node->bounds.rayIntersection(
                                               origin, direction, right_t);

  HitResult near_hit;
  if (hit_left && (!hit_right || left_t <= right_t)) {
    near_hit = closestRayIntersectionStack(bvh->left_idx, origin, direction,
                                           max_t, tree);
    max_t = near_hit.isHit() ? near_hit.t : max_t;
    if (hit_right) {
      HitResult far_hit = closestRayIntersectionStack(bvh->right_idx, origin,
                                                      direction, max_t, tree);
      if (far_hit.isHit() && (!near_hit.isHit() || far_hit.t < near_hit.t)) {
        near_hit = far_hit;
      }
    }
  } else if (hit_right && (!hit_left || right_t <= left_t)) {
    near_hit = closestRayIntersectionStack(bvh->right_idx, origin, direction,
                                           max_t, tree);
    max_t = near_hit.isHit() ? near_hit.t : max_t;
    if (hit_left) {
      HitResult far_hit = closestRayIntersectionStack(bvh->left_idx, origin,
                                                      direction, max_t, tree);
      if (far_hit.isHit() && (!near_hit.isHit() || far_hit.t < near_hit.t)) {
        near_hit = far_hit;
      }
    }
  }
  return near_hit;
}

HitResult closestRayIntersection(const Vector& origin, const Vector& direction,
                                 const BVHTree& tree) {
  if (tree.Fptr == nullptr) {
    // Ray intersection only supported for triangle meshes, return a miss
    return HitResult();
  }
  return closestRayIntersectionStack(0, origin, direction, INFINITY, tree);
}

void allRayIntersectionsStack(int node_idx, const Vector& origin,
                              const Vector& direction, const BVHTree& tree,
                              AllHitsResult& results) {
  if (node_idx < 0 || node_idx >= tree.num_nodes) {
    return;
  }

  const BVH* bvh = &tree.nodes[node_idx];

  if (bvh->isLeaf()) {
    // Already know Fptr is non-null
    Vector v0 = bvh->getLeafCorner(0, tree.indices, tree.Vptr, tree.Fptr);
    Vector v1 = bvh->getLeafCorner(1, tree.indices, tree.Vptr, tree.Fptr);
    Vector v2 = bvh->getLeafCorner(2, tree.indices, tree.Vptr, tree.Fptr);
    Vector intersection;
    double u, v, t;
    bool hit = rayTriangleIntersection(v0, v1, v2, origin, direction, u, v, t,
                                       intersection);
    if (hit) {
      results.emplace_back(intersection, t, u, v, bvh->leafIdx(tree.indices));
    }
    return;
  }

  const BVH* left_node =
      bvh->hasLeftChild() ? &tree.nodes[bvh->left_idx] : nullptr;
  const BVH* right_node =
      bvh->hasRightChild() ? &tree.nodes[bvh->right_idx] : nullptr;
  double left_t, right_t;
  bool hit_left = left_node == nullptr ? false
                                       : left_node->bounds.rayIntersection(
                                             origin, direction, left_t);
  bool hit_right = right_node == nullptr ? false
                                         : right_node->bounds.rayIntersection(
                                               origin, direction, right_t);

  if (hit_left) {
    allRayIntersectionsStack(bvh->left_idx, origin, direction, tree, results);
  }
  if (hit_right) {
    allRayIntersectionsStack(bvh->right_idx, origin, direction, tree, results);
  }
}

AllHitsResult allRayIntersections(const Vector& origin, const Vector& direction,
                                  const BVHTree& tree) {
  AllHitsResult results;
  if (tree.Fptr != nullptr) {
    allRayIntersectionsStack(0, origin, direction, tree, results);
  }
  return results;
}

} // namespace pybvh
