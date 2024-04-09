#include "bvh_query.h"

#include <algorithm>
#include <iostream>

#include "closest_point.h"

namespace pybvh {

QueryResult minDistStack(int node_idx, const Vector& q, const BVHTree& tree,
                         double max_radius) {
  if (node_idx < 0 || node_idx >= tree.num_nodes) {
    return QueryResult(max_radius);
  }

  const BVH* bvh = &tree.nodes[node_idx];

  if (bvh->isLeaf()) {
    Vector p;
    if (tree.Fptr != nullptr) {
      Vector v0 = bvh->getLeafCorner(0, tree.indices, tree.Vptr, tree.Fptr);
      Vector v1 = bvh->getLeafCorner(1, tree.indices, tree.Vptr, tree.Fptr);
      Vector v2 = bvh->getLeafCorner(2, tree.indices, tree.Vptr, tree.Fptr);
      double s, t;
      p = closestPointOnTriangle(v0, v1, v2, q, s, t);
    } else if (tree.Eptr != nullptr) {
      Vector v0 = bvh->getLeafEndpoint(0, tree.indices, tree.Vptr, tree.Eptr);
      Vector v1 = bvh->getLeafEndpoint(1, tree.indices, tree.Vptr, tree.Eptr);
      double t;
      p = closestPointOnEdge(v0, v1, q, t);
    } else {
      p = bvh->getLeafVertex(tree.indices, tree.Vptr);
    }
    QueryResult result = QueryResult(p, (q-p).squaredNorm(), bvh->leafIdx(tree.indices));
    return result.dist < max_radius ? result : QueryResult(max_radius);
  }

  double box_dist = bvh->bounds.squaredDist(q);
  if (box_dist > max_radius) {
    return QueryResult(max_radius);
  }

  const BVH* left_node =
      bvh->left_idx >= tree.num_nodes ? nullptr : &tree.nodes[bvh->left_idx];
  const BVH* right_node =
      bvh->right_idx >= tree.num_nodes ? nullptr : &tree.nodes[bvh->right_idx];
  double left_box_dist = left_node == nullptr ? 0 : left_node->bounds.squaredDist(q);
  double right_box_dist = right_node == nullptr ? 0 : right_node->bounds.squaredDist(q);

  QueryResult child_result(max_radius);
  if (left_box_dist < right_box_dist) {
    child_result = minDistStack(bvh->left_idx, q, tree, max_radius);

    QueryResult far_result =
        minDistStack(bvh->right_idx, q, tree, child_result.dist);
    if (child_result.dist > far_result.dist) {
      child_result = far_result;
    }
  } else {
    child_result = minDistStack(bvh->right_idx, q, tree, max_radius);

    QueryResult far_result =
        minDistStack(bvh->left_idx, q, tree, child_result.dist);
    if (child_result.dist > far_result.dist) {
      child_result = far_result;
    }
  }

  return child_result;
}

QueryResult minDist(const Vector& q, const BVHTree& tree, double max_radius) {
  return minDistStack(0, q, tree, max_radius);
}

void pushOntoPQueue(KNNQueryResult& results, int k, double max_radius,
                    const QueryResult& result) {
  if (results.size() < k && result.dist < max_radius) {
    results.push(result);
  } else if (result.dist < results.top().dist) {
    results.pop();
    results.push(result);
  }
}

void knnStack(int node_idx, const Vector& q, const int k, const BVHTree& tree,
              double max_radius, KNNQueryResult& results) {
  if (node_idx < 0 || node_idx >= tree.num_nodes) {
    return;
  }

  const BVH* bvh = &tree.nodes[node_idx];

  if (bvh->isLeaf()) {
    Vector p;
    if (tree.Fptr != nullptr) {
      Vector v0 = bvh->getLeafCorner(0, tree.indices, tree.Vptr, tree.Fptr);
      Vector v1 = bvh->getLeafCorner(1, tree.indices, tree.Vptr, tree.Fptr);
      Vector v2 = bvh->getLeafCorner(2, tree.indices, tree.Vptr, tree.Fptr);
      double s, t;
      p = closestPointOnTriangle(v0, v1, v2, q, s, t);
    } else if (tree.Eptr != nullptr) {
      Vector v0 = bvh->getLeafEndpoint(0, tree.indices, tree.Vptr, tree.Eptr);
      Vector v1 = bvh->getLeafEndpoint(1, tree.indices, tree.Vptr, tree.Eptr);
      double t;
      p = closestPointOnEdge(v0, v1, q, t);
    } else {
      p = bvh->getLeafVertex(tree.indices, tree.Vptr);
    }
    QueryResult leaf_result = QueryResult(p, (q-p).squaredNorm(), bvh->leafIdx(tree.indices));
    pushOntoPQueue(results, k, max_radius, leaf_result);
  }

  double box_dist = bvh->bounds.squaredDist(q);
  if (box_dist > max_radius) {
    return;
  }

  const BVH* left_node =
      bvh->left_idx >= tree.num_nodes ? nullptr : &tree.nodes[bvh->left_idx];
  const BVH* right_node =
      bvh->right_idx >= tree.num_nodes ? nullptr : &tree.nodes[bvh->right_idx];
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

KNNQueryResult knn(const Vector& q, int k, const BVHTree& tree, double max_radius) {
  auto comp = [](const QueryResult& r0, const QueryResult& r1) {
    return r0.dist < r1.dist;
  };
  KNNQueryResult results(comp);
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
    if (tree.Fptr != nullptr) {
      Vector v0 = bvh->getLeafCorner(0, tree.indices, tree.Vptr, tree.Fptr);
      Vector v1 = bvh->getLeafCorner(1, tree.indices, tree.Vptr, tree.Fptr);
      Vector v2 = bvh->getLeafCorner(2, tree.indices, tree.Vptr, tree.Fptr);
      double s, t;
      p = closestPointOnTriangle(v0, v1, v2, q, s, t);
    } else if (tree.Eptr != nullptr) {
      Vector v0 = bvh->getLeafEndpoint(0, tree.indices, tree.Vptr, tree.Eptr);
      Vector v1 = bvh->getLeafEndpoint(1, tree.indices, tree.Vptr, tree.Eptr);
      double t;
      p = closestPointOnEdge(v0, v1, q, t);
    } else {
      p = bvh->getLeafVertex(tree.indices, tree.Vptr);
    }
    QueryResult leaf_result = QueryResult(p, (q-p).squaredNorm(), bvh->leafIdx(tree.indices));
    if (leaf_result.dist <= radius) {
      results.push_back(leaf_result);
    }
  }

  double box_dist = bvh->bounds.squaredDist(q);
  if (box_dist > radius) {
    return;
  }

  const BVH* left_node =
      bvh->left_idx >= tree.num_nodes ? nullptr : &tree.nodes[bvh->left_idx];
  const BVH* right_node =
      bvh->right_idx >= tree.num_nodes ? nullptr : &tree.nodes[bvh->right_idx];
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

} // namespace pybvh
