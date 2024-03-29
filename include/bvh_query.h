#pragma once

#include <functional>
#include <vector>
#include <queue>

#include "bvh.h"
#include "vector.h"

namespace pybvh {

struct QueryResult {
  Vector point;
  double dist; // squared distance, actually
  int idx;
  // TODO: possibly add barycentric coordinates here too

  QueryResult() : dist(INFINITY), idx(-1) {}
  QueryResult(double dist) : dist(dist), idx(-1) {}
  QueryResult(Vector p, double dist, int idx) : point(p), dist(dist), idx(idx) {}
};

using QueryGreater = std::function<bool(const QueryResult&, const QueryResult&)>;

using KNNQueryResult =
    std::priority_queue<QueryResult, std::vector<QueryResult>, QueryGreater>;
void pushOntoPQueue(KNNQueryResult& results, int k, double max_radius,
                    const QueryResult& result);

QueryResult minDist(const Vector& q, const BVHTree& tree,
                    double max_radius = INFINITY);

KNNQueryResult knn(const Vector& q, int k, const BVHTree& tree,
                   double max_radius = INFINITY);

} // namespace pybvh
