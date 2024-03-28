#pragma once

#include "bvh.h"
#include "vector.h"

namespace pybvh {

struct QueryResult {
  Vector point;
  double dist;
  int idx;
  // TODO: possibly add barycentric coordinates here too

  QueryResult(double dist) : dist(dist), idx(-1) {}
  QueryResult(Vector p, double dist, int idx) : point(p), dist(dist), idx(idx) {}
};

// what does the query interface look like? do we just support one point and let
// pybind11 vectorize it automatically? what if we want a parallelized loop?
// just do one query for now and go from there
QueryResult minDist(const Vector& q, const BVHTree& tree,
                    double max_radius = INFINITY);

} // namespace pybvh
