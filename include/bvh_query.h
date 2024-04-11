#pragma once

#include <functional>
#include <vector>
#include <queue>

#include "bvh.h"
#include "vector.h"

namespace pybvh {

struct DistResult {
  Vector point;
  double dist; // squared distance, actually
  int idx;
  // TODO: possibly add barycentric coordinates here too

  DistResult() : dist(INFINITY), idx(-1) {}
  DistResult(double dist) : dist(dist), idx(-1) {}
  DistResult(const Vector& p, double dist, int idx)
      : point(p), dist(dist), idx(idx) {}
};

using DistComp = std::function<bool(const DistResult&, const DistResult&)>;

using KNNResult =
    std::priority_queue<DistResult, std::vector<DistResult>, DistComp>;
void pushOntoPQueue(KNNResult& results, int k, double max_radius,
                    const DistResult& result);

using RadiusResult = std::vector<DistResult>;

struct HitResult {
  Vector point;
  double t;
  double u, v;
  int idx;

  HitResult() : t(-1), u(-1), v(-1), idx(-1) {}
  HitResult(const Vector& p, double t, double u, double v, int idx)
      : point(p), t(t), u(u), v(v), idx(idx) {}

  bool isHit() const { return idx != -1; }
};

using AllHitsResult = std::vector<HitResult>;

DistResult minDist(const Vector& q, const BVHTree& tree,
                   double max_radius = INFINITY);

KNNResult knn(const Vector& q, int k, const BVHTree& tree,
              double max_radius = INFINITY);

RadiusResult radiusSearch(const Vector& q, double radius, const BVHTree& tree);

HitResult closestRayIntersection(const Vector& origin, const Vector& direction,
                                 const BVHTree& tree);

AllHitsResult allRayIntersections(const Vector& origin, const Vector& direction,
                                  const BVHTree& tree);

} // namespace pybvh
