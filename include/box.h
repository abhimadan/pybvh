#pragma once

#include "vector.h"

namespace pybvh {

struct Box {
  Vector lower, upper;

  Box(int dim = 3)
      : lower(INFINITY), upper(-INFINITY) {}

  void expand(const Vector& p) {
    lower = lower.min(p);
    upper = upper.max(p);
  }

  void expand(const Box& b) {
    expand(b.lower);
    expand(b.upper);
  }

  Vector diagonal() const {
    return upper - lower;
  }

  Vector center() const {
    return ((upper + lower) / 2.0);
  }

  bool contains(const Vector& p) const {
    return lower <= p && p <= upper;
  }

  Vector closestPoint(const Vector& p) const {
    return upper.min(lower.max(p));
  }

  double dist(const Vector& p) const {
    return (closestPoint(p) - p).norm();
  }

  double squaredDist(const Vector& p) const {
    return (closestPoint(p) - p).squaredNorm();
  }

  double maxDist(const Vector& p) const {
    double cur_max = 0;
    Vector corner = lower;
    for (int x = 0; x < 2; x++) {
      corner(0) = x == 0 ? lower(0) : upper(0);
      for (int y = 0; y < 2; y++) {
        corner(1) = y == 0 ? lower(1) : upper(1);
        for (int z = 0; z < 2; z++) {
          corner(2) = y == 0 ? lower(2) : upper(2);
          double corner_dist = (corner - p).norm();
          if (corner_dist > cur_max) {
            cur_max = corner_dist;
          }
        }
      }
    }
    return cur_max;
  }

  double volume() const { return diagonal().prod(); }

  Vector interpolate(const Vector& t) const {
    return lower + t*diagonal();
  }

  double interpolateCoord(double t, int i) const {
    return lower(i) + t*diagonal()(i);
  }
};

} // namespace pybvh
