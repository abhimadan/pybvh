#pragma once

#include <cmath>

#include "vector.h"

namespace pybvh {

Vector closestPointOnEdge(const Vector& v0, const Vector& v1, const Vector& p,
                          double& t);
Vector closestPointOnTriangle(const Vector& v0, const Vector& v1,
                              const Vector& v2, const Vector& p, double& s,
                              double& t);

} // namespace pybvh
