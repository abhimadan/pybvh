#pragma once

#include <cmath>

#include "vector.h"

namespace pybvh {

bool rayTriangleIntersection(const Vector& v0, const Vector& v1,
                             const Vector& v2, const Vector& origin,
                             const Vector& direction, double& u, double& v,
                             double& t, Vector& intersection);

} // namespace pybvh
