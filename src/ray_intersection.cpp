#include "ray_intersection.h"

namespace pybvh {

bool rayTriangleIntersection(const Vector& v0, const Vector& v1,
                             const Vector& v2, const Vector& origin,
                             const Vector& direction, double& u, double& v,
                             double& t, Vector& intersection) {
  // We have the equation u*e1 + v*e2 = o + t*d - v0, which rearranges to:
  // t*(-d) + u*e1 + v*e2 = o - v0, or in matrix form:
  // [ -d e1 e2 ][t;u;v] = o - v0
  Vector e1 = v1-v0;
  Vector e2 = v2-v0;
  Vector d_cross_e2 = direction.cross(e2);
  Vector e1_cross_d = e1.cross(direction);
  Vector normal = e1.cross(e2);
  double det = d_cross_e2.dot(e1); // (e2 x (-d)) . e1 == (d x e2) . e1

  double eps = 1e-8;
  if (fabs(det) < eps) {
    return false; // ray parallel to triangle
  }

  double det_inv =  1.0/det;
  Vector rhs = origin - v0;

  double u_numerator = d_cross_e2.dot(rhs);
  u = u_numerator*det_inv;
  if (u < 0 || u > 1) {
    return false; // out of triangle
  }

  double v_numerator = e1_cross_d.dot(rhs);
  v = v_numerator*det_inv;
  if (v < 0 || u + v > 1) {
    return false;
  }

  double t_numerator = normal.dot(rhs);
  t = t_numerator*det_inv;
  if (t < eps) {
    return false;
  }

  intersection = origin + t*direction;
  return true;
}

} // namespace pybvh
