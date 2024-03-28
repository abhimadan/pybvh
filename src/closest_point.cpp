#include "closest_point.h"

#include "bvh.h"

namespace pybvh {

Vector closestPointOnEdge(const Vector& v0, const Vector& v1, const Vector& p,
                          double& t) {
  Vector dir = v1 - v0;
  t = fmin(fmax(dir.dot(p - v0)/dir.squaredNorm(), 0.0f), 1.0f);
  return v0 + t * dir;
}

Vector closestPointOnTriangle(const Vector& v0, const Vector& v1,
                              const Vector& v2, const Vector& p, double& s,
                              double& t) {
  // Hand-coded version (based on:
  // https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf)
  Vector e1 = v1-v0;
  Vector e2 = v2-v0;
  Vector ep = v0-p;
  double a = e1.squaredNorm();
  double b = e1.dot(e2);
  double c = e2.squaredNorm();
  double d = e1.dot(ep);
  double e = e2.dot(ep);

  double cd = c*d;
  s = fma(b, e, -cd) + fma(-c, d, cd);
  double ae = a*e;
  t = fma(b, d, -ae) + fma(-a, e, ae);

  double bb = b*b;
  double det = fabs(fma(a, c, -bb) + fma(-b, b, bb)) + 1e-10;

  if (s + t <= det) {
    if (s < 0.f) {
      if (t < 0.f) {
        // Region 4: both coordinates of the global min are negative
        if (d < 0.f) {
          // On edge t=0
          t = 0.f;
          if (-d >= a) {
            s = 1.f;
          } else {
            s = -d/a;
          }
        } else {
          // On edge s=0
          s = 0.f;
          if (e >= 0.f) {
            t = 0.f;
          } else if (-e >= c) {
            t = 1.f;
          } else {
            t = -e/c;
          }
        }
      } else {
        // Region 3: s coordinate is negative and the sum is < 1
        // On edge s=0
        s = 0.f;
        if (e >= 0.f) {
          t = 0.f;
        } else if (-e >= c) {
          t = 1.f;
        } else {
          t = -e/c;
        }
      }
    } else if (t < 0.f) {
      // Region 5: t coordinate is negative and the sum is < 1
      // On edge t=0
      t = 0.f;
      if (d >= 0.f) {
        s = 0.f;
      } else if (-d >= a) {
        s = 1.f;
      } else {
        s = -d/a;
      }
    } else {
      // Region 0: in the triangle
      s /= det;
      t /= det;
    }
  } else {
    if (s < 0.f) {
      // Region 2: s coordinate is negative and sum is > 1
      double tmp0 = b+d;
      double tmp1 = c+e;
      if (tmp1 > tmp0) {
        // Edge s+t=1
        double numer = tmp1 - tmp0;
        double denom = a - 2.f*b + c;
        if (numer >= denom) {
          s = 1.f;
        } else {
          s = numer/denom;
        }
        t = 1.f-s;
      } else {
        // Edge s=0
        s = 0.f;
        if (tmp1 <= 0.f) {
          t = 1.f;
        } else if (e >= 0.f) {
          t = 0.f;
        } else {
          t = -e / c;
        }
      }
    } else if (t < 0.f) {
      // Region 6: t coordinate is negative and sum is > 1
      double tmp0 = b+e;
      double tmp1 = a+d;
      if (tmp1 > tmp0) {
        // Edge s+t=1
        double numer = tmp1 - tmp0;
        double denom = a - 2.f*b + c;
        if (numer >= denom) {
          t = 1.f;
        } else {
          t = numer/denom;
        }
        s = 1.f-t;
      } else {
        // Edge t=0
        t = 0.f;
        if (tmp1 <= 0.f) {
          s = 1.f;
        } else if (d >= 0.f) {
          s = 0.f;
        } else {
          s = -d/a;
        }
      }
    } else {
      // Region 1: both coordinates are positive and sum > 1
      // On edge s+t=1
      double numer = (c+e) - (b+d);
      if (numer <= 0.f) {
        s = 0.f;
      } else {
        double denom = a - 2.f*b + c;
        if (numer >= denom) {
          s = 1.f;
        } else {
          s = numer/denom;
        }
      }
      t = 1.f-s;
    }
  }

  return v0 + s*e1 + t*e2;
}

} // namespace pybvh
