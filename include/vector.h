#pragma once

#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

namespace pybvh {

struct Vector {
  double vals[3];

  Vector() : Vector(0) {}
  explicit Vector(double v) : Vector(v, v, v) {}
  Vector(double x, double y, double z) {
    this->x() = x;
    this->y() = y;
    this->z() = z;
  }

  inline double x() const { return vals[0]; }
  inline double& x() { return vals[0]; }
  inline double y() const { return vals[1]; }
  inline double& y() { return vals[1]; }
  inline double z() const { return vals[2]; }
  inline double& z() { return vals[2]; }

  inline double& operator()(int i) {
    assert(i >= 0 && i < 3);
    return vals[i];
  }

  inline double operator()(int i) const {
    assert(i >= 0 && i < 3);
    return vals[i];
  }

  inline double dot(const Vector& b) const {
    return x()*b.x() + y()*b.y() + z()*b.z();
  }

  inline double squaredNorm() const {
    return dot(*this);
  }

  double norm() const {
    return sqrt(squaredNorm());
  }

  void normalize();
  Vector normalized() const;

  Vector reciprocal() const;

  Vector min(const Vector& b) const {
    return Vector(fmin(x(), b.x()), fmin(y(), b.y()), fmin(z(), b.z()));
  }

  Vector max(const Vector& b) const {
    return Vector(fmax(x(), b.x()), fmax(y(), b.y()), fmax(z(), b.z()));
  }

  Vector cross(const Vector& b) const {
    return Vector(y() * b.z() - z() * b.y(), z() * b.x() - x() * b.z(),
                  x() * b.y() - y() * b.x());
  }

  double maxCoeff(int* idx) const {
    *idx = 0;
    double cur_max = vals[0];
    for (int i = 1; i < 3; i++) {
      if (vals[i] > cur_max) {
        cur_max = vals[i];
        *idx = i;
      }
    }
    return cur_max;
  }
  double maxCoeff() const {
    int idx;
    return maxCoeff(&idx);
  }

  double minCoeff(int* idx) const {
    *idx = 0;
    double cur_min = vals[0];
    for (int i = 1; i < 3; i++) {
      if (vals[i] < cur_min) {
        cur_min = vals[i];
        *idx = i;
      }
    }
    return cur_min;
  }
  double minCoeff() const {
    int idx;
    return minCoeff(&idx);
  }

  double sum() const {
    return x()+y()+z();
  }

  double prod() const {
    return x()*y()*z();
  }
};

// Standard vector ops
Vector operator+(const Vector& a, const Vector& b);
Vector& operator+=(Vector& a, const Vector& b);
Vector operator*(double k, const Vector& a);
Vector operator*(const Vector& a, double k);
Vector operator*(const Vector& a, const Vector& b);
Vector& operator*=(Vector& a, double k);
Vector operator/(const Vector& a, double k);
Vector operator/(double k, const Vector& a);
Vector operator/(const Vector& a, const Vector& b);
Vector& operator/=(Vector& a, double k);
Vector operator-(const Vector& a, const Vector& b);
Vector& operator-=(Vector& a, const Vector& b);

// "All"-based comparison operators (if "any"-based is needed, will need to
// re-evaluate this choice)
bool operator<(const Vector& a, const Vector& b);
bool operator>(const Vector& a, const Vector& b);
bool operator<=(const Vector& a, const Vector& b);
bool operator>=(const Vector& a, const Vector& b);

// Int rows for storing triangle corner indices
// TODO: make this a templated type (and the type above)
struct IndexVector3 {
  int vals[3];

  IndexVector3() {
    vals[0] = 0;
    vals[1] = 0;
    vals[2] = 0;
  }
  IndexVector3(int i0, int i1, int i2) {
    vals[0] = i0;
    vals[1] = i1;
    vals[2] = i2;
  }

  int& operator()(int i) {
    assert(i >= 0 && i < 3);
    return vals[i];
  }

  int operator()(int i) const {
    assert(i >= 0 && i < 3);
    return vals[i];
  }
};

struct IndexVector2 {
  int vals[2];

  IndexVector2() {
    vals[0] = 0;
    vals[1] = 0;
  }
  IndexVector2(int i0, int i1) {
    vals[0] = i0;
    vals[1] = i1;
  }

  int& operator()(int i) {
    assert(i >= 0 && i < 2);
    return vals[i];
  }

  int operator()(int i) const {
    assert(i >= 0 && i < 2);
    return vals[i];
  }
};

} // namespace pybvh
