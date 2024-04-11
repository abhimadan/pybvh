#include "vector.h"

namespace pybvh {

void Vector::normalize() {
  double l = norm();
  *this /= l;
}

Vector Vector::normalized() const {
  double l = norm();
  return (*this)/l;
}

Vector Vector::reciprocal() const {
  return Vector(1.0/vals[0], 1.0/vals[1], 1.0/vals[2]);
}

Vector operator+(const Vector& a, const Vector& b) {
  Vector c;
  c.x() = a.x() + b.x();
  c.y() = a.y() + b.y();
  c.z() = a.z() + b.z();
  return c;
}

Vector& operator+=(Vector& a, const Vector& b) {
  a.x() += b.x();
  a.y() += b.y();
  a.z() += b.z();
  return a;
}

Vector operator*(double k, const Vector& a) {
  Vector b;
  b.x() = k*a.x();
  b.y() = k*a.y();
  b.z() = k*a.z();
  return b;
}

Vector operator*(const Vector& a, double k) {
  return k*a;
}

Vector operator*(const Vector& a, const Vector& b) {
  return Vector(a(0)*b(0), a(1)*b(1), a(2)*b(2));
}

Vector& operator*=(Vector& a, double k) {
  a.x() *= k;
  a.y() *= k;
  a.z() *= k;
  return a;
}

Vector operator/(const Vector& a, double k) {
  return (1.0/k)*a;
}

Vector operator/(double k, const Vector& a) {
  return k*a.reciprocal();
}

Vector operator/(const Vector& a, const Vector& b) {
  return a*b.reciprocal();
}

Vector& operator/=(Vector& a, double k) {
  a *= 1.0/k;
  return a;
}

Vector operator-(const Vector& a, const Vector& b) {
  return a + (-1.0*b);
}

Vector& operator-(Vector& a, const Vector& b) {
  a += (-1.0)*b;
  return a;
}

bool operator<(const Vector& a, const Vector& b) {
  return a.x() < b.x() && a.y() < b.y() && a.z() < b.z();
}
bool operator>(const Vector& a, const Vector& b) {
  return a.x() > b.x() && a.y() > b.y() && a.z() > b.z();
}
// The negation of the above operators produces an "any" operator
bool operator<=(const Vector& a, const Vector& b) {
  return a.x() <= b.x() && a.y() <= b.y() && a.z() <= b.z();
}
bool operator>=(const Vector& a, const Vector& b) {
  return a.x() >= b.x() && a.y() >= b.y() && a.z() >= b.z();
}

} // namespace pybvh
