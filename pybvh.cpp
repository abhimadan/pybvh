#include <pybind11/pybind11.h>

#include <iostream>
#include <vector>

#include "bvh.h"
#include "bvh_query.h"

namespace py = pybind11;

// This structure owns all the cpu memory it needs, and the binding should make
// sure that the lifetime of this struct is longer than the geometry arrays used
// to build it
struct PyBVHTree {
  std::vector<pybvh::BVH> nodes;
  std::vector<int> indices;
  std::vector<pybvh::Vector> V;
  std::vector<pybvh::IndexVector3> F;
  std::vector<pybvh::IndexVector2> E;

  pybvh::BVHTree tree_ptrs;  // this is what the C++ API uses

  void fillV(const double* Vdata, int num_vertices, const py::ssize_t strides[2]) {
    V.reserve(num_vertices);
    for (int i = 0; i < num_vertices; i++) {
      double x = *(Vdata + i*strides[0] + 0*strides[1]);
      double y = *(Vdata + i*strides[0] + 1*strides[1]);
      double z = *(Vdata + i*strides[0] + 2*strides[1]);
      V.emplace_back(x, y, z);
    }
  }

  void fillF(const int* Fdata, int num_triangles, const py::ssize_t strides[2]) {
    F.reserve(num_triangles);
    for (int i = 0; i < num_triangles; i++) {
      int i0 = *(Fdata + i*strides[0] + 0*strides[1]);
      int i1 = *(Fdata + i*strides[0] + 1*strides[1]);
      int i2 = *(Fdata + i*strides[0] + 2*strides[1]);
      F.emplace_back(i0, i1, i2);
    }
  }

  void fillE(const int* Edata, int num_edges, const py::ssize_t strides[2]) {
    E.reserve(num_edges);
    for (int i = 0; i < num_edges; i++) {
      int i0 = *(Edata + i*strides[0] + 0*strides[1]);
      int i1 = *(Edata + i*strides[0] + 1*strides[1]);
      E.emplace_back(i0, i1);
    }
  }

  void finishBuild() {
    const pybvh::Vector* Vptr = V.empty() ? nullptr : V.data();
    const pybvh::IndexVector3* Fptr = F.empty() ? nullptr : F.data();
    const pybvh::IndexVector2* Eptr = E.empty() ? nullptr : E.data();

    tree_ptrs = pybvh::BVHTree(nodes, indices, Vptr, Fptr, Eptr);
  }
};

PyBVHTree buildBVHPoints(py::buffer V_py) {
  // Request buffer info
  py::buffer_info V_info = V_py.request();

  // Do some validation checks on the buffer
  if (V_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (V_info.ndim != 2) {
    throw std::runtime_error("Incompatible dimension: expected a 2d array");
  }
  if (V_info.shape[1] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 columns");
  }

  PyBVHTree tree;

  // Copy the geometry into the row-major buffers we use
  int num_vertices = V_info.shape[0];
  py::ssize_t Vstrides[2] = {V_info.strides[0] / (py::ssize_t)sizeof(double),
                             V_info.strides[1] / (py::ssize_t)sizeof(double)};
  auto Vdata = static_cast<const double*>(V_info.ptr);
  tree.fillV(Vdata, num_vertices, Vstrides);

  tree.indices.reserve(num_vertices);
  for (int i = 0; i < num_vertices; i++) {
    tree.indices.push_back(i);
  }
  pybvh::buildBVHPoints(tree.V.data(), tree.indices, 0, num_vertices,
                        tree.nodes);

  tree.finishBuild();
  return tree;
}

PyBVHTree buildBVHTriangles(py::buffer V_py, py::buffer F_py) {
  // Request buffer info
  py::buffer_info V_info = V_py.request();
  py::buffer_info F_info = F_py.request();

  // Do some validation checks on the buffers
  if (V_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (F_info.format != py::format_descriptor<int>::format()) {
    throw std::runtime_error("Incompatible data format: expected a int array");
  }
  if (V_info.ndim != 2) {
    throw std::runtime_error("Incompatible dimension: expected a 2d array");
  }
  if (F_info.ndim != 2) {
    throw std::runtime_error("Incompatible dimension: expected a 2d array");
  }
  if (V_info.shape[1] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 columns");
  }
  if (F_info.shape[1] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 columns");
  }

  PyBVHTree tree;

  // Copy the geometry into the row-major buffers we use
  int num_vertices = V_info.shape[0];
  auto Vdata = static_cast<const double*>(V_info.ptr);
  py::ssize_t Vstrides[2] = {V_info.strides[0] / (py::ssize_t)sizeof(double),
                             V_info.strides[1] / (py::ssize_t)sizeof(double)};
  tree.fillV(Vdata, num_vertices, Vstrides);

  int num_triangles = F_info.shape[0];
  auto Fdata = static_cast<const int*>(F_info.ptr);
  py::ssize_t Fstrides[2] = {F_info.strides[0] / (py::ssize_t)sizeof(int),
                             F_info.strides[1] / (py::ssize_t)sizeof(int)};
  tree.fillF(Fdata, num_triangles, Fstrides);

  tree.indices.reserve(num_triangles);
  for (int i = 0; i < num_triangles; i++) {
    tree.indices.push_back(i);
  }
  pybvh::buildBVHTriangles(tree.V.data(), tree.F.data(), tree.indices, 0,
                           num_triangles, tree.nodes);

  tree.finishBuild();
  return tree;
}

PyBVHTree buildBVHEdges(py::buffer V_py, py::buffer E_py) {
  // Request buffer info
  py::buffer_info V_info = V_py.request();
  py::buffer_info E_info = E_py.request();

  // Do some validation checks on the buffers
  if (V_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (E_info.format != py::format_descriptor<int>::format()) {
    throw std::runtime_error("Incompatible data format: expected a int array");
  }
  if (V_info.ndim != 2) {
    throw std::runtime_error("Incompatible dimension: expected a 2d array");
  }
  if (E_info.ndim != 2) {
    throw std::runtime_error("Incompatible dimension: expected a 2d array");
  }
  if (V_info.shape[1] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 columns");
  }
  if (E_info.shape[1] != 2) {
    throw std::runtime_error("Incompatible dimension: expected 2 columns");
  }

  PyBVHTree tree;

  // Copy the geometry into the row-major buffers we use
  int num_vertices = V_info.shape[0];
  auto Vdata = static_cast<const double*>(V_info.ptr);
  py::ssize_t Vstrides[2] = {V_info.strides[0] / (py::ssize_t)sizeof(double),
                             V_info.strides[1] / (py::ssize_t)sizeof(double)};
  tree.fillV(Vdata, num_vertices, Vstrides);

  int num_edges = E_info.shape[0];
  auto Edata = static_cast<const int*>(E_info.ptr);
  py::ssize_t Estrides[2] = {E_info.strides[0] / (py::ssize_t)sizeof(int),
                             E_info.strides[1] / (py::ssize_t)sizeof(int)};
  tree.fillE(Edata, num_edges, Estrides);

  tree.indices.reserve(num_edges);
  for (int i = 0; i < num_edges; i++) {
    tree.indices.push_back(i);
  }
  pybvh::buildBVHEdges(tree.V.data(), tree.E.data(), tree.indices, 0, num_edges,
                       tree.nodes);

  tree.finishBuild();
  return tree;
}

py::tuple minDist(py::buffer q_py, const PyBVHTree& tree) {
  py::buffer_info q_info = q_py.request();
  if (q_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (q_info.ndim != 1) {
    throw std::runtime_error("Incompatible dimension: expected a 1d array");
  }
  if (q_info.shape[0] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 entries");
  }

  auto qptr = static_cast<pybvh::Vector*>(q_info.ptr);
  pybvh::QueryResult result = pybvh::minDist(*qptr, tree.tree_ptrs);
  return py::make_tuple(result.dist, result.idx);
}

PYBIND11_MODULE(pybvh, m) {
  m.doc() = "A plugin to expose a C++ BVH implementation in Python";

  py::class_<PyBVHTree>(m, "BVHTree");

  m.def("build_bvh_points", &buildBVHPoints,
        "Builds a BVH over the given point set", py::keep_alive<0, 1>());
  m.def("build_bvh_triangles", &buildBVHTriangles,
        "Builds a BVH over the given triangle mesh", py::keep_alive<0, 1>(),
        py::keep_alive<0, 2>());
  m.def("build_bvh_edges", &buildBVHEdges,
        "Builds a BVH over the given edge mesh", py::keep_alive<0, 1>(),
        py::keep_alive<0, 2>());
  m.def("min_dist", &minDist, "Computes the minimum distance to the point set");
}
