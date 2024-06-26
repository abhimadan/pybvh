#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

#include "bvh.h"
#include "bvh_query.h"

namespace py = pybind11;

namespace pybvh_bindings {

// This structure owns all the cpu memory it needs, and the binding should make
// sure that the lifetime of this struct is longer than the geometry arrays used
// to build it.
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

} // namespace pybvh_bindings

using namespace pybvh_bindings;

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

std::vector<pybvh::DistResult> minDist(py::buffer q_py,
                                        const PyBVHTree& tree) {
  py::buffer_info q_info = q_py.request();
  if (q_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (q_info.ndim != 2) {
    throw std::runtime_error("Incompatible dimension: expected a 2d array");
  }
  if (q_info.shape[1] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 columns");
  }

  int num_queries = q_info.shape[0];
  py::ssize_t qstrides[2] = {q_info.strides[0] / (py::ssize_t)sizeof(double),
                             q_info.strides[1] / (py::ssize_t)sizeof(double)};
  auto qdata = static_cast<const double*>(q_info.ptr);

  // TODO: this ends up as a list, can convert to numpy arrays on this side
  // maybe (or I guess it's easier on the python side since with this structure
  // we'd need a copy regardless)
  // I guess we want this as numpy arrays instead?
  // This is still a bit slower than the libigl version (probably because we're
  // converting column-major to row-major and not using Eigen), but it has
  // reasonable performance
  int num_threads = 8;
  int batch_size = num_queries / num_threads + 1;
  std::vector<pybvh::DistResult> results(num_queries);
  auto eval_query_batch = [&](int thread_idx) {
    int start = batch_size * thread_idx;
    int end = std::min(batch_size * (thread_idx + 1), num_queries);
    for (int i = start; i < end; i++) {
      double x = *(qdata + i * qstrides[0] + 0 * qstrides[1]);
      double y = *(qdata + i * qstrides[0] + 1 * qstrides[1]);
      double z = *(qdata + i * qstrides[0] + 2 * qstrides[1]);
      pybvh::Vector q(x, y, z);
      results[i] = pybvh::minDist(q, tree.tree_ptrs);
    }
  };

  std::vector<std::thread> query_threads;
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    query_threads.emplace_back(eval_query_batch, thread_idx);
  }
  for (auto& t : query_threads) {
    t.join();
  }

  return results;
}

std::vector<std::vector<pybvh::DistResult>> knn(py::buffer q_py, int k,
                                                 const PyBVHTree& tree) {
  py::buffer_info q_info = q_py.request();
  if (q_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (q_info.ndim != 2) {
    throw std::runtime_error("Incompatible dimension: expected a 2d array");
  }
  if (q_info.shape[1] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 columns");
  }

  int num_queries = q_info.shape[0];
  py::ssize_t qstrides[2] = {q_info.strides[0] / (py::ssize_t)sizeof(double),
                             q_info.strides[1] / (py::ssize_t)sizeof(double)};
  auto qdata = static_cast<const double*>(q_info.ptr);

  // TODO: this ends up as a list, can convert to numpy arrays on this side
  // maybe (or I guess it's easier on the python side since with this structure
  // we'd need a copy regardless)
  // I guess we want this as numpy arrays instead?
  // This is still a bit slower than the libigl version (probably because we're
  // converting column-major to row-major and not using Eigen), but it has
  // reasonable performance
  int num_threads = 8;
  int batch_size = num_queries / num_threads + 1;
  std::vector<std::vector<pybvh::DistResult>> results(num_queries);
  for (std::vector<pybvh::DistResult>& r : results) {
    r.resize(k);
  }
  auto eval_query_batch = [&](int thread_idx) {
    int start = batch_size * thread_idx;
    int end = std::min(batch_size * (thread_idx + 1), num_queries);
    for (int i = start; i < end; i++) {
      double x = *(qdata + i * qstrides[0] + 0 * qstrides[1]);
      double y = *(qdata + i * qstrides[0] + 1 * qstrides[1]);
      double z = *(qdata + i * qstrides[0] + 2 * qstrides[1]);
      pybvh::Vector q(x, y, z);
      pybvh::KNNResult result = pybvh::knn(q, k, tree.tree_ptrs);
      for (int j = 0; j < k && !result.empty(); j++) {
        pybvh::DistResult r = result.top();
        results[i][j] = r;
        result.pop();
      }
    }
  };

  std::vector<std::thread> query_threads;
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    query_threads.emplace_back(eval_query_batch, thread_idx);
  }
  for (auto& t : query_threads) {
    t.join();
  }

  return results;
}

std::vector<pybvh::RadiusResult> radiusSearch(
    py::buffer q_py, double radius, const PyBVHTree& tree) {
  py::buffer_info q_info = q_py.request();
  if (q_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (q_info.ndim != 2) {
    throw std::runtime_error("Incompatible dimension: expected a 2d array");
  }
  if (q_info.shape[1] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 columns");
  }

  int num_queries = q_info.shape[0];
  py::ssize_t qstrides[2] = {q_info.strides[0] / (py::ssize_t)sizeof(double),
                             q_info.strides[1] / (py::ssize_t)sizeof(double)};
  auto qdata = static_cast<const double*>(q_info.ptr);

  // TODO: this ends up as a list, can convert to numpy arrays on this side
  // maybe (or I guess it's easier on the python side since with this structure
  // we'd need a copy regardless)
  // I guess we want this as numpy arrays instead?
  // This is still a bit slower than the libigl version (probably because we're
  // converting column-major to row-major and not using Eigen), but it has
  // reasonable performance
  int num_threads = 8;
  int batch_size = num_queries / num_threads + 1;
  std::vector<pybvh::RadiusResult> results(num_queries);
  auto eval_query_batch = [&](int thread_idx) {
    int start = batch_size * thread_idx;
    int end = std::min(batch_size * (thread_idx + 1), num_queries);
    for (int i = start; i < end; i++) {
      double x = *(qdata + i * qstrides[0] + 0 * qstrides[1]);
      double y = *(qdata + i * qstrides[0] + 1 * qstrides[1]);
      double z = *(qdata + i * qstrides[0] + 2 * qstrides[1]);
      pybvh::Vector q(x, y, z);
      results[i] = pybvh::radiusSearch(q, radius, tree.tree_ptrs);
    }
  };

  std::vector<std::thread> query_threads;
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    query_threads.emplace_back(eval_query_batch, thread_idx);
  }
  for (auto& t : query_threads) {
    t.join();
  }

  return results;
}

std::vector<pybvh::HitResult> closestRayIntersection(py::buffer o_py,
                                                     py::buffer d_py,
                                                     const PyBVHTree& tree) {
  py::buffer_info o_info = o_py.request();
  if (o_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (o_info.ndim != 1) {
    throw std::runtime_error("Incompatible dimension: expected a 1d array");
  }
  if (o_info.shape[0] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 entries");
  }

  py::buffer_info d_info = d_py.request();
  if (d_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (d_info.ndim != 2) {
    throw std::runtime_error("Incompatible dimension: expected a 2d array");
  }
  if (d_info.shape[1] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 entries");
  }

  auto odata = static_cast<const double*>(o_info.ptr);
  py::ssize_t ostride = o_info.strides[0] / (py::ssize_t)sizeof(double);
  pybvh::Vector origin(*odata, *(odata + ostride), *(odata + 2 * ostride));

  int num_queries = d_info.shape[0];
  py::ssize_t dstrides[2] = {d_info.strides[0] / (py::ssize_t)sizeof(double),
                             d_info.strides[1] / (py::ssize_t)sizeof(double)};
  auto ddata = static_cast<const double*>(d_info.ptr);

  int num_threads = 8;
  int batch_size = num_queries / num_threads + 1;
  std::vector<pybvh::HitResult> results(num_queries);
  auto eval_query_batch = [&](int thread_idx) {
    int start = batch_size * thread_idx;
    int end = std::min(batch_size * (thread_idx + 1), num_queries);
    for (int i = start; i < end; i++) {
      double x = *(ddata + i * dstrides[0] + 0 * dstrides[1]);
      double y = *(ddata + i * dstrides[0] + 1 * dstrides[1]);
      double z = *(ddata + i * dstrides[0] + 2 * dstrides[1]);
      pybvh::Vector direction(x, y, z);
      results[i] =
          pybvh::closestRayIntersection(origin, direction, tree.tree_ptrs);
    }
  };

  std::vector<std::thread> query_threads;
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    query_threads.emplace_back(eval_query_batch, thread_idx);
  }
  for (auto& t : query_threads) {
    t.join();
  }

  return results;
}

std::vector<std::vector<pybvh::HitResult>> allRayIntersections(
    py::buffer o_py, py::buffer d_py, const PyBVHTree& tree) {
  py::buffer_info o_info = o_py.request();
  if (o_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (o_info.ndim != 1) {
    throw std::runtime_error("Incompatible dimension: expected a 1d array");
  }
  if (o_info.shape[0] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 entries");
  }

  py::buffer_info d_info = d_py.request();
  if (d_info.format != py::format_descriptor<double>::format()) {
    throw std::runtime_error(
        "Incompatible data format: expected a double array");
  }
  if (d_info.ndim != 2) {
    throw std::runtime_error("Incompatible dimension: expected a 2d array");
  }
  if (d_info.shape[1] != 3) {
    throw std::runtime_error("Incompatible dimension: expected 3 entries");
  }

  auto odata = static_cast<const double*>(o_info.ptr);
  py::ssize_t ostride = o_info.strides[0] / (py::ssize_t)sizeof(double);
  pybvh::Vector origin(*odata, *(odata + ostride), *(odata + 2 * ostride));

  int num_queries = d_info.shape[0];
  py::ssize_t dstrides[2] = {d_info.strides[0] / (py::ssize_t)sizeof(double),
                             d_info.strides[1] / (py::ssize_t)sizeof(double)};
  auto ddata = static_cast<const double*>(d_info.ptr);

  int num_threads = 8;
  int batch_size = num_queries / num_threads + 1;
  std::vector<std::vector<pybvh::HitResult>> direct_results(num_queries);
  std::vector<size_t> thread_max_hits(num_threads, 0);
  auto eval_query_batch = [&](int thread_idx) {
    int start = batch_size * thread_idx;
    int end = std::min(batch_size * (thread_idx + 1), num_queries);
    for (int i = start; i < end; i++) {
      double x = *(ddata + i * dstrides[0] + 0 * dstrides[1]);
      double y = *(ddata + i * dstrides[0] + 1 * dstrides[1]);
      double z = *(ddata + i * dstrides[0] + 2 * dstrides[1]);
      pybvh::Vector direction(x, y, z);
      direct_results[i] =
          pybvh::allRayIntersections(origin, direction, tree.tree_ptrs);
      thread_max_hits[thread_idx] =
          std::max(thread_max_hits[thread_idx], direct_results[i].size());
    }
  };

  std::vector<std::thread> query_threads;
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    query_threads.emplace_back(eval_query_batch, thread_idx);
  }
  for (auto& t : query_threads) {
    t.join();
  }

  // Now transpose the results so we essentially get a stack of "images" per hit
  size_t max_hits =
      *std::max_element(thread_max_hits.begin(), thread_max_hits.end());
  std::vector<std::vector<pybvh::HitResult>> results(max_hits);
  for (auto& result_image : results) {
    result_image.resize(num_queries);
  }
  for (int i = 0; i < num_queries; i++) {
    const std::vector<pybvh::HitResult>& pixel_result = direct_results[i];
    int num_hits = pixel_result.size();
    for (int j = 0; j < num_hits; j++) {
      results[j][i] = pixel_result[j];
    }
  }

  return results;
}

PYBIND11_MODULE(pybvh, m) {
  m.doc() = "A plugin to expose a C++ BVH implementation in Python";

  py::class_<PyBVHTree>(m, "BVHTree");

  py::class_<pybvh::Vector>(m, "Vector", py::buffer_protocol())
      .def_buffer([](pybvh::Vector& v) -> py::buffer_info {
        return py::buffer_info(v.vals, sizeof(double),
                               py::format_descriptor<double>::format(), 1, {3},
                               {sizeof(double)});
      });

  py::class_<pybvh::DistResult>(m, "DistResult")
    .def_readonly("point", &pybvh::DistResult::point)
    .def_readonly("dist", &pybvh::DistResult::dist)
    .def_readonly("u", &pybvh::DistResult::u)
    .def_readonly("v", &pybvh::DistResult::v)
    .def_readonly("idx", &pybvh::DistResult::idx);

  py::class_<pybvh::HitResult>(m, "HitResult")
    .def_readonly("point", &pybvh::HitResult::point)
    .def_readonly("t", &pybvh::HitResult::t)
    .def_readonly("u", &pybvh::HitResult::u)
    .def_readonly("v", &pybvh::HitResult::v)
    .def_readonly("idx", &pybvh::HitResult::idx)
    .def("is_hit", &pybvh::HitResult::isHit);

  m.def("build_bvh_points", &buildBVHPoints,
        "Builds a BVH over the given point set", py::keep_alive<0, 1>());
  m.def("build_bvh_triangles", &buildBVHTriangles,
        "Builds a BVH over the given triangle mesh", py::keep_alive<0, 1>(),
        py::keep_alive<0, 2>());
  m.def("build_bvh_edges", &buildBVHEdges,
        "Builds a BVH over the given edge mesh", py::keep_alive<0, 1>(),
        py::keep_alive<0, 2>());

  // I think this uses the move policy by default, which should be fine, but I
  // don't fully understand whether or not we're skipping the additional copy to
  // a python list. I guess making the vector opaque would solve this though
  m.def("min_dist", &minDist,
        "Computes the minimum distance of each query point to the mesh");

  m.def("knn", &knn,
        "Computes the k nearest neighbours of each query point on the mesh");

  m.def("radius_search", &radiusSearch,
        "Computes the points on the mesh within the specified (squared) radius "
        "of the query points");

  m.def(
      "closest_ray_intersection", &closestRayIntersection,
      "Computes the ray intersections of each given ray with a common origin");

  m.def(
      "all_ray_intersections", &allRayIntersections,
      "Computes all ray intersections of each given ray with a common origin");
}
