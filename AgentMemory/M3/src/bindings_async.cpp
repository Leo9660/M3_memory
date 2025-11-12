#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <cstring>

#include "m3_async.h"

namespace py = pybind11;
using namespace m3;

PYBIND11_MODULE(_m3_async, m) {
    m.doc() = "M3 Async Engine (multi-IVF + async writers)";

    // ----- Metric enum -----
    py::enum_<Metric>(m, "Metric")
        .value("L2",     Metric::L2)
        .value("IP",     Metric::IP)
        .value("COSINE", Metric::COSINE)
        .export_values();

    // ----- AsyncEngine -----
    py::class_<AsyncEngine>(m, "AsyncEngine")
        .def(py::init<>())

        // start/stop
        .def("start",
             [](AsyncEngine& e, int writers, int maint) {
                 {
                     py::gil_scoped_release _g;
                     e.start(writers, maint);
                 }
             },
             py::arg("writer_threads") = 2,
             py::arg("maintenance_threads") = 1)

        .def("stop",
             [](AsyncEngine& e) {
                 {
                     py::gil_scoped_release _g;
                     e.stop();
                 }
             })

        // create one IVF index under index_id
        .def("create_ivf",
             [](AsyncEngine& e,
                int index_id,
                int dim,
                Metric metric,
                bool normalized,
                py::array_t<float, py::array::c_style> centroids) {
                 auto buf = centroids.request();
                 if (buf.ndim != 2) {
                     throw std::runtime_error("centroids must be 2D [nlist, dim]");
                 }
                 if (buf.shape[1] != dim) {
                     throw std::runtime_error(
                         "centroids dim mismatch: expected " + std::to_string(dim) +
                         ", got " + std::to_string(buf.shape[1]));
                 }
                 std::vector<float> c;
                 c.assign((float*)buf.ptr, (float*)buf.ptr + buf.size);
                 {
                     py::gil_scoped_release _g;
                     e.create_ivf(index_id, dim, metric, normalized, c);
                 }
             },
             py::arg("index_id"),
             py::arg("dim"),
             py::arg("metric"),
             py::arg("normalized"),
             py::arg("centroids"))

        // ---- policy setters ----
        .def("set_queue_policy",
             [](AsyncEngine& e,
                size_t capacity,
                size_t pop_batch_max,
                bool block_on_full) {
                 AsyncQueuePolicy p;
                 p.capacity = capacity;
                 p.pop_batch_max = pop_batch_max;
                 p.block_on_full = block_on_full;
                 e.set_queue_policy(p);
             },
             py::arg("capacity") = 4096,
             py::arg("pop_batch_max") = 64,
             py::arg("block_on_full") = true)

        .def("set_search_policy",
             [](AsyncEngine& e,
                int default_nprobe,
                bool parallel_queries) {
                 AsyncSearchPolicy p;
                 p.default_nprobe = default_nprobe;
                 p.parallel_queries = parallel_queries;
                 e.set_search_policy(p);
             },
             py::arg("default_nprobe") = 8,
             py::arg("parallel_queries") = false)

        .def("set_maintenance_policy",
             [](AsyncEngine& e,
                double period_sec,
                size_t split_threshold,
                double compact_ratio) {
                 AsyncMaintenancePolicy p;
                 p.period_sec = period_sec;
                 p.split_threshold = split_threshold;
                 p.compact_ratio = compact_ratio;
                 e.set_maintenance_policy(p);
             },
             py::arg("period_sec") = 1.0,
             py::arg("split_threshold") = 200000,
             py::arg("compact_ratio") = 0.7)

        // ---- enqueue_insert(index_id, cluster_id, ids, vecs) ----
        .def("enqueue_insert",
             [](AsyncEngine& e,
                int index_id,
                int cluster_id,
                py::array_t<int64_t, py::array::c_style> ids,
                py::array_t<float,   py::array::c_style> vecs) {
                 auto idb = ids.request();
                 auto vb  = vecs.request();
                 if (idb.ndim != 1) {
                     throw std::runtime_error("ids must be 1D [N]");
                 }
                 if (vb.ndim != 2) {
                     throw std::runtime_error("vectors must be 2D [N, D]");
                 }
                 if (idb.shape[0] != vb.shape[0]) {
                     throw std::runtime_error("ids.size != vectors.N");
                 }
                 int expected_dim = e.dim_of(index_id);
                 if (expected_dim <= 0) {
                     throw std::runtime_error("enqueue_insert: unknown index_id (call create_ivf first)");
                 }
                 if (vb.shape[1] != expected_dim) {
                     throw std::runtime_error(
                         "enqueue_insert: vectors dim mismatch, expected " +
                         std::to_string(expected_dim) + ", got " +
                         std::to_string(vb.shape[1]));
                 }
                 InsertOp op;
                 op.index_id   = index_id;
                 op.cluster_id = cluster_id;
                 op.ids.assign((int64_t*)idb.ptr,
                               (int64_t*)idb.ptr + idb.shape[0]);
                 op.vecs.assign((float*)vb.ptr,
                                (float*)vb.ptr + vb.shape[0] * vb.shape[1]);
                 bool ok;
                 {
                     py::gil_scoped_release _g;
                     ok = e.enqueue_insert(std::move(op));
                 }
                 return ok;
             },
             py::arg("index_id"),
             py::arg("cluster_id"),
             py::arg("ids"),
             py::arg("vectors"))

        // ---- enqueue_insert_auto(index_id, ids, vecs) ----
        .def("enqueue_insert_auto",
             [](AsyncEngine& e,
                int index_id,
                py::array_t<int64_t, py::array::c_style> ids,
                py::array_t<float,   py::array::c_style> vecs) {
                 auto idb = ids.request();
                 auto vb  = vecs.request();
                 if (idb.ndim != 1) {
                     throw std::runtime_error("ids must be 1D [N]");
                 }
                 if (vb.ndim != 2) {
                     throw std::runtime_error("vectors must be 2D [N, D]");
                 }
                 if (idb.shape[0] != vb.shape[0]) {
                     throw std::runtime_error("ids.size != vectors.N");
                 }
                 int expected_dim = e.dim_of(index_id);
                 if (expected_dim <= 0) {
                     throw std::runtime_error("enqueue_insert_auto: unknown index_id (call create_ivf first)");
                 }
                 if (vb.shape[1] != expected_dim) {
                     throw std::runtime_error(
                         "enqueue_insert_auto: vectors dim mismatch, expected " +
                         std::to_string(expected_dim) + ", got " +
                         std::to_string(vb.shape[1]));
                 }

                 std::vector<DocId> ids_vec(idb.shape[0]);
                 if (idb.shape[0] > 0) {
                     std::memcpy(ids_vec.data(),
                                 idb.ptr,
                                 ids_vec.size() * sizeof(DocId));
                 }
                 std::vector<float> vecs_vec(vb.shape[0] * vb.shape[1]);
                 if (!vecs_vec.empty()) {
                     std::memcpy(vecs_vec.data(),
                                 vb.ptr,
                                 vecs_vec.size() * sizeof(float));
                 }
                 bool ok;
                 {
                     py::gil_scoped_release _g;
                     ok = e.enqueue_insert_auto(index_id, ids_vec, vecs_vec);
                 }
                 return ok;
             },
             py::arg("index_id"),
             py::arg("ids"),
             py::arg("vectors"))

        // ---- enqueue_update(index_id, cluster_id, ...) ----
        .def("enqueue_update",
             [](AsyncEngine& e,
                int index_id,
                int cluster_id,
                py::array_t<int64_t, py::array::c_style> ids,
                py::array_t<float,   py::array::c_style> vecs,
                bool insert_if_absent) {
                 auto idb = ids.request();
                 auto vb  = vecs.request();
                 if (idb.ndim != 1) {
                     throw std::runtime_error("ids must be 1D [N]");
                 }
                 if (vb.ndim != 2) {
                     throw std::runtime_error("vectors must be 2D [N, D]");
                 }
                 if (idb.shape[0] != vb.shape[0]) {
                     throw std::runtime_error("ids.size != vectors.N");
                 }
                 int expected_dim = e.dim_of(index_id);
                 if (expected_dim <= 0) {
                     throw std::runtime_error("enqueue_update: unknown index_id (call create_ivf first)");
                 }
                 if (vb.shape[1] != expected_dim) {
                     throw std::runtime_error(
                         "enqueue_update: vectors dim mismatch, expected " +
                         std::to_string(expected_dim) + ", got " +
                         std::to_string(vb.shape[1]));
                 }
                 UpdateOp op;
                 op.index_id = index_id;
                 op.cluster_id = cluster_id;
                 op.insert_if_absent = insert_if_absent;
                 op.ids.assign((int64_t*)idb.ptr,
                               (int64_t*)idb.ptr + idb.shape[0]);
                 op.vecs.assign((float*)vb.ptr,
                                (float*)vb.ptr + vb.shape[0] * vb.shape[1]);
                 bool ok;
                 {
                     py::gil_scoped_release _g;
                     ok = e.enqueue_update(std::move(op));
                 }
                 return ok;
             },
             py::arg("index_id"),
             py::arg("cluster_id"),
             py::arg("ids"),
             py::arg("vectors"),
             py::arg("insert_if_absent") = false)

        // ---- enqueue_delete(index_id, cluster_id, ids) ----
        .def("enqueue_delete",
             [](AsyncEngine& e,
                int index_id,
                int cluster_id,
                py::array_t<int64_t, py::array::c_style> ids) {
                 auto idb = ids.request();
                 if (idb.ndim != 1) {
                     throw std::runtime_error("ids must be 1D [N]");
                 }
                 DeleteOp op;
                 op.index_id   = index_id;
                 op.cluster_id = cluster_id;
                 op.ids.assign((int64_t*)idb.ptr,
                               (int64_t*)idb.ptr + idb.shape[0]);
                 bool ok;
                 {
                     py::gil_scoped_release _g;
                     ok = e.enqueue_delete(std::move(op));
                 }
                 return ok;
             },
             py::arg("index_id"),
             py::arg("cluster_id"),
             py::arg("ids"))

        // ---- load_cluster(index_id, cluster_id, ids, vecs) ----
        .def("load_cluster",
             [](AsyncEngine& e,
                int index_id,
                int cluster_id,
                py::array_t<int64_t, py::array::c_style> ids,
                py::array_t<float,   py::array::c_style> vecs) {
                 auto idb = ids.request();
                 auto vb  = vecs.request();
                 if (idb.ndim != 1) {
                     throw std::runtime_error("load_cluster: ids must be 1D [N]");
                 }
                 if (vb.ndim != 2) {
                     throw std::runtime_error("load_cluster: vectors must be 2D [N, D]");
                 }
                 if (idb.shape[0] != vb.shape[0]) {
                     throw std::runtime_error("load_cluster: ids.size != vectors.N");
                 }
                 int expected_dim = e.dim_of(index_id);
                 if (expected_dim <= 0) {
                     throw std::runtime_error("load_cluster: unknown index_id (call create_ivf first)");
                 }
                 if (vb.shape[1] != expected_dim) {
                     throw std::runtime_error(
                         "load_cluster: vectors dim mismatch, expected " +
                         std::to_string(expected_dim) + ", got " +
                         std::to_string(vb.shape[1]));
                 }

                 std::vector<DocId> ids_vec(idb.shape[0]);
                 if (idb.shape[0] > 0) {
                     std::memcpy(ids_vec.data(),
                                 idb.ptr,
                                 ids_vec.size() * sizeof(DocId));
                 }
                 std::vector<float> vecs_vec(vb.shape[0] * vb.shape[1]);
                 if (!vecs_vec.empty()) {
                     std::memcpy(vecs_vec.data(),
                                 vb.ptr,
                                 vecs_vec.size() * sizeof(float));
                 }
                 {
                     py::gil_scoped_release _g;
                     e.load_cluster(index_id, cluster_id, ids_vec, vecs_vec);
                 }
             },
             py::arg("index_id"),
             py::arg("cluster_id"),
             py::arg("ids"),
             py::arg("vectors"))

        // flush
        .def("flush",
             [](AsyncEngine& e) {
                 {
                     py::gil_scoped_release _g;
                     e.flush();
                 }
                 //  printf("Flush complete.\n");
             })

        // ---- search(index_id, q, k) ----
        .def("search",
             [](AsyncEngine& e,
                int index_id,
                py::array_t<float, py::array::c_style> queries,
                int k) {
                 auto buf = queries.request();
                 if (buf.ndim != 2) {
                     throw std::runtime_error("queries must be 2D [Q, D]");
                 }
                 int expected_dim = e.dim_of(index_id);
                 if (expected_dim <= 0) {
                     throw std::runtime_error("search: unknown index_id (call create_ivf first)");
                 }
                 if (buf.shape[1] != expected_dim) {
                     throw std::runtime_error(
                         "search: query dim mismatch, expected " +
                         std::to_string(expected_dim) + ", got " +
                         std::to_string(buf.shape[1]));
                 }
                 std::vector<std::vector<DocId>> out_ids;
                 std::vector<std::vector<float>> out_scores;
                 out_ids.reserve(buf.shape[0]);
                 out_scores.reserve(buf.shape[0]);
                 {
                    py::gil_scoped_release _g;
                    e.search(index_id,
                          (const float*)buf.ptr,
                          (size_t)buf.shape[0],
                          k,
                          /*nprobe=*/-1,
                          out_ids,
                          out_scores);
                }
                //  printf("Query %d: final results:", index_id);
                //  for (size_t i = 0; i < out_ids.size(); ++i) {
                //         for (size_t j = 0; j < out_ids[i].size(); ++j) {
                //             printf(" (id=%ld, score=%.4f)", out_ids[i][j], out_scores[i][j]);
                //         }
                //  }
                //  printf("Finished\n");
                 return py::make_tuple(out_ids, out_scores);
             },
             py::arg("index_id"),
             py::arg("queries"),
             py::arg("k"))

        // ---- search(index_id, q, k, nprobe) ----
        .def("search",
             [](AsyncEngine& e,
                int index_id,
                py::array_t<float, py::array::c_style> queries,
                int k,
                int nprobe) {
                 auto buf = queries.request();
                 if (buf.ndim != 2) {
                     throw std::runtime_error("queries must be 2D [Q, D]");
                 }
                 int expected_dim = e.dim_of(index_id);
                 if (expected_dim <= 0) {
                     throw std::runtime_error("search: unknown index_id (call create_ivf first)");
                 }
                 if (buf.shape[1] != expected_dim) {
                     throw std::runtime_error(
                         "search: query dim mismatch, expected " +
                         std::to_string(expected_dim) + ", got " +
                         std::to_string(buf.shape[1]));
                 }
                 std::vector<std::vector<DocId>> out_ids;
                 std::vector<std::vector<float>> out_scores;
                 out_ids.reserve(buf.shape[0]);
                 out_scores.reserve(buf.shape[0]);
                 {
                    py::gil_scoped_release _g;
                    e.search(index_id,
                            (const float*)buf.ptr,
                            (size_t)buf.shape[0],
                            k,
                            nprobe,
                            out_ids,
                            out_scores);
                 }
                 return py::make_tuple(out_ids, out_scores);
             },
             py::arg("index_id"),
             py::arg("queries"),
             py::arg("k"),
             py::arg("nprobe"))

        // ---- info ----
        .def("dim_of",
             [](AsyncEngine& e, int index_id) {
                 return e.dim_of(index_id);
             })
        .def("metric_of",
             [](AsyncEngine& e, int index_id) {
                 return e.metric_of(index_id);
             })
        .def("normalized_of",
             [](AsyncEngine& e, int index_id) {
                 return e.normalized_of(index_id);
             })
        .def("nlist_of",
             [](AsyncEngine& e, int index_id) {
                 return e.nlist_of(index_id);
             });
}
