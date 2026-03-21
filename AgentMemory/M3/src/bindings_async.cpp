#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <cstring>

#include "m3_async.h"
#include "m3_multi_level.h"
#include "m3_fsm.h"

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

    // ----- MultiLevelConfig -----
    py::class_<MultiLevelConfig>(m, "MultiLevelConfig")
        .def(py::init<>())
        .def_readwrite("l0_nlist", &MultiLevelConfig::l0_nlist)
        .def_readwrite("l1_nlist", &MultiLevelConfig::l1_nlist)
        .def_readwrite("l2_nlist", &MultiLevelConfig::l2_nlist)
        .def_readwrite("l0_new_cluster_threshold", &MultiLevelConfig::l0_new_cluster_threshold)
        .def_readwrite("search_threshold", &MultiLevelConfig::search_threshold)
        .def_readwrite("l0_merge_threshold", &MultiLevelConfig::l0_merge_threshold)
        .def_readwrite("l0_max_nlist", &MultiLevelConfig::l0_max_nlist);

    // ----- CacheConfig -----
    py::class_<CacheConfig>(m, "CacheConfig")
        .def(py::init<>())
        .def_readwrite("l0_max_clusters", &CacheConfig::l0_max_clusters)
        .def_readwrite("l0_max_vectors_per_cluster", &CacheConfig::l0_max_vectors_per_cluster)
        .def_readwrite("l1_max_clusters", &CacheConfig::l1_max_clusters)
        .def_readwrite("l1_max_vectors_per_cluster", &CacheConfig::l1_max_vectors_per_cluster)
        .def_readwrite("l0_eviction_ratio", &CacheConfig::l0_eviction_ratio)
        .def_readwrite("l1_eviction_ratio", &CacheConfig::l1_eviction_ratio)
        .def_readwrite("cold_time_ns", &CacheConfig::cold_time_ns)
        .def_readwrite("l0_neighborhood_k", &CacheConfig::l0_neighborhood_k)
        .def_readwrite("l1_neighborhood_k", &CacheConfig::l1_neighborhood_k)
        .def_readwrite("max_promote_per_query", &CacheConfig::max_promote_per_query)
        .def_readwrite("prefetch_queue_capacity", &CacheConfig::prefetch_queue_capacity);

    // ----- MultiLevelIndex -----
    py::class_<MultiLevelIndex>(m, "MultiLevelIndex")
        .def(py::init([](int dim,
                         Metric metric,
                         bool normalized,
                         const MultiLevelConfig& cfg) {
                 return new MultiLevelIndex(dim, metric, normalized, cfg);
             }),
             py::arg("dim"),
             py::arg("metric"),
             py::arg("normalized") = true,
             py::arg("config") = MultiLevelConfig())
        .def("set_l0_centroids",
             [](MultiLevelIndex& idx, py::array_t<float, py::array::c_style> centroids) {
                 auto buf = centroids.request();
                 if (buf.ndim != 2) {
                     throw std::runtime_error("centroids must be 2D [nlist, dim]");
                 }
                 if (buf.shape[1] != idx.dim()) {
                     throw std::runtime_error("centroids dim mismatch");
                 }
                 std::vector<float> c;
                 c.assign((float*)buf.ptr, (float*)buf.ptr + buf.size);
                 py::gil_scoped_release _g;
                 idx.set_l0_centroids(c);
             })
        .def("set_l1_centroids",
             [](MultiLevelIndex& idx, py::array_t<float, py::array::c_style> centroids) {
                 auto buf = centroids.request();
                 if (buf.ndim != 2) throw std::runtime_error("centroids must be 2D [nlist, dim]");
                 if (buf.shape[1] != idx.dim()) throw std::runtime_error("centroids dim mismatch");
                 std::vector<float> c;
                 c.assign((float*)buf.ptr, (float*)buf.ptr + buf.size);
                 py::gil_scoped_release _g;
                 idx.set_l1_centroids(c);
             })
        .def("set_l2_centroids",
             [](MultiLevelIndex& idx, py::array_t<float, py::array::c_style> centroids) {
                 auto buf = centroids.request();
                 if (buf.ndim != 2) throw std::runtime_error("centroids must be 2D [nlist, dim]");
                 if (buf.shape[1] != idx.dim()) throw std::runtime_error("centroids dim mismatch");
                 std::vector<float> c;
                 c.assign((float*)buf.ptr, (float*)buf.ptr + buf.size);
                 py::gil_scoped_release _g;
                 idx.set_l2_centroids(c);
             })
        .def("set_cache_config",
             [](MultiLevelIndex& idx, const CacheConfig& cfg) {
                 idx.set_cache_config(cfg);
             })
        .def("insert",
             [](MultiLevelIndex& idx,
                py::array_t<int64_t, py::array::c_style> ids,
                py::array_t<float,   py::array::c_style> vecs) {
                 auto idb = ids.request();
                 auto vb  = vecs.request();
                 if (idb.ndim != 1) throw std::runtime_error("ids must be 1D [N]");
                 if (vb.ndim != 2)  throw std::runtime_error("vectors must be 2D [N, D]");
                 if (idb.shape[0] != vb.shape[0]) throw std::runtime_error("ids.size != vectors.N");
                 if (vb.shape[1] != idx.dim()) throw std::runtime_error("vectors dim mismatch");
                 py::gil_scoped_release _g;
                 idx.insert((const DocId*)idb.ptr, (const float*)vb.ptr, (size_t)idb.shape[0]);
             })
        .def("update",
             [](MultiLevelIndex& idx,
                py::array_t<int64_t, py::array::c_style> ids,
                py::array_t<float,   py::array::c_style> vecs,
                bool insert_if_absent) {
                 auto idb = ids.request();
                 auto vb  = vecs.request();
                 if (idb.ndim != 1) throw std::runtime_error("ids must be 1D [N]");
                 if (vb.ndim != 2)  throw std::runtime_error("vectors must be 2D [N, D]");
                 if (idb.shape[0] != vb.shape[0]) throw std::runtime_error("ids.size != vectors.N");
                 if (vb.shape[1] != idx.dim()) throw std::runtime_error("vectors dim mismatch");
                 py::gil_scoped_release _g;
                 idx.update((const DocId*)idb.ptr, (const float*)vb.ptr, (size_t)idb.shape[0],
                            insert_if_absent);
             },
             py::arg("ids"),
             py::arg("vectors"),
             py::arg("insert_if_absent") = false)
        .def("erase",
             [](MultiLevelIndex& idx,
                py::array_t<int64_t, py::array::c_style> ids) {
                 auto idb = ids.request();
                 if (idb.ndim != 1) throw std::runtime_error("ids must be 1D [N]");
                 py::gil_scoped_release _g;
                 idx.erase((const DocId*)idb.ptr, (size_t)idb.shape[0]);
             })
        .def("search",
             [](const MultiLevelIndex& idx,
                py::array_t<float, py::array::c_style> queries,
                int k,
                int nprobe) {
                 auto buf = queries.request();
                 if (buf.ndim != 2) throw std::runtime_error("queries must be 2D [Q, D]");
                 if (buf.shape[1] != idx.dim()) throw std::runtime_error("query dim mismatch");
                 std::vector<std::vector<DocId>> out_ids;
                 std::vector<std::vector<float>> out_scores;
                 {
                     py::gil_scoped_release _g;
                     idx.search((const float*)buf.ptr,
                                (size_t)buf.shape[0],
                                k,
                                nprobe,
                                out_ids,
                                out_scores,
                                nullptr);
                 }
                 return py::make_tuple(out_ids, out_scores);
             },
             py::arg("queries"),
             py::arg("k"),
             py::arg("nprobe") = -1)
        // FSM-aware search overload: pass a RequestTrajectory to enable
        // FSM probe reordering, trajectory tracking, and predictive prefetch.
        .def("search",
             [](const MultiLevelIndex& idx,
                py::array_t<float, py::array::c_style> queries,
                int k,
                int nprobe,
                RequestTrajectory* traj) {
                 auto buf = queries.request();
                 if (buf.ndim != 2) throw std::runtime_error("queries must be 2D [Q, D]");
                 if (buf.shape[1] != idx.dim()) throw std::runtime_error("query dim mismatch");
                 std::vector<std::vector<DocId>> out_ids;
                 std::vector<std::vector<float>> out_scores;
                 {
                     py::gil_scoped_release _g;
                     idx.search((const float*)buf.ptr,
                                (size_t)buf.shape[0],
                                k,
                                nprobe,
                                out_ids,
                                out_scores,
                                traj);
                 }
                 return py::make_tuple(out_ids, out_scores);
             },
             py::arg("queries"),
             py::arg("k"),
             py::arg("nprobe") = -1,
             py::arg("traj") = nullptr,
             "Search with optional FSM trajectory tracking.\n\n"
             "Parameters\n"
             "----------\n"
             "queries : np.ndarray [Q, D] float32\n"
             "k       : int   number of results per query\n"
             "nprobe  : int   clusters to probe (-1 = all)\n"
             "traj    : RequestTrajectory or None\n"
             "    If provided: FSM predicts probe order at L0 and L1,\n"
             "    append_step() is called automatically after each layer,\n"
             "    and predictive prefetch is enqueued at the end.\n"
             "    Call traj.finalize() then fsm_table().update_from_trajectory()\n"
             "    after the full request completes.")
        // ---- FSM accessors ----
        .def("set_fsm_config",
             [](MultiLevelIndex& idx, const FSMConfig& cfg) {
                 idx.set_fsm_config(cfg);
             },
             py::arg("config"),
             "Replace the FSM config in-place (patterns are preserved).")
        .def("fsm_table",
             [](MultiLevelIndex& idx) -> FSMTable& {
                 return idx.fsm_table();
             },
             py::return_value_policy::reference_internal,
             "Return a reference to the index's FSMTable.\n"
             "Use this to call update_from_trajectory() and inspect patterns.")
        .def("maintenance_pass",
             [](MultiLevelIndex& idx) {
                 py::gil_scoped_release _g;
                 idx.maintenance_pass();
             })
        .def("dim", &MultiLevelIndex::dim)
        .def("metric", &MultiLevelIndex::metric)
        .def("normalized", &MultiLevelIndex::normalized);

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

      // Attach a MultiLevelIndex so that AsyncEngine's maintenance threads
      // call its maintenance_pass() (driving L0/L1/L2 cache maintenance).
      .def("attach_multilevel_index",
           [](AsyncEngine& e, MultiLevelIndex& idx) {
               e.set_multilevel_index(&idx);
           },
           py::arg("index"))

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

        // ---- split/merge helpers (sync) ----
        .def("split_cluster",
             [](AsyncEngine& e, int index_id, int cluster_id, size_t max_vectors_before_split) {
                 py::gil_scoped_release _g;
                 return e.split_cluster(index_id, cluster_id, max_vectors_before_split);
             },
             py::arg("index_id"),
             py::arg("cluster_id"),
             py::arg("max_vectors_before_split"))
        .def("merge_clusters",
             [](AsyncEngine& e, int index_id, int cluster_id_a, int cluster_id_b) {
                 py::gil_scoped_release _g;
                 e.merge_clusters(index_id, cluster_id_a, cluster_id_b);
             },
             py::arg("index_id"),
             py::arg("cluster_id_a"),
             py::arg("cluster_id_b"))
        .def("cluster_live_size",
             [](const AsyncEngine& e, int index_id, int cluster_id) {
                 py::gil_scoped_release _g;
                 return e.cluster_live_size(index_id, cluster_id);
             },
             py::arg("index_id"),
             py::arg("cluster_id"))
        .def("cluster_valid",
             [](const AsyncEngine& e, int index_id, int cluster_id) {
                 py::gil_scoped_release _g;
                 return e.cluster_valid(index_id, cluster_id);
             },
             py::arg("index_id"),
             py::arg("cluster_id"))

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

    // ==================================================================
    // FSM types — registered in the same module so RequestTrajectory can
    // be passed directly into MultiLevelIndex.search(traj=...) without
    // crossing module boundaries.
    // ==================================================================

    // Helper: 1-D float32 numpy → std::vector<float>
    auto np_to_vec = [](py::array_t<float, py::array::c_style> arr) {
        auto buf = arr.request();
        if (buf.ndim != 1)
            throw std::runtime_error("embedding must be a 1-D float32 array");
        const float* ptr = static_cast<const float*>(buf.ptr);
        return std::vector<float>(ptr, ptr + buf.shape[0]);
    };

    // Helper: flat or 2-D float32 numpy → flat std::vector<float>
    auto np_to_flat = [](py::array_t<float, py::array::c_style> arr) {
        auto buf = arr.request();
        if (buf.ndim != 1 && buf.ndim != 2)
            throw std::runtime_error(
                "cluster_centroids must be 1-D (nlist*dim,) or 2-D (nlist, dim) float32");
        const float* ptr = static_cast<const float*>(buf.ptr);
        return std::vector<float>(ptr, ptr + buf.size);
    };

    // ----- FSMLayer -----
    py::enum_<FSMLayer>(m, "FSMLayer",
            "Layer tag for a trajectory step (L0 or L1).\n"
            "L0 and L1 share the same cluster-id namespace.")
        .value("L0", FSMLayer::L0)
        .value("L1", FSMLayer::L1)
        .export_values();

    // ----- FSMStep -----
    py::class_<FSMStep>(m, "FSMStep",
            "One step in a RequestTrajectory.\n\n"
            "Attributes: cluster_id (int), layer (FSMLayer)")
        .def(py::init<>())
        .def_readwrite("cluster_id", &FSMStep::cluster_id)
        .def_readwrite("layer",      &FSMStep::layer)
        .def("__repr__", [](const FSMStep& s) {
            return "<FSMStep cluster_id=" + std::to_string(s.cluster_id)
                 + " layer=" + (s.layer == FSMLayer::L0 ? "L0" : "L1") + ">";
        });

    // ----- FSMConfig -----
    py::class_<FSMConfig>(m, "FSMConfig",
            "Tuning knobs for the FSM pattern table.\n\n"
            "max_patterns, max_states_per_fsm, merge_threshold, match_threshold,\n"
            "merge_dist, alpha_et (early-termination factor), dagent_window.")
        .def(py::init<>())
        .def_readwrite("max_patterns",       &FSMConfig::max_patterns)
        .def_readwrite("max_states_per_fsm", &FSMConfig::max_states_per_fsm)
        .def_readwrite("merge_threshold",    &FSMConfig::merge_threshold)
        .def_readwrite("match_threshold",    &FSMConfig::match_threshold)
        .def_readwrite("merge_dist",         &FSMConfig::merge_dist)
        .def_readwrite("alpha_et",           &FSMConfig::alpha_et)
        .def_readwrite("dagent_window",      &FSMConfig::dagent_window)
        .def("__repr__", [](const FSMConfig& c) {
            return "<FSMConfig max_patterns=" + std::to_string(c.max_patterns)
                 + " alpha_et=" + std::to_string(c.alpha_et)
                 + " match_threshold=" + std::to_string(c.match_threshold) + ">";
        });

    // ----- FSMState -----
    py::class_<FSMState>(m, "FSMState",
            "One state in an FSM pattern: cluster_id, delta, centroid.")
        .def(py::init<>())
        .def_readwrite("cluster_id", &FSMState::cluster_id)
        .def_readwrite("delta",      &FSMState::delta)
        .def_readwrite("centroid",   &FSMState::centroid)
        .def("__repr__", [](const FSMState& s) {
            return "<FSMState cluster_id=" + std::to_string(s.cluster_id)
                 + " delta=" + std::to_string(s.delta) + ">";
        });

    // ----- FSMPattern -----
    py::class_<FSMPattern>(m, "FSMPattern",
            "A single FSM entry P = (S, T).\n"
            "states: dict[int, FSMState], transitions: dict[int, set[int]]")
        .def(py::init<>())
        .def_readwrite("pattern_id",  &FSMPattern::pattern_id)
        .def_readwrite("frequency",   &FSMPattern::frequency)
        .def_readwrite("last_used",   &FSMPattern::last_used)
        .def_readwrite("states",      &FSMPattern::states)
        .def_readwrite("transitions", &FSMPattern::transitions)
        .def("has_transition",       &FSMPattern::has_transition,
             py::arg("from_cluster"), py::arg("to_cluster"))
        .def("predict_next",         &FSMPattern::predict_next,
             py::arg("last_cluster"))
        .def("predict_next_ranked",  &FSMPattern::predict_next_ranked,
             py::arg("last_cluster"),
             "All outgoing transitions sorted by delta ascending (tightest first).")
        .def("num_transitions",      &FSMPattern::num_transitions)
        .def("__repr__", [](const FSMPattern& p) {
            return "<FSMPattern id=" + std::to_string(p.pattern_id)
                 + " states=" + std::to_string(p.states.size())
                 + " freq=" + std::to_string(p.frequency) + ">";
        });

    // ----- RequestTrajectory -----
    py::class_<RequestTrajectory>(m, "RequestTrajectory",
            "Tracks the cluster-access sequence for one request.\n\n"
            "The C++ search() fills this automatically when passed as traj=...\n"
            "Python only needs to call finalize() then update_from_trajectory().")
        .def(py::init([](std::string rid) {
                 RequestTrajectory t;
                 t.request_id = std::move(rid);
                 return t;
             }),
             py::arg("request_id"))
        .def("append_step",
             [np_to_vec](RequestTrajectory& t,
                py::array_t<float, py::array::c_style> embedding,
                int cluster_id,
                FSMLayer layer,
                uint64_t time_ns) {
                 t.append_step(np_to_vec(embedding), cluster_id, layer, time_ns);
             },
             py::arg("embedding"),
             py::arg("cluster_id"),
             py::arg("layer"),
             py::arg("time_ns") = 0,
             "Record one completed search step manually (for offline use).")
        .def("finalize",     &RequestTrajectory::finalize,
             "Mark complete before calling update_from_trajectory().")
        .def("cluster_id_sequence",
             [](const RequestTrajectory& t) { return t.cluster_id_sequence(); },
             "Flat list of cluster ids (FSMLayer info stripped).")
        .def_readwrite("request_id",  &RequestTrajectory::request_id)
        .def_readwrite("steps",       &RequestTrajectory::steps)
        .def_readwrite("is_complete", &RequestTrajectory::is_complete)
        .def_property_readonly("length",
             [](const RequestTrajectory& t){ return t.length(); })
        .def_property_readonly("timestamps_ns",
             [](const RequestTrajectory& t) -> std::vector<uint64_t> {
                 return t.timestamps_ns;
             })
        .def("__repr__", [](const RequestTrajectory& t) {
            return "<RequestTrajectory id='" + t.request_id
                 + "' steps=" + std::to_string(t.length())
                 + " complete=" + (t.is_complete ? "True" : "False") + ">";
        });

    // ----- PredictResult -----
    py::class_<FSMTable::PredictResult>(m, "PredictResult",
            "Result of FSMTable.match_and_predict().\n\n"
            "ranked_clusters: list[int]  probe these first (empty = no prediction)\n"
            "best_score: float\n"
            "best_pattern_id: int")
        .def(py::init<>())
        .def_readwrite("ranked_clusters", &FSMTable::PredictResult::ranked_clusters)
        .def_readwrite("best_score",      &FSMTable::PredictResult::best_score)
        .def_readwrite("best_pattern_id", &FSMTable::PredictResult::best_pattern_id)
        .def("__repr__", [](const FSMTable::PredictResult& r) {
            std::string s = "<PredictResult ranked=[";
            for (size_t i = 0; i < r.ranked_clusters.size(); ++i) {
                if (i) s += ",";
                s += std::to_string(r.ranked_clusters[i]);
            }
            return s + "] score=" + std::to_string(r.best_score) + ">";
        });

    // ----- FSMTable -----
    py::class_<FSMTable>(m, "FSMTable",
            "Np-capped FSM pattern table. Thread-safe.\n\n"
            "Obtain via idx.fsm_table() rather than constructing directly.")
        .def(py::init<>())
        .def(py::init<FSMConfig>(), py::arg("config"))
        .def("match_and_predict",
             [](FSMTable& t, const RequestTrajectory& traj, FSMLayer layer) {
                 return t.match_and_predict(traj, layer);
             },
             py::arg("trajectory"),
             py::arg("layer") = FSMLayer::L0,
             "Return PredictResult.ranked_clusters — clusters to probe first.")
        .def("update_from_trajectory",
             [np_to_flat](FSMTable& t,
                const RequestTrajectory& traj,
                py::array_t<float, py::array::c_style> centroids,
                int dim,
                float delta_default) {
                 t.update_from_trajectory(traj, np_to_flat(centroids), dim, delta_default);
             },
             py::arg("trajectory"),
             py::arg("cluster_centroids"),
             py::arg("dim"),
             py::arg("cluster_delta_default") = 1.0f,
             "Update or create a pattern from a finalized trajectory.\n"
             "cluster_centroids: np.ndarray float32 shape (nlist*dim,) or (nlist, dim)")
        .def("invalidate_cluster", &FSMTable::invalidate_cluster,
             py::arg("cluster_id"),
             "Remove all states/transitions for cluster_id (call on topology change).")
        .def("set_config", &FSMTable::set_config, py::arg("config"),
             "Replace config in-place (patterns preserved).")
        .def("compute_similarity_transitions",
             [](FSMTable& t, const FSMPattern& pat, const std::vector<int>& seq) {
                 return t.compute_similarity_transitions(pat, seq);
             },
             py::arg("pattern"), py::arg("cluster_sequence"))
        .def("compute_similarity_full",
             [](FSMTable& t, const FSMPattern& pat,
                const std::vector<int>& seq,
                py::array_t<float, py::array::c_style> emb) {
                 auto buf = emb.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("embeddings must be 2-D [T, D]");
                 const float* ptr = static_cast<const float*>(buf.ptr);
                 std::vector<std::vector<float>> eseq(static_cast<size_t>(buf.shape[0]));
                 for (ssize_t r = 0; r < buf.shape[0]; ++r)
                     eseq[static_cast<size_t>(r)].assign(
                         ptr + r * buf.shape[1], ptr + (r + 1) * buf.shape[1]);
                 return t.compute_similarity_full(pat, seq, eseq);
             },
             py::arg("pattern"), py::arg("cluster_sequence"), py::arg("embeddings"))
        .def("num_patterns",  &FSMTable::num_patterns)
        .def("get_pattern",
             [](FSMTable& t, int pid) -> py::object {
                 const FSMPattern* p = t.get_pattern(pid);
                 if (!p) return py::none();
                 return py::cast(*p);
             },
             py::arg("pattern_id"))
        .def("merge_patterns", &FSMTable::merge_patterns,
             py::arg("pattern_id_a"), py::arg("pattern_id_b"))
        .def("clear",  &FSMTable::clear)
        .def("config", &FSMTable::config)
        .def("__repr__", [](const FSMTable& t) {
            return "<FSMTable patterns=" + std::to_string(t.num_patterns()) + ">";
        });
}
