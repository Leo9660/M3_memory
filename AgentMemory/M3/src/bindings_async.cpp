#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <cstring>

#include "m3_async.h"
#include "m3_multi_level.h"
#include "gpu_coordinator.h"

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

    // ----- DagentUpdateMode enum -----
    py::enum_<DagentUpdateMode>(m, "DagentUpdateMode")
        .value("cache_level_k", DagentUpdateMode::cache_level_k)
        .value("true_k",        DagentUpdateMode::true_k)
        .export_values();

    // ----- CacheConfig -----
    py::class_<CacheConfig>(m, "CacheConfig")
        .def(py::init<>())
        .def_readwrite("l0_max_clusters",            &CacheConfig::l0_max_clusters)
        .def_readwrite("l0_max_vectors_per_cluster", &CacheConfig::l0_max_vectors_per_cluster)
        .def_readwrite("l1_max_clusters",            &CacheConfig::l1_max_clusters)
        .def_readwrite("l1_max_vectors_per_cluster", &CacheConfig::l1_max_vectors_per_cluster)
        .def_readwrite("l0_eviction_ratio",          &CacheConfig::l0_eviction_ratio)
        .def_readwrite("l1_eviction_ratio",          &CacheConfig::l1_eviction_ratio)
        .def_readwrite("cold_time_ns",               &CacheConfig::cold_time_ns)
        .def_readwrite("l1_neighborhood_k",          &CacheConfig::l1_neighborhood_k)
        .def_readwrite("l0_neighborhood_k",          &CacheConfig::l0_neighborhood_k)
        .def_readwrite("max_promote_per_query",      &CacheConfig::max_promote_per_query)
        .def_readwrite("l0_nprobe",                  &CacheConfig::l0_nprobe)
        .def_readwrite("l1_nprobe",                  &CacheConfig::l1_nprobe)
        .def_readwrite("alpha_et",                   &CacheConfig::alpha_et)
        .def_readwrite("dagent_window",              &CacheConfig::dagent_window)
        .def_readwrite("dagent_mode",                &CacheConfig::dagent_mode)
        .def_readwrite("calibration_interval",       &CacheConfig::calibration_interval)
        .def_readwrite("alpha_et_adapt_rate",        &CacheConfig::alpha_et_adapt_rate);

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
        .def("get_cache_stats",
             [](const MultiLevelIndex& idx) {
                 auto s = idx.get_cache_stats();
                 py::dict d;
                 d["l0_clusters"]       = s.l0_clusters;
                 d["l0_total_vecs"]     = s.l0_total_vecs;
                 d["l1_clusters"]       = s.l1_clusters;
                 d["l1_total_vecs"]     = s.l1_total_vecs;
                 d["l1_dedup_set_size"] = s.l1_dedup_set_size;
                 d["l0_l1_overlap"]     = s.l0_l1_overlap;
                 d["dagent"]            = s.dagent;
                 d["dynamic_threshold"] = s.dynamic_threshold;
                 return d;
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
                                out_scores);
                 }
                 return py::make_tuple(out_ids, out_scores);
             },
             py::arg("queries"),
             py::arg("k"),
             py::arg("nprobe") = -1)
        .def("maintenance_pass",
             [](MultiLevelIndex& idx) {
                 py::gil_scoped_release _g;
                 idx.maintenance_pass();
             })
        .def("load_cluster",
             [](MultiLevelIndex& idx,
                int cid,
                py::array_t<int64_t, py::array::c_style> ids,
                py::array_t<float,   py::array::c_style> vecs) {
                 auto idb = ids.request();
                 auto vb  = vecs.request();
                 if (idb.ndim != 1) throw std::runtime_error("ids must be 1D [N]");
                 if (vb.ndim != 2)  throw std::runtime_error("vectors must be 2D [N, D]");
                 if (idb.shape[0] != vb.shape[0]) throw std::runtime_error("ids.size != vectors.N");
                 if (vb.shape[1] != idx.dim()) throw std::runtime_error("vectors dim mismatch");
                 py::gil_scoped_release _g;
                 idx.load_cluster(cid, (const DocId*)idb.ptr, (const float*)vb.ptr,
                                  (size_t)idb.shape[0]);
             },
             py::arg("cluster_id"),
             py::arg("ids"),
             py::arg("vectors"))
        .def("dim", &MultiLevelIndex::dim)
        .def("metric", &MultiLevelIndex::metric)
        .def("normalized", &MultiLevelIndex::normalized)
        // Wire a GpuCoordinator into this index (pass None to disconnect).
        .def("set_gpu_coordinator",
             [](MultiLevelIndex& idx, py::object coord_obj) {
                 if (coord_obj.is_none()) {
                     idx.set_gpu_coordinator(nullptr);
                 } else {
                     auto& coord = coord_obj.cast<GpuCoordinator&>();
                     idx.set_gpu_coordinator(&coord);
                 }
             },
             py::arg("coordinator"))
#ifdef M3_WITH_FSM
        // ---- FSM-aware search (only available when built with M3_WITH_FSM) ----
        .def("search_fsm",
             [](const MultiLevelIndex& idx,
                py::array_t<float, py::array::c_style> queries,
                int k,
                int nprobe,
                py::object fsm_table_obj,
                py::object traj_obj) {
                 auto buf = queries.request();
                 if (buf.ndim != 2) throw std::runtime_error("queries must be 2D [Q, D]");
                 if (buf.shape[1] != idx.dim()) throw std::runtime_error("query dim mismatch");

                 const fsm::FSMTable*    fsm_ptr  = nullptr;
                 fsm::RequestTrajectory* traj_ptr = nullptr;
                 if (!fsm_table_obj.is_none())
                     fsm_ptr  = &fsm_table_obj.cast<fsm::FSMTable&>();
                 if (!traj_obj.is_none())
                     traj_ptr = &traj_obj.cast<fsm::RequestTrajectory&>();

                 std::vector<std::vector<DocId>> out_ids;
                 std::vector<std::vector<float>> out_scores;
                 {
                     py::gil_scoped_release _g;
                     idx.search_fsm((const float*)buf.ptr,
                                    (size_t)buf.shape[0],
                                    k, nprobe,
                                    fsm_ptr, traj_ptr,
                                    out_ids, out_scores);
                 }
                 return py::make_tuple(out_ids, out_scores);
             },
             py::arg("queries"),
             py::arg("k"),
             py::arg("nprobe")    = -1,
             py::arg("fsm_table") = py::none(),
             py::arg("traj")      = py::none())
        .def("nearest_l2_centroid",
             [](const MultiLevelIndex& idx,
                py::array_t<float, py::array::c_style> query) {
                 auto buf = query.request();
                 if (buf.ndim != 1 && !(buf.ndim == 2 && buf.shape[0] == 1))
                     throw std::runtime_error("query must be 1D [D] or 2D [1,D]");
                 const float* qptr = (const float*)buf.ptr;
                 py::gil_scoped_release _g;
                 return idx.nearest_l2_centroid(qptr);
             },
             py::arg("query"))
#endif // M3_WITH_FSM
        ; // end MultiLevelIndex

#ifdef M3_WITH_FSM
    // ----- fsm::RequestTrajectory -----
    py::class_<fsm::RequestTrajectory>(m, "RequestTrajectory")
        .def(py::init<std::string>(), py::arg("request_id"))
        .def("append_step",
             [](fsm::RequestTrajectory& t,
                int cid,
                py::array_t<float, py::array::c_style> vec) {
                 auto buf = vec.request();
                 if (buf.ndim != 1 && !(buf.ndim == 2 && buf.shape[0] == 1))
                     throw std::runtime_error("vec must be 1D [D] or 2D [1,D]");
                 t.append_step(cid, (const float*)buf.ptr, (int)buf.size);
             },
             py::arg("cid"),
             py::arg("vec"))
        .def("length",     &fsm::RequestTrajectory::length)
        .def("clear",      &fsm::RequestTrajectory::clear)
        .def_readonly("request_id", &fsm::RequestTrajectory::request_id);

    // ----- fsm::FSMConfig -----
    py::class_<fsm::FSMConfig>(m, "FSMConfig")
        .def(py::init<>())
        .def_readwrite("max_patterns",        &fsm::FSMConfig::max_patterns)
        .def_readwrite("ns_max_states",       &fsm::FSMConfig::ns_max_states)
        .def_readwrite("d_merge",             &fsm::FSMConfig::d_merge)
        .def_readwrite("reinforce_threshold", &fsm::FSMConfig::reinforce_threshold)
        .def_readwrite("min_hits_to_predict", &fsm::FSMConfig::min_hits_to_predict)
        .def_readwrite("min_traj_len",        &fsm::FSMConfig::min_traj_len);

    // ----- fsm::FSMTable -----
    py::class_<fsm::FSMTable>(m, "FSMTable")
        .def(py::init<fsm::FSMConfig>(), py::arg("config") = fsm::FSMConfig())
        .def("match_and_predict",
             [](const fsm::FSMTable& t, const fsm::RequestTrajectory& traj) {
                 py::gil_scoped_release _g;
                 return t.match_and_predict(traj);
             },
             py::arg("traj"))
        .def("update_from_trajectory",
             [](fsm::FSMTable& t,
                const fsm::RequestTrajectory& traj,
                py::array_t<float, py::array::c_style> centroids) {
                 auto buf = centroids.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("centroids must be 2D [nlist, dim]");
                 {
                     py::gil_scoped_release _g;
                     t.update_from_trajectory(traj,
                                              (const float*)buf.ptr,
                                              (int)buf.shape[0],
                                              (int)buf.shape[1]);
                 }
             },
             py::arg("traj"),
             py::arg("centroids"))
        .def("num_patterns", &fsm::FSMTable::num_patterns)
        .def("total_hits",   &fsm::FSMTable::total_hits);
#endif // M3_WITH_FSM

    // ----- GpuCoordinator -----
    py::class_<GpuCoordinator>(m, "GpuCoordinator")
        // keep_alive<0,1>: the new GpuCoordinator keeps idx alive (idx must outlive coordinator).
        .def(py::init([](MultiLevelIndex& idx,
                         size_t gpu_budget_bytes,
                         int    dim,
                         Metric metric,
                         bool   normalized,
                         size_t insert_buf_cap) {
                 return new GpuCoordinator(idx, gpu_budget_bytes, dim,
                                           metric, normalized, insert_buf_cap);
             }),
             py::arg("idx"),
             py::arg("gpu_budget_bytes"),
             py::arg("dim"),
             py::arg("metric"),
             py::arg("normalized")     = false,
             py::arg("insert_buf_cap") = 128,
             py::keep_alive<0, 1>())   // GpuCoordinator (0) keeps idx (1) alive
        .def("promote_to_gpu",
             &GpuCoordinator::promote_to_gpu,
             py::arg("cid"),
             py::call_guard<py::gil_scoped_release>())
        .def("enqueue_promote", &GpuCoordinator::enqueue_promote, py::arg("cid"))
        .def("enqueue_demote",  &GpuCoordinator::enqueue_demote,  py::arg("cid"))
        .def("drain_pending",
             &GpuCoordinator::drain_pending,
             py::call_guard<py::gil_scoped_release>())
        .def("flush_buffers",
             &GpuCoordinator::flush_buffers,
             py::call_guard<py::gil_scoped_release>())
        .def("rebalance",
             &GpuCoordinator::rebalance,
             py::call_guard<py::gil_scoped_release>())
        .def("start_background",
             &GpuCoordinator::start_background,
             py::arg("flush_ms")        =    500,
             py::arg("maintenance_ms")  =   5000,
             py::arg("rebalance_ms")    =    500,
             py::arg("split_every_ops") =  20000,
             py::arg("split_threshold") = 200000)
        .def("stop_background",
             &GpuCoordinator::stop_background,
             py::call_guard<py::gil_scoped_release>())
        .def("is_gpu_resident",    &GpuCoordinator::is_gpu_resident,    py::arg("cid"))
        .def("gpu_bytes_used",     &GpuCoordinator::gpu_bytes_used)
        .def("gpu_budget_bytes",   &GpuCoordinator::gpu_budget_bytes)
        .def("gpu_resident_cids",  &GpuCoordinator::gpu_resident_cids)
        .def("background_running", &GpuCoordinator::background_running);

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

        // ---- search(index_id, q, k) / search(index_id, q, k, nprobe) ----
        // Returns (ids, scores) as numpy arrays of shape [Q, k] (int64 / float32).
        // Empty slots are filled with id=-1, score=inf  (same convention as FAISS).
        // Callers iterating over rows should skip entries where id == -1.
        .def("search",
             [](AsyncEngine& e,
                int index_id,
                py::array_t<float, py::array::c_style> queries,
                int k) {
                 auto buf = queries.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("queries must be 2D [Q, D]");
                 int expected_dim = e.dim_of(index_id);
                 if (expected_dim <= 0)
                     throw std::runtime_error("search: unknown index_id");
                 if (buf.shape[1] != expected_dim)
                     throw std::runtime_error("search: query dim mismatch");

                 std::vector<std::vector<DocId>> out_ids;
                 std::vector<std::vector<float>> out_scores;
                 {
                     py::gil_scoped_release _g;
                     e.search(index_id, (const float*)buf.ptr,
                              (size_t)buf.shape[0], k, /*nprobe=*/-1,
                              out_ids, out_scores);
                 }
                 const py::ssize_t Q = (py::ssize_t)out_ids.size();
                 auto ids_arr    = py::array_t<int64_t>({Q, (py::ssize_t)k});
                 auto scores_arr = py::array_t<float>  ({Q, (py::ssize_t)k});
                 auto ids_p    = ids_arr.mutable_unchecked<2>();
                 auto scores_p = scores_arr.mutable_unchecked<2>();
                 for (py::ssize_t qi = 0; qi < Q; ++qi) {
                     const auto& ri = out_ids[(size_t)qi];
                     const auto& rs = out_scores[(size_t)qi];
                     for (int j = 0; j < k; ++j) {
                         ids_p(qi,j)    = j < (int)ri.size() ? ri[j]   : -1;
                         scores_p(qi,j) = j < (int)rs.size() ? rs[j]
                                          : std::numeric_limits<float>::infinity();
                     }
                 }
                 return py::make_tuple(ids_arr, scores_arr);
             },
             py::arg("index_id"), py::arg("queries"), py::arg("k"))

        .def("search",
             [](AsyncEngine& e,
                int index_id,
                py::array_t<float, py::array::c_style> queries,
                int k,
                int nprobe) {
                 auto buf = queries.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("queries must be 2D [Q, D]");
                 int expected_dim = e.dim_of(index_id);
                 if (expected_dim <= 0)
                     throw std::runtime_error("search: unknown index_id");
                 if (buf.shape[1] != expected_dim)
                     throw std::runtime_error("search: query dim mismatch");

                 std::vector<std::vector<DocId>> out_ids;
                 std::vector<std::vector<float>> out_scores;
                 {
                     py::gil_scoped_release _g;
                     e.search(index_id, (const float*)buf.ptr,
                              (size_t)buf.shape[0], k, nprobe,
                              out_ids, out_scores);
                 }
                 const py::ssize_t Q = (py::ssize_t)out_ids.size();
                 auto ids_arr    = py::array_t<int64_t>({Q, (py::ssize_t)k});
                 auto scores_arr = py::array_t<float>  ({Q, (py::ssize_t)k});
                 auto ids_p    = ids_arr.mutable_unchecked<2>();
                 auto scores_p = scores_arr.mutable_unchecked<2>();
                 for (py::ssize_t qi = 0; qi < Q; ++qi) {
                     const auto& ri = out_ids[(size_t)qi];
                     const auto& rs = out_scores[(size_t)qi];
                     for (int j = 0; j < k; ++j) {
                         ids_p(qi,j)    = j < (int)ri.size() ? ri[j]   : -1;
                         scores_p(qi,j) = j < (int)rs.size() ? rs[j]
                                          : std::numeric_limits<float>::infinity();
                     }
                 }
                 return py::make_tuple(ids_arr, scores_arr);
             },
             py::arg("index_id"), py::arg("queries"), py::arg("k"), py::arg("nprobe"))

        // ---- search_on_batch(index_id, queries, k, cluster_ids) ----
        // Vector scan only — skip M3 centroid scoring entirely.
        // cluster_ids: int32 [Q, nprobe] of original cluster IDs (e.g. from faiss quantizer.search).
        // Returns (ids [Q,k] int64, scores [Q,k] float32).
        .def("search_on_batch",
             [](AsyncEngine& e,
                int index_id,
                py::array_t<float,   py::array::c_style> queries,
                int k,
                py::array_t<int32_t, py::array::c_style> cluster_ids) {
                 auto qbuf = queries.request();
                 auto cbuf = cluster_ids.request();
                 if (qbuf.ndim != 2)
                     throw std::runtime_error("search_on_batch: queries must be 2D [Q, D]");
                 if (cbuf.ndim != 2)
                     throw std::runtime_error("search_on_batch: cluster_ids must be 2D [Q, nprobe]");
                 const size_t q_rows = (size_t)qbuf.shape[0];
                 const int    nprobe = (int)cbuf.shape[1];
                 if ((size_t)cbuf.shape[0] != q_rows)
                     throw std::runtime_error("search_on_batch: cluster_ids row count != queries row count");

                 std::vector<std::vector<DocId>> out_ids;
                 std::vector<std::vector<float>> out_scores;
                 {
                     py::gil_scoped_release _g;
                     e.search_on_batch(index_id,
                                       (const float*)qbuf.ptr, q_rows, k,
                                       (const int*)cbuf.ptr,   nprobe,
                                       out_ids, out_scores);
                 }
                 const py::ssize_t Q = (py::ssize_t)out_ids.size();
                 auto ids_arr    = py::array_t<int64_t>({Q, (py::ssize_t)k});
                 auto scores_arr = py::array_t<float>  ({Q, (py::ssize_t)k});
                 auto ids_p    = ids_arr.mutable_unchecked<2>();
                 auto scores_p = scores_arr.mutable_unchecked<2>();
                 for (py::ssize_t qi = 0; qi < Q; ++qi) {
                     const auto& ri = out_ids[(size_t)qi];
                     const auto& rs = out_scores[(size_t)qi];
                     for (int j = 0; j < k; ++j) {
                         ids_p(qi,j)    = j < (int)ri.size() ? ri[j]   : -1;
                         scores_p(qi,j) = j < (int)rs.size() ? rs[j]
                                          : std::numeric_limits<float>::infinity();
                     }
                 }
                 return py::make_tuple(ids_arr, scores_arr);
             },
             py::arg("index_id"), py::arg("queries"), py::arg("k"), py::arg("cluster_ids"))

        // ---- score_centroids(index_id, q) -> (scores [Q, NL], orig_ids [NL]) ----
        // Returns raw centroid distances before any top-nprobe selection.
        // scores[qi, ci] = M3's distance from query qi to compact centroid ci.
        // orig_ids[ci]   = the original FAISS cluster ID for compact centroid ci.
        // Reindex in Python: m3_by_orig[qi, orig_ids[ci]] = scores[qi, ci]
        .def("score_centroids",
             [](AsyncEngine& e,
                int index_id,
                py::array_t<float, py::array::c_style> queries) {
                 auto buf = queries.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("score_centroids: queries must be 2D [Q, D]");
                 const size_t q_rows = (size_t)buf.shape[0];

                 std::vector<float> out_scores;
                 std::vector<int>   out_orig_ids;
                 {
                     py::gil_scoped_release _g;
                     e.score_centroids(index_id, (const float*)buf.ptr,
                                       q_rows, out_scores, out_orig_ids);
                 }

                 const py::ssize_t NL = (py::ssize_t)out_orig_ids.size();

                 auto scores_arr = py::array_t<float>(
                     {(py::ssize_t)q_rows, NL},
                     out_scores.data());
                 auto ids_arr = py::array_t<int32_t>(
                     {NL},
                     out_orig_ids.data());

                 return py::make_tuple(scores_arr, ids_arr);
             },
             py::arg("index_id"), py::arg("queries"))

        // ---- select_clusters(index_id, q, nprobe) -> int32 array [Q, nprobe] ----
        // Returns the exact cluster IDs M3's C++ centroid scoring picks — no vector scan.
        // Compare against faiss_index.quantizer.search(q, nprobe)[1] to see whether
        // the recall gap comes from cluster selection or from within-cluster vector scan.
        .def("select_clusters",
             [](AsyncEngine& e,
                int index_id,
                py::array_t<float, py::array::c_style> queries,
                int nprobe) {
                 auto buf = queries.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("select_clusters: queries must be 2D [Q, D]");
                 const size_t q_rows = (size_t)buf.shape[0];

                 std::vector<std::vector<int>> out;
                 {
                     py::gil_scoped_release _g;
                     e.select_clusters(index_id, (const float*)buf.ptr,
                                       q_rows, nprobe, out);
                 }

                 // Pack into [Q, nprobe] int32 array; -1 for any unfilled slot.
                 auto result = py::array_t<int32_t>({(py::ssize_t)q_rows,
                                                     (py::ssize_t)nprobe});
                 auto r = result.mutable_unchecked<2>();
                 for (py::ssize_t qi = 0; qi < (py::ssize_t)q_rows; ++qi) {
                     for (int j = 0; j < nprobe; ++j)
                         r(qi, j) = (j < (int)out[(size_t)qi].size())
                                    ? out[(size_t)qi][j] : -1;
                 }
                 return result;
             },
             py::arg("index_id"), py::arg("queries"), py::arg("nprobe"))

        // ---- search_profiled(index_id, q, k, nprobe) -> (ids, scores, profile_dict) ----
        // profile_dict keys (all in milliseconds, cumulative across OMP threads):
        //   snapshot_ms, c_norms_ms, sgemm_ms, select_ms, lock_ms, scan_ms, output_ms
        //   n_queries, n_clusters, nprobe_used
        .def("search_profiled",
             [](AsyncEngine& e,
                int index_id,
                py::array_t<float, py::array::c_style> queries,
                int k,
                int nprobe) {
                 auto buf = queries.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("queries must be 2D [Q, D]");
                 int expected_dim = e.dim_of(index_id);
                 if (expected_dim <= 0)
                     throw std::runtime_error("search_profiled: unknown index_id");
                 if (buf.shape[1] != expected_dim)
                     throw std::runtime_error("search_profiled: query dim mismatch");

                 std::vector<std::vector<DocId>> out_ids;
                 std::vector<std::vector<float>> out_scores;
                 IVFIndex::SearchProfile prof;

                 {
                     py::gil_scoped_release _g;
                     e.search_profiled(index_id,
                                       (const float*)buf.ptr,
                                       (size_t)buf.shape[0],
                                       k, nprobe,
                                       out_ids, out_scores, prof);
                 }

                 const py::ssize_t Q = (py::ssize_t)out_ids.size();
                 auto ids_arr    = py::array_t<int64_t>({Q, (py::ssize_t)k});
                 auto scores_arr = py::array_t<float>  ({Q, (py::ssize_t)k});
                 auto ids_p    = ids_arr.mutable_unchecked<2>();
                 auto scores_p = scores_arr.mutable_unchecked<2>();
                 for (py::ssize_t qi = 0; qi < Q; ++qi) {
                     const auto& ri = out_ids[(size_t)qi];
                     const auto& rs = out_scores[(size_t)qi];
                     for (int j = 0; j < k; ++j) {
                         ids_p(qi,j)    = j < (int)ri.size() ? ri[j]   : -1;
                         scores_p(qi,j) = j < (int)rs.size() ? rs[j]
                                          : std::numeric_limits<float>::infinity();
                     }
                 }

                 py::dict pd;
                 pd["snapshot_ms"]  = prof.snapshot_ms;
                 pd["c_norms_ms"]   = prof.c_norms_ms;
                 pd["sgemm_ms"]     = prof.sgemm_ms;
                 pd["select_ms"]    = prof.select_ms;
                 pd["lock_ms"]      = prof.lock_ms;
                 pd["scan_ms"]      = prof.scan_ms;
                 pd["output_ms"]    = prof.output_ms;
                 pd["n_queries"]    = prof.n_queries;
                 pd["n_clusters"]   = prof.n_clusters;
                 pd["nprobe_used"]  = prof.nprobe_used;
                 return py::make_tuple(ids_arr, scores_arr, pd);
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
