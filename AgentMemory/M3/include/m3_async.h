#pragma once

#include <pthread.h>
#include <deque>
#include <variant>
#include <atomic>
#include <memory>
#include <vector>
#include <cstdint>
#include <unordered_map>

#include "base.h"     // Metric, DocId
#include "m3_index.h" // IVFIndex

namespace m3 {

// ================= write ops (per-index, per-cluster) =================

struct InsertOp {
    int index_id;     // which IVF
    int cluster_id;   // which cluster inside that IVF
    std::vector<DocId> ids;   // len = n_rows
    std::vector<float> vecs;  // row-major, len = n_rows * dim
};

struct UpdateOp {
    int index_id;
    int cluster_id;
    std::vector<DocId> ids;
    std::vector<float> vecs;
    bool insert_if_absent = false;
};

struct DeleteOp {
    int index_id;
    int cluster_id;
    std::vector<DocId> ids;
};

using WriteOp = std::variant<InsertOp, UpdateOp, DeleteOp>;

// ================= policies =================

struct AsyncQueuePolicy {
    size_t capacity       = 4096;
    size_t pop_batch_max  = 64;
    bool   block_on_full  = true;
};

struct AsyncSearchPolicy {
    int  default_nprobe    = 8;
    bool parallel_queries  = false;
};

struct AsyncMaintenancePolicy {
    double period_sec      = 1.0;
    size_t split_threshold = 200000;
    double compact_ratio   = 0.7;
};

// ================= AsyncEngine (multi-index) =================

class AsyncEngine {
public:
    AsyncEngine();
    ~AsyncEngine();

    // ---- index lifecycle ----
    // create one IVF index under index_id
    // centroids: [nlist, dim]
    void create_ivf(int index_id,
                    int dim,
                    Metric metric,
                    bool normalized,
                    const std::vector<float>& centroids);

    // start background threads
    void start(int num_writer_threads = 2,
               int num_maintenance_threads = 1);

    void stop();

    void set_queue_policy(const AsyncQueuePolicy& p);
    void set_search_policy(const AsyncSearchPolicy& p);
    void set_maintenance_policy(const AsyncMaintenancePolicy& p);

    // ---- async writes ----
    bool enqueue_insert(InsertOp op);
    bool enqueue_update(UpdateOp op);
    bool enqueue_delete(DeleteOp op);

    // wait until queue is drained and applied
    void flush();

    // ---- search (sync) ----
    // search on a specific index
    void search(int index_id,
                const float* queries, size_t q_rows, int k, int nprobe,
                std::vector<std::vector<DocId>>& out_ids,
                std::vector<std::vector<float>>& out_scores) const;

    void search(int index_id,
                const float* queries, size_t q_rows, int k,
                std::vector<std::vector<DocId>>& out_ids,
                std::vector<std::vector<float>>& out_scores) const;

    // ---- direct cluster rebuild (sync, bypasses async queue) ----
    void load_cluster(int index_id,
                      int cluster_id,
                      const std::vector<DocId>& ids,
                      const std::vector<float>& vecs);

    bool enqueue_insert_auto(int index_id,
                             const std::vector<DocId>& ids,
                             const std::vector<float>& vecs);

    // ---- basic info ----
    int    dim_of(int index_id) const;
    Metric metric_of(int index_id) const;
    bool   normalized_of(int index_id) const;
    int    nlist_of(int index_id) const;

private:
    // index_id -> IVFIndex
    std::unordered_map<int, std::shared_ptr<IVFIndex>> indices_;

    // protects indices_ map and index pointers
    mutable pthread_rwlock_t indices_rwlock_;

    // global bounded queue
    pthread_mutex_t q_mtx_;
    pthread_cond_t  q_not_empty_;
    pthread_cond_t  q_not_full_;
    std::deque<WriteOp> queue_;
    AsyncQueuePolicy    q_policy_;

    // background threads
    std::vector<pthread_t> writer_threads_;
    std::vector<pthread_t> maintenance_threads_;

    // policies
    AsyncSearchPolicy      search_policy_;
    AsyncMaintenancePolicy maint_policy_;

    std::atomic<bool> running_{false};

    // stats
    std::atomic<uint64_t> writes_enqueued_{0};
    std::atomic<uint64_t> writes_applied_{0};

private:
    // worker entrypoints
    static void* writer_main(void* arg);
    void writer_loop();

    static void* maintenance_main(void* arg);
    void maintenance_loop();

    // queue helpers
    bool pop_batch(std::vector<WriteOp>& batch, size_t max_n);

    // apply a batch to the right IVF index using a snapshot of all indices
    void apply_batch_noindexlock(
        const std::vector<WriteOp>& batch,
        const std::unordered_map<int, std::shared_ptr<IVFIndex>>& indices_snap);

    // maintenance
    void run_maintenance_once();
};

} // namespace m3
