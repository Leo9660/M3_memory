#include "m3_async.h"

#include <stdexcept>
#include <algorithm>
#include <utility>
#include <chrono>
#include <thread>
#include <unordered_map>

namespace m3 {

// ==================== ctor / dtor ====================

AsyncEngine::AsyncEngine() {
    // protects indices_ map
    pthread_rwlock_init(&indices_rwlock_, nullptr);

    // queue lock & cond
    pthread_mutex_init(&q_mtx_, nullptr);
    pthread_cond_init(&q_not_empty_, nullptr);
    pthread_cond_init(&q_not_full_, nullptr);
}

AsyncEngine::~AsyncEngine() {
    stop();

    pthread_rwlock_destroy(&indices_rwlock_);

    pthread_mutex_destroy(&q_mtx_);
    pthread_cond_destroy(&q_not_empty_);
    pthread_cond_destroy(&q_not_full_);
}

// ==================== index lifecycle ====================

void AsyncEngine::create_ivf(int index_id,
                             int dim,
                             Metric metric,
                             bool normalized,
                             const std::vector<float>& centroids) {
    auto idx = std::make_shared<IVFIndex>(dim, metric, normalized);
    if (!centroids.empty()) {
        idx->set_centroids(centroids);
    }

    pthread_rwlock_wrlock(&indices_rwlock_);
    indices_[index_id] = std::move(idx);
    pthread_rwlock_unlock(&indices_rwlock_);
}

// ==================== start / stop ====================

void AsyncEngine::start(int num_writer_threads, int num_maintenance_threads) {
    if (running_.exchange(true)) {
        return; // already running
    }

    // writer threads
    writer_threads_.resize(num_writer_threads);
    for (int i = 0; i < num_writer_threads; ++i) {
        if (pthread_create(&writer_threads_[i], nullptr,
                           &AsyncEngine::writer_main, this) != 0) {
            running_ = false;
            throw std::runtime_error("AsyncEngine::start: failed to create writer thread");
        }
    }

    // maintenance threads
    maintenance_threads_.resize(num_maintenance_threads);
    for (int i = 0; i < num_maintenance_threads; ++i) {
        if (pthread_create(&maintenance_threads_[i], nullptr,
                           &AsyncEngine::maintenance_main, this) != 0) {
            running_ = false;
            throw std::runtime_error("AsyncEngine::start: failed to create maintenance thread");
        }
    }
}

void AsyncEngine::stop() {
    if (!running_.exchange(false)) {
        return; // already stopped
    }

    // wake up anyone waiting on the queue
    pthread_mutex_lock(&q_mtx_);
    pthread_cond_broadcast(&q_not_empty_);
    pthread_cond_broadcast(&q_not_full_);
    pthread_mutex_unlock(&q_mtx_);

    // join writers
    for (auto& th : writer_threads_) {
        if (th) {
            pthread_join(th, nullptr);
        }
    }
    writer_threads_.clear();

    // join maintenance
    for (auto& th : maintenance_threads_) {
        if (th) {
            pthread_join(th, nullptr);
        }
    }
    maintenance_threads_.clear();
}

// ==================== policy setters ====================

void AsyncEngine::set_queue_policy(const AsyncQueuePolicy& p) {
    q_policy_ = p;
}

void AsyncEngine::set_search_policy(const AsyncSearchPolicy& p) {
    search_policy_ = p;
}

void AsyncEngine::set_maintenance_policy(const AsyncMaintenancePolicy& p) {
    maint_policy_ = p;
}

// ==================== enqueue (3 kinds) ====================

bool AsyncEngine::enqueue_insert(InsertOp op) {
    pthread_mutex_lock(&q_mtx_);
    while (running_ && queue_.size() >= q_policy_.capacity) {
        if (!q_policy_.block_on_full) {
            pthread_mutex_unlock(&q_mtx_);
            return false;
        }
        pthread_cond_wait(&q_not_full_, &q_mtx_);
    }
    if (!running_) {
        pthread_mutex_unlock(&q_mtx_);
        return false;
    }

    queue_.emplace_back(std::move(op));
    ++writes_enqueued_;

    pthread_cond_signal(&q_not_empty_);
    pthread_mutex_unlock(&q_mtx_);
    return true;
}

bool AsyncEngine::enqueue_update(UpdateOp op) {
    pthread_mutex_lock(&q_mtx_);
    while (running_ && queue_.size() >= q_policy_.capacity) {
        if (!q_policy_.block_on_full) {
            pthread_mutex_unlock(&q_mtx_);
            return false;
        }
        pthread_cond_wait(&q_not_full_, &q_mtx_);
    }
    if (!running_) {
        pthread_mutex_unlock(&q_mtx_);
        return false;
    }

    queue_.emplace_back(std::move(op));
    ++writes_enqueued_;

    pthread_cond_signal(&q_not_empty_);
    pthread_mutex_unlock(&q_mtx_);
    return true;
}

bool AsyncEngine::enqueue_delete(DeleteOp op) {
    pthread_mutex_lock(&q_mtx_);
    while (running_ && queue_.size() >= q_policy_.capacity) {
        if (!q_policy_.block_on_full) {
            pthread_mutex_unlock(&q_mtx_);
            return false;
        }
        pthread_cond_wait(&q_not_full_, &q_mtx_);
    }
    if (!running_) {
        pthread_mutex_unlock(&q_mtx_);
        return false;
    }

    queue_.emplace_back(std::move(op));
    ++writes_enqueued_;

    pthread_cond_signal(&q_not_empty_);
    pthread_mutex_unlock(&q_mtx_);
    return true;
}

// ==================== flush ====================

void AsyncEngine::flush() {
    pthread_mutex_lock(&q_mtx_);
    while (running_ &&
           (!queue_.empty() ||
            writes_applied_.load(std::memory_order_relaxed) <
            writes_enqueued_.load(std::memory_order_relaxed))) {
        // use not_full_ as a generic wakeup
        pthread_cond_wait(&q_not_full_, &q_mtx_);
    }
    pthread_mutex_unlock(&q_mtx_);
}

// ==================== search (sync) ====================

void AsyncEngine::search(int index_id,
                         const float* queries, size_t q_rows, int k, int nprobe,
                         std::vector<std::vector<DocId>>& out_ids,
                         std::vector<std::vector<float>>& out_scores) const {
    // take snapshot of that index
    std::shared_ptr<IVFIndex> idx;
    pthread_rwlock_rdlock(&indices_rwlock_);
    auto it = indices_.find(index_id);
    if (it != indices_.end()) {
        idx = it->second;
    }
    pthread_rwlock_unlock(&indices_rwlock_);

    if (!idx) {
        out_ids.assign(q_rows, {});
        out_scores.assign(q_rows, {});
        return;
    }

    int use_nprobe = (nprobe > 0) ? nprobe : search_policy_.default_nprobe;
    idx->search_nprobe(queries, q_rows, k, use_nprobe, out_ids, out_scores);

    // printf("Search on index %d done.\n", index_id);
}

void AsyncEngine::search(int index_id,
                         const float* queries, size_t q_rows, int k,
                         std::vector<std::vector<DocId>>& out_ids,
                         std::vector<std::vector<float>>& out_scores) const {
    search(index_id, queries, q_rows, k, /*nprobe=*/-1, out_ids, out_scores);
}

void AsyncEngine::load_cluster(int index_id,
                               int cluster_id,
                               const std::vector<DocId>& ids,
                               const std::vector<float>& vecs) {
    std::shared_ptr<IVFIndex> idx;
    pthread_rwlock_rdlock(&indices_rwlock_);
    auto it = indices_.find(index_id);
    if (it != indices_.end()) {
        idx = it->second;
    }
    pthread_rwlock_unlock(&indices_rwlock_);

    if (!idx) {
        throw std::out_of_range("AsyncEngine::load_cluster: index not found");
    }

    const size_t n_rows = ids.size();
    const size_t expected = static_cast<size_t>(idx->dim()) * n_rows;
    if (vecs.size() != expected) {
        throw std::invalid_argument("AsyncEngine::load_cluster: vecs.size mismatch");
    }

    idx->rebuild_cluster(cluster_id, ids.data(), vecs.data(), n_rows);
}

bool AsyncEngine::enqueue_insert_auto(int index_id,
                                      const std::vector<DocId>& ids,
                                      const std::vector<float>& vecs) {
    std::shared_ptr<IVFIndex> idx;
    pthread_rwlock_rdlock(&indices_rwlock_);
    auto it = indices_.find(index_id);
    if (it != indices_.end()) {
        idx = it->second;
    }
    pthread_rwlock_unlock(&indices_rwlock_);

    if (!idx) {
        throw std::out_of_range("AsyncEngine::enqueue_insert_auto: index not found");
    }

    const size_t n_rows = ids.size();
    if (n_rows == 0) return true;
    const int dim = idx->dim();
    if (vecs.size() != n_rows * (size_t)dim) {
        throw std::invalid_argument("AsyncEngine::enqueue_insert_auto: vecs size mismatch");
    }

    std::vector<int> assignments;
    idx->nearest_clusters(vecs.data(), n_rows, assignments);

    std::unordered_map<int, InsertOp> per_cluster;
    per_cluster.reserve(assignments.size());

    for (size_t i = 0; i < n_rows; ++i) {
        int cid = assignments[i];
        if (cid < 0) {
            throw std::runtime_error("AsyncEngine::enqueue_insert_auto: unable to assign cluster");
        }
        auto [it_op, inserted] = per_cluster.emplace(cid, InsertOp{});
        InsertOp& op = it_op->second;
        if (inserted) {
            op.index_id = index_id;
            op.cluster_id = cid;
        }
        op.ids.push_back(ids[i]);
        const float* row = vecs.data() + i * (size_t)dim;
        op.vecs.insert(op.vecs.end(), row, row + dim);
    }

    bool ok = true;
    for (auto& kv : per_cluster) {
        if (!enqueue_insert(std::move(kv.second))) {
            ok = false;
        }
    }
    return ok;
}

// ==================== info ====================

int AsyncEngine::dim_of(int index_id) const {
    std::shared_ptr<IVFIndex> idx;
    pthread_rwlock_rdlock(&indices_rwlock_);
    auto it = indices_.find(index_id);
    if (it != indices_.end()) idx = it->second;
    pthread_rwlock_unlock(&indices_rwlock_);
    return idx ? idx->dim() : 0;
}

Metric AsyncEngine::metric_of(int index_id) const {
    std::shared_ptr<IVFIndex> idx;
    pthread_rwlock_rdlock(&indices_rwlock_);
    auto it = indices_.find(index_id);
    if (it != indices_.end()) idx = it->second;
    pthread_rwlock_unlock(&indices_rwlock_);
    return idx ? idx->metric() : Metric::COSINE;
}

bool AsyncEngine::normalized_of(int index_id) const {
    std::shared_ptr<IVFIndex> idx;
    pthread_rwlock_rdlock(&indices_rwlock_);
    auto it = indices_.find(index_id);
    if (it != indices_.end()) idx = it->second;
    pthread_rwlock_unlock(&indices_rwlock_);
    return idx ? idx->normalized() : true;
}

int AsyncEngine::nlist_of(int index_id) const {
    std::shared_ptr<IVFIndex> idx;
    pthread_rwlock_rdlock(&indices_rwlock_);
    auto it = indices_.find(index_id);
    if (it != indices_.end()) idx = it->second;
    pthread_rwlock_unlock(&indices_rwlock_);
    return idx ? idx->nlist() : 0;
}

// ==================== apply batch (no async-layer lock) ====================

void AsyncEngine::apply_batch_noindexlock(
    const std::vector<WriteOp>& batch,
    const std::unordered_map<int, std::shared_ptr<IVFIndex>>& indices_snap) {

    for (const auto& op : batch) {
        if (std::holds_alternative<InsertOp>(op)) {
            const auto& ins = std::get<InsertOp>(op);
            auto it = indices_snap.find(ins.index_id);
            if (it == indices_snap.end() || !it->second) continue;
            it->second->add_batch(ins.cluster_id,
                                  ins.ids.data(),
                                  ins.vecs.data(),
                                  ins.ids.size());
            ++writes_applied_;
        } else if (std::holds_alternative<UpdateOp>(op)) {
            const auto& up = std::get<UpdateOp>(op);
            auto it = indices_snap.find(up.index_id);
            if (it == indices_snap.end() || !it->second) continue;
            it->second->update_batch(up.cluster_id,
                                     up.ids.data(),
                                     up.vecs.data(),
                                     up.ids.size(),
                                     up.insert_if_absent);
            ++writes_applied_;
        } else {
            const auto& del = std::get<DeleteOp>(op);
            auto it = indices_snap.find(del.index_id);
            if (it == indices_snap.end() || !it->second) continue;
            it->second->erase_batch(del.cluster_id,
                                    del.ids.data(),
                                    del.ids.size());
            ++writes_applied_;
        }
    }
}

// ==================== writer threads ====================

void* AsyncEngine::writer_main(void* arg) {
    auto* self = static_cast<AsyncEngine*>(arg);
    self->writer_loop();
    return nullptr;
}

void AsyncEngine::writer_loop() {
    std::vector<WriteOp> batch;
    batch.reserve(q_policy_.pop_batch_max);

    while (running_) {
        batch.clear();
        if (!pop_batch(batch, q_policy_.pop_batch_max)) {
            continue;
        }

        // snapshot all indices once
        std::unordered_map<int, std::shared_ptr<IVFIndex>> indices_snap;
        pthread_rwlock_rdlock(&indices_rwlock_);
        indices_snap = indices_;
        pthread_rwlock_unlock(&indices_rwlock_);

        // apply to the right index without holding async-layer lock
        apply_batch_noindexlock(batch, indices_snap);

        // notify flush-waiters
        pthread_mutex_lock(&q_mtx_);
        pthread_cond_broadcast(&q_not_full_);
        pthread_mutex_unlock(&q_mtx_);
    }
}

// ==================== maintenance threads ====================

void* AsyncEngine::maintenance_main(void* arg) {
    auto* self = static_cast<AsyncEngine*>(arg);
    self->maintenance_loop();
    return nullptr;
}

void AsyncEngine::maintenance_loop() {
    using namespace std::chrono;
    const auto period = duration<double>(
        maint_policy_.period_sec > 0.0 ? maint_policy_.period_sec : 1.0);

    while (running_) {
        std::this_thread::sleep_for(period);
        if (!running_) break;
        run_maintenance_once();
    }
}

// ==================== queue helpers ====================

bool AsyncEngine::pop_batch(std::vector<WriteOp>& batch, size_t max_n) {
    pthread_mutex_lock(&q_mtx_);
    while (queue_.empty() && running_) {
        pthread_cond_wait(&q_not_empty_, &q_mtx_);
    }
    if (!running_) {
        pthread_mutex_unlock(&q_mtx_);
        return false;
    }

    size_t n = std::min(max_n, queue_.size());
    for (size_t i = 0; i < n; ++i) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
    }

    // space is freed
    pthread_cond_broadcast(&q_not_full_);
    pthread_mutex_unlock(&q_mtx_);
    return !batch.empty();
}

// ==================== maintenance helper ====================

void AsyncEngine::run_maintenance_once() {
    // snapshot all indices
    std::unordered_map<int, std::shared_ptr<IVFIndex>> indices_snap;
    pthread_rwlock_rdlock(&indices_rwlock_);
    indices_snap = indices_;
    pthread_rwlock_unlock(&indices_rwlock_);

    for (auto& kv : indices_snap) {
        if (kv.second) {
            kv.second->maintenance_pass();
        }
    }
}

} // namespace m3
