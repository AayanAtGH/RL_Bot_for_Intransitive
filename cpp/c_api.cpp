#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "bitboard.hpp"
#include "board.hpp"
#include "mcts_core.hpp"
#include "tactical_prover.hpp"
#include "tb_index.hpp"
#include "tb_probe.hpp"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

using namespace intransitive;

namespace {

inline FastBoard make_board(const int8_t* grid, int side, int quiet, int ply) {
    FastBoard b;
    b.from_array(grid, static_cast<int8_t>(side), static_cast<uint8_t>(std::clamp(quiet, 0, 255)), static_cast<uint16_t>(std::max(0, ply)));
    return b;
}

struct SingleMCTSHandle {
    MCTSTree tree;
    std::vector<int32_t> pending_leaf_idx;
    std::vector<FastBoard> pending_leaf_board;
    int pending_sims_done = 0;

    explicit SingleMCTSHandle(const tb::TablebaseManager* tb, uint64_t seed)
        : tree(tb, nullptr, seed) {}
};

}  // namespace

extern "C" {

// ---- Board & Verification Helpers ----

EXPORT int32_t core_legal_actions(
    const int8_t* grid, int32_t side, int32_t quiet, uint16_t* out_actions
) {
    FastBoard b = make_board(grid, side, quiet, 0);
    return b.gen_legal_actions(out_actions);
}

EXPORT int32_t core_terminal_result(const int8_t* grid, int32_t side, int32_t quiet) {
    FastBoard b = make_board(grid, side, quiet, 0);
    return b.terminal_result();
}

EXPORT void core_encode_features_11ch(
    const int8_t* grid, int32_t side, int32_t quiet, float* out_obs
) {
    FastBoard b = make_board(grid, side, quiet, 0);
    b.encode_features_11ch(out_obs);
}

EXPORT int32_t core_prove_corner_win_3ply(const int8_t* grid, int32_t side, int32_t quiet) {
    FastBoard b = make_board(grid, side, quiet, 0);
    if (b.terminal_result() != 0) return -1;
    int w1 = find_immediate_win_action(b);
    if (w1 >= 0) return w1;
    return prove_corner_win_3ply(b);
}

// ---- Tablebase API ----

EXPORT void* tb_create_and_load(const char* dir_path, int32_t* out_max_pieces) {
    auto* tb = new tb::TablebaseManager();
    if (!tb->load(dir_path ? dir_path : "")) {
        delete tb;
        if (out_max_pieces) *out_max_pieces = 0;
        return nullptr;
    }
    if (out_max_pieces) *out_max_pieces = tb->max_pieces();
    return tb;
}

EXPORT void tb_destroy(void* handle) {
    delete static_cast<tb::TablebaseManager*>(handle);
}

EXPORT int32_t tb_probe_state(
    void* handle,
    const int8_t* grid,
    int32_t side,
    int32_t quiet,
    int32_t ply,
    int32_t* out_wdl,
    int32_t* out_dtz,
    int32_t* out_raw_wdl
) {
    auto* tb = static_cast<tb::TablebaseManager*>(handle);
    if (!tb) return 0;
    FastBoard b = make_board(grid, side, quiet, ply);
    tb::TBProbeResult r = tb->probe(b);
    if (!r.available) return 0;
    if (out_wdl) *out_wdl = r.wdl;
    if (out_dtz) *out_dtz = r.dtz;
    if (out_raw_wdl) *out_raw_wdl = r.raw_wdl;
    return 1;
}

EXPORT int32_t tb_probe_best_action(
    void* handle, const int8_t* grid, int32_t side, int32_t quiet, int32_t ply
) {
    auto* tb = static_cast<tb::TablebaseManager*>(handle);
    if (!tb) return -1;
    FastBoard b = make_board(grid, side, quiet, ply);
    return tb->probe_best_action(b);
}

// ---- Single-Game MCTS API ----

EXPORT void* mcts_create(
    void* tb_handle,
    float c_puct,
    float fpu_reduction,
    float root_fpu_reduction,
    float dirichlet_alpha,
    float dirichlet_eps,
    int32_t use_tactical_prover,
    uint64_t seed
) {
    auto* tb = static_cast<tb::TablebaseManager*>(tb_handle);
    auto* h = new SingleMCTSHandle(tb, seed);
    MCTSParams p;
    p.c_puct = c_puct;
    p.fpu_reduction = fpu_reduction;
    p.root_fpu_reduction = root_fpu_reduction;
    p.dirichlet_alpha = dirichlet_alpha;
    p.dirichlet_eps = dirichlet_eps;
    p.use_tactical_prover = (use_tactical_prover != 0);
    p.use_solver = true;
    p.use_tb = (tb != nullptr);
    h->tree.set_params(p);
    return h;
}

EXPORT void mcts_destroy(void* handle) {
    delete static_cast<SingleMCTSHandle*>(handle);
}

EXPORT void mcts_set_tablebase(void* handle, void* tb_handle) {
    auto* h = static_cast<SingleMCTSHandle*>(handle);
    if (!h) return;
    auto* tb = static_cast<tb::TablebaseManager*>(tb_handle);
    h->tree.set_tablebase(tb);
    MCTSParams p = h->tree.params();
    p.use_tb = (tb != nullptr);
    h->tree.set_params(p);
}

EXPORT int32_t mcts_start_search(
    void* handle,
    const int8_t* grid,
    int32_t side,
    int32_t quiet,
    int32_t ply,
    const uint64_t* history_hashes,
    int32_t num_history,
    int32_t try_reuse,
    int32_t add_noise
) {
    auto* h = static_cast<SingleMCTSHandle*>(handle);
    if (!h) return 0;
    FastBoard b = make_board(grid, side, quiet, ply);
    bool reused = false;
    if (try_reuse) {
        reused = h->tree.try_advance_to_board(b);
    }
    if (!reused) {
        std::vector<uint64_t> hist;
        if (history_hashes && num_history > 0) {
            hist.assign(history_hashes, history_hashes + num_history);
        }
        h->tree.reset(b, hist.empty() ? nullptr : &hist);
    }
    if (add_noise && h->tree.root_node().expanded) {
        h->tree.add_root_dirichlet_noise();
    }
    h->pending_leaf_idx.clear();
    h->pending_leaf_board.clear();
    return reused ? 1 : 0;
}

EXPORT void mcts_advance_root(void* handle, uint16_t canon_action) {
    auto* h = static_cast<SingleMCTSHandle*>(handle);
    if (h) h->tree.advance_root(canon_action);
}

EXPORT int32_t mcts_collect_batch(
    void* handle,
    int32_t max_batch,
    int32_t sims_remaining,
    int32_t add_noise_if_root_expanded,
    float* out_obs,
    uint8_t* out_masks,
    int32_t* out_sims_resolved_immediately
) {
    auto* h = static_cast<SingleMCTSHandle*>(handle);
    if (!h) return 0;
    h->pending_leaf_idx.clear();
    h->pending_leaf_board.clear();

    int resolved = 0;
    int target = std::min(max_batch, sims_remaining);
    const auto& t = tables();

    // Limit total consecutive cache/tactical resolutions per call so we can't spin forever
    int budget = std::max(target, sims_remaining);
    for (int step = 0; step < budget && static_cast<int>(h->pending_leaf_idx.size()) < target && resolved < sims_remaining; ++step) {
        if (h->tree.root_node().proven != PROVEN_UNKNOWN && h->tree.root_node().expanded) {
            break;
        }
        bool use_vl = (max_batch > 1);
        int32_t leaf_idx = -1;
        FastBoard leaf_board;
        bool needs_gpu = h->tree.select_or_resolve_leaf(use_vl, leaf_idx, leaf_board);
        if (!needs_gpu) {
            ++resolved;
            if (add_noise_if_root_expanded && h->tree.root_node().expanded && h->tree.root_visit_count() <= 2) {
                h->tree.add_root_dirichlet_noise();
            }
            continue;
        }

        // Deduplicate leaf_idx in same virtual-loss batch
        bool dup = false;
        for (int32_t existing : h->pending_leaf_idx) {
            if (existing == leaf_idx) { dup = true; break; }
        }
        if (dup) {
            if (use_vl) h->tree.revert_virtual_loss(leaf_idx);
            break;
        }

        int bpos = static_cast<int>(h->pending_leaf_idx.size());
        leaf_board.encode_features_11ch(out_obs + bpos * 11 * NUM_SQUARES);
        uint8_t* mask_ptr = out_masks + bpos * ACTION_SIZE;
        std::memset(mask_ptr, 0, ACTION_SIZE);
        uint16_t legal[80];
        int nl = leaf_board.gen_legal_actions(legal);
        for (int k = 0; k < nl; ++k) {
            uint16_t ca = (leaf_board.side == BLUE) ? legal[k] : t.rotate_action[legal[k]];
            mask_ptr[ca] = 1;
        }
        h->pending_leaf_idx.push_back(leaf_idx);
        h->pending_leaf_board.push_back(leaf_board);
    }

    if (out_sims_resolved_immediately) *out_sims_resolved_immediately = resolved;
    return static_cast<int32_t>(h->pending_leaf_idx.size());
}

EXPORT void mcts_apply_batch(
    void* handle,
    const float* probs,
    const float* vals,
    int32_t batch_count,
    int32_t used_vl,
    int32_t add_noise_if_root_expanded
) {
    auto* h = static_cast<SingleMCTSHandle*>(handle);
    if (!h) return;
    int n = std::min<int>(batch_count, static_cast<int>(h->pending_leaf_idx.size()));
    bool was_unexpanded = !h->tree.root_node().expanded;
    for (int i = 0; i < n; ++i) {
        h->tree.apply_leaf_eval(
            h->pending_leaf_idx[i],
            h->pending_leaf_board[i],
            probs + i * ACTION_SIZE,
            vals[i],
            used_vl != 0
        );
    }
    if (add_noise_if_root_expanded && was_unexpanded && h->tree.root_node().expanded) {
        h->tree.add_root_dirichlet_noise();
    }
    h->pending_leaf_idx.clear();
    h->pending_leaf_board.clear();
}

EXPORT void mcts_get_result(
    void* handle,
    float temperature,
    float* out_policy_648,
    double* out_root_q,
    int32_t* out_visit_count,
    int32_t* out_pv_depth,
    int32_t* out_proven,
    int32_t* out_top1_visits,
    int32_t* out_top2_visits
) {
    auto* h = static_cast<SingleMCTSHandle*>(handle);
    if (!h) return;
    int32_t t1 = 0, t2 = 0;
    h->tree.extract_root_policy(temperature, out_policy_648, &t1, &t2);
    const MCTSNode& r = h->tree.root_node();
    if (out_root_q) *out_root_q = r.q_value();
    if (out_visit_count) *out_visit_count = r.visit_count;
    if (out_pv_depth) *out_pv_depth = h->tree.root_pv_depth();
    if (out_proven) *out_proven = r.proven;
    if (out_top1_visits) *out_top1_visits = t1;
    if (out_top2_visits) *out_top2_visits = t2;
}

// ---- Batched Multi-Game Self-Play API ----

EXPORT void* sp_create(
    void* tb_handle,
    int32_t num_games,
    int32_t full_simulations,
    int32_t fast_simulations,
    float full_search_prob,
    int32_t temp_threshold_ply,
    float c_puct,
    float fpu_reduction,
    float root_fpu_reduction,
    float dirichlet_alpha,
    float dirichlet_eps,
    uint64_t seed
) {
    SelfPlayConfig cfg;
    cfg.num_games = num_games;
    cfg.full_simulations = full_simulations;
    cfg.fast_simulations = fast_simulations;
    cfg.full_search_prob = full_search_prob;
    cfg.temp_threshold_ply = temp_threshold_ply;
    cfg.c_puct = c_puct;
    cfg.fpu_reduction = fpu_reduction;
    cfg.root_fpu_reduction = root_fpu_reduction;
    cfg.dirichlet_alpha = dirichlet_alpha;
    cfg.dirichlet_eps = dirichlet_eps;
    cfg.seed = seed;
    auto* tb = static_cast<tb::TablebaseManager*>(tb_handle);
    return new BatchMCTSEngine(cfg, tb);
}

EXPORT void sp_destroy(void* handle) {
    delete static_cast<BatchMCTSEngine*>(handle);
}

EXPORT void sp_clear_cache(void* handle) {
    auto* sp = static_cast<BatchMCTSEngine*>(handle);
    if (sp) sp->clear_cache();
}

EXPORT int32_t sp_collect_batch(void* handle, float* out_obs, uint8_t* out_masks, int32_t max_batch) {
    auto* sp = static_cast<BatchMCTSEngine*>(handle);
    if (!sp) return 0;
    return sp->collect_batch(out_obs, out_masks, max_batch);
}

EXPORT void sp_apply_evals(void* handle, const float* probs, const float* vals, int32_t batch_count) {
    auto* sp = static_cast<BatchMCTSEngine*>(handle);
    if (sp) sp->apply_evals(probs, vals, batch_count);
}

EXPORT int32_t sp_finished_sample_count(void* handle) {
    auto* sp = static_cast<BatchMCTSEngine*>(handle);
    return sp ? static_cast<int32_t>(sp->finished_sample_count()) : 0;
}

EXPORT int32_t sp_pop_finished_samples(
    void* handle,
    int8_t* out_c_grids,
    uint8_t* out_quiets,
    uint16_t* out_pol_acts,
    float* out_pol_probs,
    uint8_t* out_pol_lens,
    uint8_t* out_wdl,
    uint8_t* out_train_pol,
    int32_t max_samples
) {
    auto* sp = static_cast<BatchMCTSEngine*>(handle);
    if (!sp) return 0;
    return sp->pop_finished_samples(
        out_c_grids, out_quiets, out_pol_acts, out_pol_probs, out_pol_lens, out_wdl, out_train_pol, max_samples
    );
}

EXPORT void sp_get_stats(void* handle, uint64_t* out_11_u64) {
    auto* sp = static_cast<BatchMCTSEngine*>(handle);
    if (!sp || !out_11_u64) return;
    SelfPlayStats s = sp->get_stats();
    out_11_u64[0] = s.games_finished;
    out_11_u64[1] = s.total_plies;
    out_11_u64[2] = s.blue_wins;
    out_11_u64[3] = s.red_wins;
    out_11_u64[4] = s.draws;
    out_11_u64[5] = s.tb_terminations;
    out_11_u64[6] = s.tb_node_hits;
    out_11_u64[7] = s.prover_hits;
    out_11_u64[8] = s.cache_hits;
    out_11_u64[9] = s.nn_evals;
    out_11_u64[10] = (s.full_searches > 0) ? (s.pv_depth_sum / s.full_searches) : 0;
}

}  // extern "C"
