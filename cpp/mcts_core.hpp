#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "board.hpp"
#include "tactical_prover.hpp"
#include "tb_probe.hpp"

namespace intransitive {

enum ProvenState : uint8_t {
    PROVEN_UNKNOWN = 0,
    PROVEN_WIN = 1,   // side-to-move at THIS node is guaranteed to win
    PROVEN_LOSS = 2,  // side-to-move at THIS node is guaranteed to lose
    PROVEN_DRAW = 3,  // guaranteed draw
};

struct MCTSNode {
    int32_t parent = -1;
    int32_t first_child = -1;
    uint16_t num_children = 0;
    uint16_t action = 0;  // canonical action (0..647) from parent to this node
    float prior = 0.0f;
    int32_t visit_count = 0;
    int32_t virtual_loss = 0;
    double value_sum = 0.0;
    uint8_t proven = PROVEN_UNKNOWN;
    bool expanded = false;

    inline double q_value() const {
        return (visit_count > 0) ? (value_sum / static_cast<double>(visit_count)) : 0.0;
    }
};

struct CacheEntry {
    uint64_t key = 0;
    int8_t c_sq[NUM_SQUARES]{};
    uint8_t q_bucket = 0;
    uint8_t num_legal = 0;
    uint16_t legal_c_acts[80]{};
    float legal_priors[80]{};
    float value = 0.0f;
    bool occupied = false;
};

class ZobristNNCache {
public:
    explicit ZobristNNCache(size_t pow2_size = 16384) {
        size_t sz = 1;
        while (sz < pow2_size) sz <<= 1;
        entries_.resize(sz);
        mask_ = sz - 1;
    }

    void clear() {
        for (auto& e : entries_) e.occupied = false;
    }

    bool lookup(const FastBoard& b, uint64_t chash, const CacheEntry*& out) const {
        const auto& e = entries_[chash & mask_];
        if (!e.occupied || e.key != chash) return false;
        uint8_t qb = (b.quiet >= 150) ? b.quiet : 0;
        if (e.q_bucket != qb) return false;

        FastBoard cb = b.canonical_copy();
        if (std::memcmp(e.c_sq, cb.sq, NUM_SQUARES) != 0) return false;
        out = &e;
        return true;
    }

    void insert(
        const FastBoard& b,
        uint64_t chash,
        const uint16_t* c_acts,
        const float* c_priors,
        uint8_t num_legal,
        float val
    ) {
        auto& e = entries_[chash & mask_];
        e.key = chash;
        FastBoard cb = b.canonical_copy();
        std::memcpy(e.c_sq, cb.sq, NUM_SQUARES);
        e.q_bucket = (b.quiet >= 150) ? b.quiet : 0;
        e.num_legal = num_legal;
        for (uint8_t i = 0; i < num_legal; ++i) {
            e.legal_c_acts[i] = c_acts[i];
            e.legal_priors[i] = c_priors[i];
        }
        e.value = val;
        e.occupied = true;
    }

private:
    std::vector<CacheEntry> entries_;
    size_t mask_ = 0;
};

struct MCTSParams {
    float c_puct = 1.5f;
    float fpu_reduction = 0.25f;
    float root_fpu_reduction = 0.05f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_eps = 0.25f;
    bool use_tactical_prover = true;
    bool use_solver = true;
    bool use_tb = true;
};

class MCTSTree {
public:
    explicit MCTSTree(
        const tb::TablebaseManager* tb = nullptr,
        ZobristNNCache* shared_cache = nullptr,
        uint64_t seed = 42
    )
        : tb_(tb), cache_(shared_cache), rng_(seed) {
        if (!cache_) {
            owned_cache_ = std::make_unique<ZobristNNCache>(16384);
            cache_ = owned_cache_.get();
        }
        nodes_.reserve(4096);
    }

    void set_params(const MCTSParams& p) { params_ = p; }
    const MCTSParams& params() const { return params_; }
    void set_tablebase(const tb::TablebaseManager* tb) { tb_ = tb; }

    void reset(const FastBoard& root_board, const std::vector<uint64_t>* history_hashes = nullptr) {
        nodes_.clear();
        nodes_.emplace_back();
        root_idx_ = 0;
        root_board_ = root_board;
        history_hashes_.clear();
        if (history_hashes) {
            history_hashes_ = *history_hashes;
        } else {
            history_hashes_.push_back(root_board.hash);
        }
        tb_hits_ = 0;
        prover_hits_ = 0;
        cache_hits_ = 0;
        nn_evals_ = 0;
    }

    const FastBoard& root_board() const { return root_board_; }
    int32_t root_idx() const { return root_idx_; }
    const MCTSNode& root_node() const { return nodes_[root_idx_]; }
    const MCTSNode& node(int32_t idx) const { return nodes_[idx]; }

    uint64_t tb_hits() const { return tb_hits_; }
    uint64_t prover_hits() const { return prover_hits_; }
    uint64_t cache_hits() const { return cache_hits_; }
    uint64_t nn_evals() const { return nn_evals_; }

    // Advance root to existing child for `canon_action`. Returns true if subtree was reused.
    bool advance_root(uint16_t canon_action) {
        if (root_idx_ < 0 || root_idx_ >= static_cast<int32_t>(nodes_.size())) return false;
        const MCTSNode& r = nodes_[root_idx_];
        if (!r.expanded || r.num_children == 0) return false;

        int32_t chosen = -1;
        for (int i = 0; i < r.num_children; ++i) {
            int32_t cidx = r.first_child + i;
            if (nodes_[cidx].action == canon_action) {
                chosen = cidx;
                break;
            }
        }
        if (chosen < 0 || !nodes_[chosen].expanded || nodes_[chosen].num_children == 0) {
            return false;
        }

        uint16_t real_act = (root_board_.side == BLUE)
                                ? canon_action
                                : tables().rotate_action[canon_action];
        root_board_.push_action(real_act);
        history_hashes_.push_back(root_board_.hash);
        nodes_[chosen].parent = -1;
        root_idx_ = chosen;

        // Compact arena if node pool grew large relative to active subtree
        if (nodes_.size() > 65536 && nodes_[root_idx_].visit_count * 3 < static_cast<int32_t>(nodes_.size())) {
            compact_arena();
        }
        return true;
    }

    // Check if `target_board` is 1 legal move away from current `root_board_` and reuse subtree if found.
    bool try_advance_to_board(const FastBoard& target_board) {
        if (root_idx_ < 0 || !nodes_[root_idx_].expanded) return false;
        const MCTSNode& r = nodes_[root_idx_];
        const auto& t = tables();
        for (int i = 0; i < r.num_children; ++i) {
            uint16_t c_act = nodes_[r.first_child + i].action;
            uint16_t r_act = (root_board_.side == BLUE) ? c_act : t.rotate_action[c_act];
            FastBoard tmp = root_board_;
            tmp.push_action(r_act);
            if (tmp.hash == target_board.hash &&
                tmp.side == target_board.side &&
                tmp.quiet == target_board.quiet &&
                std::memcmp(tmp.sq, target_board.sq, NUM_SQUARES) == 0) {
                return advance_root(c_act);
            }
        }
        return false;
    }

    // Add Dirichlet noise to root children priors (normalizes priors first)
    void add_root_dirichlet_noise() {
        MCTSNode& r = nodes_[root_idx_];
        if (!r.expanded || r.num_children == 0) return;

        std::gamma_distribution<double> gamma(params_.dirichlet_alpha, 1.0);
        std::vector<double> noise(r.num_children);
        double n_sum = 0.0;
        for (int i = 0; i < r.num_children; ++i) {
            double g = gamma(rng_);
            noise[i] = g;
            n_sum += g;
        }
        if (n_sum <= 0.0) return;

        float eps = params_.dirichlet_eps;
        for (int i = 0; i < r.num_children; ++i) {
            MCTSNode& c = nodes_[r.first_child + i];
            c.prior = (1.0f - eps) * c.prior + eps * static_cast<float>(noise[i] / n_sum);
        }
    }

    // Walk down the tree via PUCT from root, applying virtual loss if `apply_vl == true`.
    // Runs Pre-NN Tactical Filter / TB / Zobrist cache on unexpanded leaves immediately!
    // If leaf STILL needs neural-net eval, returns true and populates `out_board` + `out_leaf_idx`.
    // If leaf was resolved immediately (terminal, rep, TB, prover, cache), backs up immediately and returns false.
    bool select_or_resolve_leaf(bool apply_vl, int32_t& out_leaf_idx, FastBoard& out_board) {
        int32_t curr = root_idx_;
        FastBoard b = root_board_;
        uint64_t path_hashes[512];
        int path_len = 0;
        const auto& t = tables();

        while (nodes_[curr].expanded && nodes_[curr].num_children > 0) {
            // Record current board hash in search path (excluding root which is already in history_hashes_)
            if (curr != root_idx_ && path_len < 512) {
                path_hashes[path_len++] = b.hash;
            }
            int32_t child = select_child_puct(curr);
            if (apply_vl) {
                nodes_[child].visit_count += 1;
                nodes_[child].virtual_loss += 1;
                nodes_[child].value_sum += 1.0;  // penalty from parent's perspective (-q)
            }
            uint16_t c_act = nodes_[child].action;
            uint16_t r_act = (b.side == BLUE) ? c_act : t.rotate_action[c_act];
            b.push_action(r_act);
            curr = child;

            // Stop descending if this child is already proven terminal/TB/prover so we backprop proven value cleanly
            if (nodes_[curr].proven != PROVEN_UNKNOWN && !nodes_[curr].expanded) {
                break;
            }
        }

        // Already proven unexpanded leaf (e.g. suicidal move or proven terminal)
        if (nodes_[curr].proven != PROVEN_UNKNOWN) {
            if (apply_vl) revert_virtual_loss(curr);
            float val = proven_to_value(nodes_[curr].proven);
            backpropagate(curr, val);
            return false;
        }

        // Pre-NN Exact Tactical Filter:
        // 1. Exact terminal rules
        int term = b.terminal_result();
        if (term != 0) {
            if (apply_vl) revert_virtual_loss(curr);
            uint8_t pstate = PROVEN_DRAW;
            float val = 0.0f;
            if (term != 2) {
                bool win_for_side = (term == b.side);
                pstate = win_for_side ? PROVEN_WIN : PROVEN_LOSS;
                val = win_for_side ? 1.0f : -1.0f;
            }
            nodes_[curr].expanded = true;
            nodes_[curr].proven = pstate;
            backpropagate(curr, val);
            return false;
        }

        // 2. Repetition check inside search path / game history (skip root itself!)
        if (curr != root_idx_) {
            int matches = 0;
            for (int i = 0; i < path_len; ++i) {
                if (path_hashes[i] == b.hash) { matches = 1; break; }
            }
            if (matches == 0) {
                for (uint64_t hh : history_hashes_) {
                    if (hh == b.hash) { matches = 1; break; }
                }
            }
            if (matches > 0) {
                if (apply_vl) revert_virtual_loss(curr);
                nodes_[curr].expanded = true;
                nodes_[curr].proven = PROVEN_DRAW;
                backpropagate(curr, 0.0f);
                return false;
            }
        }

        // 3. Up-to-4-piece Endgame Tablebase probe (skips NN on internal nodes; root expands children cleanly)
        if (curr != root_idx_ && params_.use_tb && tb_ && tb_->is_loaded()) {
            int tb_wdl = tb_->probe_wdl_fast(b);
            if (tb_wdl != -99) {
                if (apply_vl) revert_virtual_loss(curr);
                nodes_[curr].expanded = true;
                nodes_[curr].proven = (tb_wdl > 0) ? PROVEN_WIN : ((tb_wdl < 0) ? PROVEN_LOSS : PROVEN_DRAW);
                ++tb_hits_;
                backpropagate(curr, static_cast<float>(tb_wdl));
                return false;
            }
        }

        // 4. Pre-NN 1-ply immediate win & 3-ply corner-sprint prover
        if (params_.use_tactical_prover) {
            int win_act = find_immediate_win_action(b);
            if (win_act < 0) {
                win_act = prove_corner_win_3ply(b);
            }
            if (win_act >= 0) {
                if (apply_vl) revert_virtual_loss(curr);
                uint16_t canon_win = (b.side == BLUE) ? static_cast<uint16_t>(win_act)
                                                      : t.rotate_action[win_act];
                expand_proven_win(curr, b, canon_win);
                ++prover_hits_;
                backpropagate(curr, 1.0f);
                return false;
            }
        }

        // 5. Zobrist NN Evaluation Cache lookup
        if (cache_) {
            uint64_t chash = b.canonical_hash();
            const CacheEntry* hit = nullptr;
            if (cache_->lookup(b, chash, hit)) {
                if (apply_vl) revert_virtual_loss(curr);
                float val = expand_from_legal_priors(curr, b, hit->legal_c_acts, hit->legal_priors, hit->num_legal, hit->value);
                ++cache_hits_;
                backpropagate(curr, val);
                return false;
            }
        }

        // Cache miss: queue leaf for GPU evaluation
        out_leaf_idx = curr;
        out_board = b;
        return true;
    }

    // Apply GPU NN output `(probs_648, scalar_val)` to a pending leaf `leaf_idx` at `leaf_board`
    void apply_leaf_eval(int32_t leaf_idx, const FastBoard& leaf_board, const float* probs_648, float val, bool revert_vl) {
        if (revert_vl) revert_virtual_loss(leaf_idx);

        uint16_t real_acts[80];
        int n = leaf_board.gen_legal_actions(real_acts);
        if (n == 0) {
            nodes_[leaf_idx].expanded = true;
            nodes_[leaf_idx].proven = PROVEN_LOSS;
            backpropagate(leaf_idx, -1.0f);
            return;
        }

        const auto& t = tables();
        uint16_t c_acts[80];
        float c_priors[80];
        float p_sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            uint16_t ca = (leaf_board.side == BLUE) ? real_acts[i] : t.rotate_action[real_acts[i]];
            c_acts[i] = ca;
            float pr = std::max(0.0f, probs_648[ca]);
            c_priors[i] = pr;
            p_sum += pr;
        }
        if (p_sum > 1e-8f) {
            for (int i = 0; i < n; ++i) c_priors[i] /= p_sum;
        } else {
            float uni = 1.0f / static_cast<float>(n);
            for (int i = 0; i < n; ++i) c_priors[i] = uni;
        }

        if (cache_) {
            cache_->insert(leaf_board, leaf_board.canonical_hash(), c_acts, c_priors, static_cast<uint8_t>(n), val);
        }
        ++nn_evals_;

        float final_val = expand_from_legal_priors(leaf_idx, leaf_board, c_acts, c_priors, n, val);
        backpropagate(leaf_idx, final_val);
    }

    void revert_virtual_loss(int32_t leaf_idx) {
        int32_t curr = leaf_idx;
        while (curr != -1 && curr != root_idx_) {
            if (nodes_[curr].virtual_loss > 0) {
                nodes_[curr].visit_count -= 1;
                nodes_[curr].virtual_loss -= 1;
                nodes_[curr].value_sum -= 1.0;
            }
            curr = nodes_[curr].parent;
        }
    }

    int root_pv_depth() const {
        int depth = 0;
        int32_t curr = root_idx_;
        while (curr >= 0 && nodes_[curr].expanded && nodes_[curr].num_children > 0) {
            int32_t best_c = -1;
            int32_t best_v = 0;
            const MCTSNode& node = nodes_[curr];
            for (int i = 0; i < node.num_children; ++i) {
                int32_t ci = node.first_child + i;
                if (nodes_[ci].visit_count > best_v) {
                    best_v = nodes_[ci].visit_count;
                    best_c = ci;
                }
            }
            if (best_c < 0) break;
            ++depth;
            curr = best_c;
        }
        return depth;
    }

    int32_t root_visit_count() const {
        return (root_idx_ >= 0) ? nodes_[root_idx_].visit_count : 0;
    }

    // Extract canonical (648,) policy distribution from root visits, plus top-2 visit counts
    void extract_root_policy(float temperature, float* out_policy_648, int32_t* out_top1_visits = nullptr, int32_t* out_top2_visits = nullptr) const {
        std::memset(out_policy_648, 0, ACTION_SIZE * sizeof(float));
        if (root_idx_ < 0) return;
        const MCTSNode& r = nodes_[root_idx_];
        if (r.num_children == 0) return;

        // If root has a PROVEN_WIN child (PROVEN_LOSS from child's side), put 100% mass on proven winning move!
        int32_t proven_win_c = -1;
        int32_t best_visits = -1;
        int32_t second_visits = 0;
        int32_t best_c = -1;

        for (int i = 0; i < r.num_children; ++i) {
            const MCTSNode& c = nodes_[r.first_child + i];
            if (c.proven == PROVEN_LOSS) {
                if (proven_win_c < 0 || c.visit_count > nodes_[proven_win_c].visit_count) {
                    proven_win_c = r.first_child + i;
                }
            }
            int32_t v = c.visit_count;
            if (v > best_visits) {
                second_visits = std::max(0, best_visits);
                best_visits = v;
                best_c = r.first_child + i;
            } else if (v > second_visits) {
                second_visits = v;
            }
        }
        if (out_top1_visits) *out_top1_visits = std::max(0, best_visits);
        if (out_top2_visits) *out_top2_visits = std::max(0, second_visits);

        if (proven_win_c >= 0) {
            out_policy_648[nodes_[proven_win_c].action] = 1.0f;
            return;
        }

        if (temperature <= 1e-3f) {
            if (best_c >= 0) {
                // Tie-break by -q_value if visit_count tied
                double best_q = -1e9;
                int32_t tie_c = best_c;
                for (int i = 0; i < r.num_children; ++i) {
                    const MCTSNode& c = nodes_[r.first_child + i];
                    if (c.visit_count == best_visits) {
                        double q = (c.visit_count > 0) ? (-c.q_value()) : static_cast<double>(c.prior);
                        if (q > best_q) {
                            best_q = q;
                            tie_c = r.first_child + i;
                        }
                    }
                }
                out_policy_648[nodes_[tie_c].action] = 1.0f;
            }
            return;
        }

        // Prefer non-PROVEN_WIN children if any exist
        bool any_non_losing = false;
        for (int i = 0; i < r.num_children; ++i) {
            if (nodes_[r.first_child + i].proven != PROVEN_WIN) {
                any_non_losing = true;
                break;
            }
        }

        double inv_t = 1.0 / static_cast<double>(temperature);
        double sum = 0.0;
        double weights[80];
        for (int i = 0; i < r.num_children; ++i) {
            const MCTSNode& c = nodes_[r.first_child + i];
            if (any_non_losing && c.proven == PROVEN_WIN) {
                weights[i] = 0.0;
                continue;
            }
            double w = (c.visit_count > 0) ? std::pow(static_cast<double>(c.visit_count), inv_t) : 0.0;
            weights[i] = w;
            sum += w;
        }

        if (sum > 0.0) {
            for (int i = 0; i < r.num_children; ++i) {
                out_policy_648[nodes_[r.first_child + i].action] = static_cast<float>(weights[i] / sum);
            }
        } else if (best_c >= 0) {
            out_policy_648[nodes_[best_c].action] = 1.0f;
        }
    }

private:
    const tb::TablebaseManager* tb_ = nullptr;
    ZobristNNCache* cache_ = nullptr;
    std::unique_ptr<ZobristNNCache> owned_cache_;
    MCTSParams params_;
    std::mt19937_64 rng_;

    std::vector<MCTSNode> nodes_;
    int32_t root_idx_ = -1;
    FastBoard root_board_{};
    std::vector<uint64_t> history_hashes_;

    uint64_t tb_hits_ = 0;
    uint64_t prover_hits_ = 0;
    uint64_t cache_hits_ = 0;
    uint64_t nn_evals_ = 0;

    static inline float proven_to_value(uint8_t p) {
        if (p == PROVEN_WIN) return 1.0f;
        if (p == PROVEN_LOSS) return -1.0f;
        return 0.0f;
    }

    int32_t select_child_puct(int32_t node_idx) const {
        const MCTSNode& node = nodes_[node_idx];
        // 1. If any child is PROVEN_LOSS (for child = WIN for parent!), pick immediately!
        int32_t best_proven_win = -1;
        for (int i = 0; i < node.num_children; ++i) {
            int32_t ci = node.first_child + i;
            if (nodes_[ci].proven == PROVEN_LOSS) {
                if (best_proven_win < 0 || nodes_[ci].prior > nodes_[best_proven_win].prior) {
                    best_proven_win = ci;
                }
            }
        }
        if (best_proven_win >= 0) return best_proven_win;

        bool any_non_losing = false;
        double visited_prior_sum = 0.0;
        for (int i = 0; i < node.num_children; ++i) {
            const MCTSNode& c = nodes_[node.first_child + i];
            if (c.proven != PROVEN_WIN) any_non_losing = true;
            if (c.visit_count > 0) visited_prior_sum += static_cast<double>(c.prior);
        }

        double parent_q = (node.visit_count > 0) ? node.q_value() : 0.0;
        float fpu_red = (node_idx == root_idx_) ? params_.root_fpu_reduction : params_.fpu_reduction;
        double fpu_val = std::clamp(
            parent_q - static_cast<double>(fpu_red) * std::sqrt(std::max(0.0, visited_prior_sum)),
            -1.0,
            1.0
        );
        double sqrt_total = std::sqrt(static_cast<double>(std::max(1, node.visit_count)));

        double best_score = -1e18;
        int32_t best_child = node.first_child;
        for (int i = 0; i < node.num_children; ++i) {
            int32_t ci = node.first_child + i;
            const MCTSNode& c = nodes_[ci];
            if (any_non_losing && c.proven == PROVEN_WIN) continue;

            double q_val = (c.visit_count > 0) ? (-c.q_value()) : fpu_val;
            double u_val = static_cast<double>(params_.c_puct) * static_cast<double>(c.prior) *
                           (sqrt_total / static_cast<double>(1 + c.visit_count));
            double score = q_val + u_val;
            if (score > best_score) {
                best_score = score;
                best_child = ci;
            }
        }
        return best_child;
    }

    void expand_proven_win(int32_t node_idx, const FastBoard& b, uint16_t win_c_act) {
        uint16_t real_acts[80];
        int n = b.gen_legal_actions(real_acts);
        if (n == 0) {
            nodes_[node_idx].expanded = true;
            nodes_[node_idx].proven = PROVEN_LOSS;
            return;
        }
        const auto& t = tables();
        int32_t first_c = static_cast<int32_t>(nodes_.size());
        nodes_.resize(first_c + n);

        for (int i = 0; i < n; ++i) {
            uint16_t ca = (b.side == BLUE) ? real_acts[i] : t.rotate_action[real_acts[i]];
            MCTSNode& cn = nodes_[first_c + i];
            cn.parent = node_idx;
            cn.action = ca;
            cn.prior = (ca == win_c_act) ? 1.0f : 1e-6f;
            if (ca == win_c_act) {
                cn.proven = PROVEN_LOSS;  // child loses -> parent wins!
                cn.visit_count = 1;
                cn.value_sum = -1.0;
            }
        }
        nodes_[node_idx].first_child = first_c;
        nodes_[node_idx].num_children = static_cast<uint16_t>(n);
        nodes_[node_idx].expanded = true;
        nodes_[node_idx].proven = PROVEN_WIN;
    }

    float expand_from_legal_priors(
        int32_t node_idx,
        const FastBoard& b,
        const uint16_t* c_acts,
        const float* c_priors,
        int n,
        float net_val
    ) {
        if (n == 0) {
            nodes_[node_idx].expanded = true;
            nodes_[node_idx].proven = PROVEN_LOSS;
            return -1.0f;
        }

        const auto& t = tables();
        bool suicidal[80]{};
        int num_safe = 0;
        float safe_prior_sum = 0.0f;

        FastBoard work = b;
        for (int i = 0; i < n; ++i) {
            uint16_t ra = (b.side == BLUE) ? c_acts[i] : t.rotate_action[c_acts[i]];
            UndoInfo u = work.push_action(ra);
            bool opp_1ply_win = has_immediate_corner_win(work);
            work.pop_action(ra, u);
            suicidal[i] = opp_1ply_win;
            if (!opp_1ply_win) {
                ++num_safe;
                safe_prior_sum += c_priors[i];
            }
        }

        int32_t first_c = static_cast<int32_t>(nodes_.size());
        nodes_.resize(first_c + n);

        for (int i = 0; i < n; ++i) {
            MCTSNode& cn = nodes_[first_c + i];
            cn.parent = node_idx;
            cn.action = c_acts[i];
            if (suicidal[i]) {
                cn.prior = 1e-6f;
                cn.proven = PROVEN_WIN;  // opponent (child side) wins in 1 ply!
                cn.visit_count = 1;
                cn.value_sum = 1.0;
            } else {
                if (num_safe < n && safe_prior_sum > 1e-8f) {
                    cn.prior = c_priors[i] / safe_prior_sum;
                } else if (num_safe < n) {
                    cn.prior = 1.0f / static_cast<float>(num_safe);
                } else {
                    cn.prior = c_priors[i];
                }
            }
        }

        nodes_[node_idx].first_child = first_c;
        nodes_[node_idx].num_children = static_cast<uint16_t>(n);
        nodes_[node_idx].expanded = true;

        if (num_safe == 0) {
            nodes_[node_idx].proven = PROVEN_LOSS;
            return -1.0f;
        }
        return net_val;
    }

    void backpropagate(int32_t leaf_idx, float val) {
        int32_t curr = leaf_idx;
        double v = static_cast<double>(val);
        while (curr != -1) {
            MCTSNode& node = nodes_[curr];
            node.visit_count += 1;
            node.value_sum += v;

            // MCTS-Solver upward proof propagation
            if (params_.use_solver && node.expanded && node.num_children > 0 && node.proven == PROVEN_UNKNOWN) {
                bool any_child_loss = false;
                bool all_child_win = true;
                for (int i = 0; i < node.num_children; ++i) {
                    uint8_t cp = nodes_[node.first_child + i].proven;
                    if (cp == PROVEN_LOSS) {
                        any_child_loss = true;
                        break;
                    }
                    if (cp != PROVEN_WIN) {
                        all_child_win = false;
                    }
                }
                if (any_child_loss) {
                    node.proven = PROVEN_WIN;
                    v = 1.0;
                } else if (all_child_win) {
                    node.proven = PROVEN_LOSS;
                    v = -1.0;
                }
            }

            v = -v;
            curr = node.parent;
        }
    }

    void compact_arena() {
        std::vector<MCTSNode> new_nodes;
        new_nodes.reserve(std::max<size_t>(1024, nodes_[root_idx_].visit_count + 128));

        auto copy_subtree = [&](auto& self, int32_t old_idx, int32_t new_parent) -> int32_t {
            int32_t new_idx = static_cast<int32_t>(new_nodes.size());
            new_nodes.push_back(nodes_[old_idx]);
            new_nodes[new_idx].parent = new_parent;
            new_nodes[new_idx].first_child = -1;
            return new_idx;
        };
        (void)copy_subtree;

        // BFS copy preserves sibling contiguity (`first_child .. first_child + num_children - 1`)!
        new_nodes.push_back(nodes_[root_idx_]);
        new_nodes[0].parent = -1;
        std::vector<int32_t> old_indices;
        old_indices.reserve(4096);
        old_indices.push_back(root_idx_);

        size_t q_head = 0;
        while (q_head < new_nodes.size()) {
            int32_t cur_new = static_cast<int32_t>(q_head);
            int32_t cur_old = old_indices[q_head];
            ++q_head;

            uint16_t nc = nodes_[cur_old].num_children;
            int32_t old_fc = nodes_[cur_old].first_child;
            if (nodes_[cur_old].expanded && nc > 0 && old_fc >= 0) {
                int32_t new_fc = static_cast<int32_t>(new_nodes.size());
                new_nodes[cur_new].first_child = new_fc;
                for (uint16_t i = 0; i < nc; ++i) {
                    new_nodes.push_back(nodes_[old_fc + i]);
                    new_nodes.back().parent = cur_new;
                    old_indices.push_back(old_fc + i);
                }
            } else {
                new_nodes[cur_new].first_child = -1;
                new_nodes[cur_new].num_children = 0;
            }
        }
        nodes_.swap(new_nodes);
        root_idx_ = 0;
    }
};

// Emitted sample struct matching compact Python ReplayBuffer
struct FinishedSample {
    int8_t c_grid[NUM_SQUARES];
    uint8_t quiet;
    uint8_t pol_len;
    uint8_t wdl_target;    // 0=LOSS, 1=DRAW, 2=WIN (from acting side's perspective)
    uint8_t train_policy;  // 1 if full-cap Playout Cap Randomization turn, else 0
    uint16_t pol_actions[80];
    float pol_probs[80];
};

struct SelfPlayConfig {
    int num_games = 128;
    int full_simulations = 320;
    int fast_simulations = 64;
    float full_search_prob = 0.25f;
    int temp_threshold_ply = 24;
    int hard_max_plies = 600;
    float c_puct = 1.5f;
    float fpu_reduction = 0.25f;
    float root_fpu_reduction = 0.05f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_eps = 0.25f;
    uint64_t seed = 42;
};

struct SelfPlayStats {
    uint64_t games_finished = 0;
    uint64_t total_plies = 0;
    uint64_t blue_wins = 0;
    uint64_t red_wins = 0;
    uint64_t draws = 0;
    uint64_t tb_terminations = 0;
    uint64_t tb_node_hits = 0;
    uint64_t prover_hits = 0;
    uint64_t cache_hits = 0;
    uint64_t nn_evals = 0;
    uint64_t pv_depth_sum = 0;
    uint64_t full_searches = 0;
};

class BatchMCTSEngine {
public:
    BatchMCTSEngine(const SelfPlayConfig& cfg, const tb::TablebaseManager* tb = nullptr)
        : cfg_(cfg),
          tb_(tb),
          cache_(65536),
          rng_(cfg.seed) {
        games_.reserve(cfg.num_games);
        for (int i = 0; i < cfg.num_games; ++i) {
            games_.emplace_back(tb_, &cache_, cfg.seed + 10007ULL * (i + 1));
            MCTSParams mp;
            mp.c_puct = cfg_.c_puct;
            mp.fpu_reduction = cfg_.fpu_reduction;
            mp.root_fpu_reduction = cfg_.root_fpu_reduction;
            mp.dirichlet_alpha = cfg_.dirichlet_alpha;
            mp.dirichlet_eps = cfg_.dirichlet_eps;
            mp.use_tactical_prover = true;
            mp.use_solver = true;
            mp.use_tb = (tb_ != nullptr && tb_->is_loaded());
            games_.back().tree.set_params(mp);
            init_new_game(i);
        }
    }

    void clear_cache() { cache_.clear(); }

    // Collect up to `max_batch` pending leaf evaluations across all parallel games.
    // Returns number of leaves `B` written into `out_obs` (B*11*81 floats) and `out_masks` (B*648 uint8).
    int collect_batch(float* out_obs, uint8_t* out_masks, int max_batch) {
        pending_.clear();
        int n_games = static_cast<int>(games_.size());
        if (n_games == 0 || max_batch <= 0) return 0;

        int rounds = 0;
        const int max_rounds = n_games * 4;
        while (static_cast<int>(pending_.size()) < max_batch && rounds < max_rounds) {
            int gidx = rr_cursor_ % n_games;
            rr_cursor_++;
            rounds++;

            GameSlot& g = games_[gidx];
            if (g.waiting_on_gpu) continue;

            // Advance sims inside this game until either it finishes the turn OR hits a GPU cache miss
            for (int safety = 0; safety < 2048; ++safety) {
                if (g.tree.root_node().expanded &&
                    (g.sims_this_turn >= g.target_sims || g.tree.root_node().proven != PROVEN_UNKNOWN)) {
                    finish_turn_and_maybe_game(gidx);
                    continue;
                }

                // Ensure Dirichlet noise is added after root is expanded on full-search turns
                if (g.is_full_search && !g.noise_applied && g.tree.root_node().expanded) {
                    g.tree.add_root_dirichlet_noise();
                    g.noise_applied = true;
                }

                int32_t leaf_idx = -1;
                FastBoard leaf_board;
                bool needs_gpu = g.tree.select_or_resolve_leaf(false, leaf_idx, leaf_board);
                if (!needs_gpu) {
                    g.sims_this_turn++;
                    if (g.is_full_search && !g.noise_applied && g.tree.root_node().expanded) {
                        g.tree.add_root_dirichlet_noise();
                        g.noise_applied = true;
                    }
                    continue;
                }

                // Queue leaf for GPU
                int b_pos = static_cast<int>(pending_.size());
                leaf_board.encode_features_11ch(out_obs + b_pos * 11 * NUM_SQUARES);
                uint8_t* mask_ptr = out_masks + b_pos * ACTION_SIZE;
                std::memset(mask_ptr, 0, ACTION_SIZE);
                uint16_t legal[80];
                int nl = leaf_board.gen_legal_actions(legal);
                const auto& t = tables();
                for (int k = 0; k < nl; ++k) {
                    uint16_t ca = (leaf_board.side == BLUE) ? legal[k] : t.rotate_action[legal[k]];
                    mask_ptr[ca] = 1;
                }

                g.waiting_on_gpu = true;
                pending_.push_back({gidx, leaf_idx, leaf_board});
                break;
            }
        }
        return static_cast<int>(pending_.size());
    }

    // Apply GPU outputs (`probs`: B*648, `vals`: B) to all pending leaves and advance turns.
    void apply_evals(const float* probs, const float* vals, int batch_count) {
        int n = std::min<int>(batch_count, static_cast<int>(pending_.size()));
        for (int i = 0; i < n; ++i) {
            const PendingEval& pe = pending_[i];
            GameSlot& g = games_[pe.game_idx];
            g.tree.apply_leaf_eval(pe.leaf_idx, pe.board, probs + i * ACTION_SIZE, vals[i], false);
            g.waiting_on_gpu = false;
            g.sims_this_turn++;
            if (g.is_full_search && !g.noise_applied && g.tree.root_node().expanded) {
                g.tree.add_root_dirichlet_noise();
                g.noise_applied = true;
            }
            if (g.tree.root_node().expanded &&
                (g.sims_this_turn >= g.target_sims || g.tree.root_node().proven != PROVEN_UNKNOWN)) {
                finish_turn_and_maybe_game(pe.game_idx);
            }
        }
        pending_.clear();
    }

    size_t finished_sample_count() const { return finished_samples_.size(); }

    // Pop up to `max_samples` into caller-provided flat NumPy arrays. Returns count popped.
    int pop_finished_samples(
        int8_t* out_c_grids,     // max_samples * 81
        uint8_t* out_quiets,     // max_samples
        uint16_t* out_pol_acts,  // max_samples * 80
        float* out_pol_probs,    // max_samples * 80
        uint8_t* out_pol_lens,   // max_samples
        uint8_t* out_wdl,        // max_samples
        uint8_t* out_train_pol,  // max_samples
        int max_samples
    ) {
        int n = std::min<int>(max_samples, static_cast<int>(finished_samples_.size()));
        for (int i = 0; i < n; ++i) {
            const auto& s = finished_samples_[i];
            std::memcpy(out_c_grids + i * NUM_SQUARES, s.c_grid, NUM_SQUARES);
            out_quiets[i] = s.quiet;
            out_pol_lens[i] = s.pol_len;
            out_wdl[i] = s.wdl_target;
            out_train_pol[i] = s.train_policy;
            std::memcpy(out_pol_acts + i * 80, s.pol_actions, 80 * sizeof(uint16_t));
            std::memcpy(out_pol_probs + i * 80, s.pol_probs, 80 * sizeof(float));
        }
        if (n > 0) {
            finished_samples_.erase(finished_samples_.begin(), finished_samples_.begin() + n);
        }
        return n;
    }

    SelfPlayStats get_stats() const {
        SelfPlayStats s = stats_;
        for (const auto& g : games_) {
            s.tb_node_hits += g.tree.tb_hits();
            s.prover_hits += g.tree.prover_hits();
            s.cache_hits += g.tree.cache_hits();
            s.nn_evals += g.tree.nn_evals();
        }
        return s;
    }

private:
    struct TrajectoryStep {
        int8_t c_grid[NUM_SQUARES];
        uint8_t quiet;
        int8_t acting_side;
        uint8_t pol_len;
        uint8_t train_policy;
        uint16_t pol_actions[80];
        float pol_probs[80];
    };

    struct GameSlot {
        MCTSTree tree;
        FastBoard board;
        std::vector<uint64_t> history_hashes;
        std::vector<TrajectoryStep> trajectory;
        std::mt19937_64 rng;
        int sims_this_turn = 0;
        int target_sims = 64;
        bool is_full_search = false;
        bool noise_applied = false;
        bool waiting_on_gpu = false;

        GameSlot(const tb::TablebaseManager* tb, ZobristNNCache* cache, uint64_t seed)
            : tree(tb, cache, seed), rng(seed + 1337ULL) {}
    };

    struct PendingEval {
        int game_idx;
        int32_t leaf_idx;
        FastBoard board;
    };

    SelfPlayConfig cfg_;
    const tb::TablebaseManager* tb_ = nullptr;
    ZobristNNCache cache_;
    std::mt19937_64 rng_;
    std::vector<GameSlot> games_;
    std::vector<PendingEval> pending_;
    std::vector<FinishedSample> finished_samples_;
    SelfPlayStats stats_;
    int rr_cursor_ = 0;

    static FastBoard make_standard_start() {
        // START_FEN: "9/3RP4/2RPS4/1RPS5/1PS3sp1/5spr1/4spr2/4pr3/9 b"
        static const int8_t start_grid[81] = {
            0,0,0,0,0,0,0,0,0,
            0,0,0,1,2,0,0,0,0,
            0,0,1,2,3,0,0,0,0,
            0,1,2,3,0,0,0,0,0,
            0,2,3,0,0,0,-3,-2,0,
            0,0,0,0,0,-3,-2,-1,0,
            0,0,0,0,-3,-2,-1,0,0,
            0,0,0,0,-2,-1,0,0,0,
            0,0,0,0,0,0,0,0,0
        };
        FastBoard b;
        b.from_array(start_grid, BLUE, 0, 0);
        return b;
    }

    FastBoard sample_curriculum_start(std::mt19937_64& r) {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        double p = u01(r);
        if (p < 0.70) {
            return make_standard_start();
        }
        if (p < 0.85) {
            // Standard opening + 1..4 random legal plies
            for (int attempt = 0; attempt < 10; ++attempt) {
                FastBoard b = make_standard_start();
                int plies = 1 + static_cast<int>(r() % 4);
                bool ok = true;
                for (int k = 0; k < plies; ++k) {
                    uint16_t acts[80];
                    int n = b.gen_legal_actions(acts);
                    if (n == 0) { ok = false; break; }
                    b.push_action(acts[r() % n]);
                    if (b.terminal_result() != 0) { ok = false; break; }
                }
                if (ok) {
                    b.ply = 0;
                    return b;
                }
            }
            return make_standard_start();
        }

        // 15%: Open-board 5..8 piece endgames (3v2, 2v3, 3v3, 4v3, 3v4, 4v4)
        static constexpr int PAIRS[6][2] = {{3, 2}, {2, 3}, {3, 3}, {4, 3}, {3, 4}, {4, 4}};
        for (int attempt = 0; attempt < 30; ++attempt) {
            int pidx = static_cast<int>(r() % 6);
            int nb = PAIRS[pidx][0];
            int nr = PAIRS[pidx][1];
            FastBoard b;
            b.side = (r() % 2 == 0) ? BLUE : RED;
            BB128 occ{};

            for (int i = 0; i < nb; ++i) {
                int s = static_cast<int>(r() % 80);  // 0..79 (excludes BLUE_GOAL=80)
                while (occ.test(s)) s = (s + 1) % 80;
                occ.set(s);
                int8_t pt = static_cast<int8_t>(1 + (r() % 3));
                b.bb[pt - 1].set(s);
                b.sq[s] = pt;
            }
            for (int i = 0; i < nr; ++i) {
                int s = 1 + static_cast<int>(r() % 80);  // 1..80 (excludes RED_GOAL=0)
                while (occ.test(s)) s = 1 + (s % 80);
                occ.set(s);
                int pt = 1 + static_cast<int>(r() % 3);
                b.bb[3 + pt - 1].set(s);
                b.sq[s] = static_cast<int8_t>(-pt);
            }
            b.recompute_hash();
            if (b.terminal_result() != 0) continue;
            if (has_immediate_corner_win(b)) continue;
            if (prove_corner_win_3ply(b) >= 0) continue;
            return b;
        }
        return make_standard_start();
    }

    void setup_next_turn_budget(GameSlot& g) {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        g.sims_this_turn = 0;
        g.noise_applied = false;
        g.is_full_search = (u01(g.rng) < static_cast<double>(cfg_.full_search_prob));
        g.target_sims = g.is_full_search ? cfg_.full_simulations : cfg_.fast_simulations;
    }

    void init_new_game(int gidx) {
        GameSlot& g = games_[gidx];
        g.board = sample_curriculum_start(g.rng);
        g.history_hashes.clear();
        g.history_hashes.push_back(g.board.hash);
        g.trajectory.clear();
        g.waiting_on_gpu = false;
        g.tree.reset(g.board, &g.history_hashes);
        setup_next_turn_budget(g);
    }

    void emit_finished_game(GameSlot& g, int blue_winner, bool via_tb) {
        // blue_winner: +1 BLUE, -1 RED, 0 DRAW
        stats_.games_finished++;
        stats_.total_plies += g.board.ply;
        if (blue_winner > 0) stats_.blue_wins++;
        else if (blue_winner < 0) stats_.red_wins++;
        else stats_.draws++;
        if (via_tb) stats_.tb_terminations++;

        for (const auto& step : g.trajectory) {
            FinishedSample fs{};
            std::memcpy(fs.c_grid, step.c_grid, NUM_SQUARES);
            fs.quiet = step.quiet;
            fs.pol_len = step.pol_len;
            fs.train_policy = step.train_policy;
            std::memcpy(fs.pol_actions, step.pol_actions, step.pol_len * sizeof(uint16_t));
            std::memcpy(fs.pol_probs, step.pol_probs, step.pol_len * sizeof(float));

            if (blue_winner == 0) {
                fs.wdl_target = 1;  // DRAW
            } else if (blue_winner == step.acting_side) {
                fs.wdl_target = 2;  // WIN
            } else {
                fs.wdl_target = 0;  // LOSS
            }
            finished_samples_.push_back(fs);
        }
    }

    void finish_turn_and_maybe_game(int gidx) {
        GameSlot& g = games_[gidx];
        float policy[ACTION_SIZE];
        float temp = (g.board.ply < cfg_.temp_threshold_ply) ? 1.0f : 0.15f;
        g.tree.extract_root_policy(temp, policy);

        if (g.is_full_search) {
            stats_.pv_depth_sum += static_cast<uint64_t>(g.tree.root_pv_depth());
            stats_.full_searches++;
        }

        // Record sparse trajectory entry
        TrajectoryStep ts{};
        FastBoard cb = g.board.canonical_copy();
        std::memcpy(ts.c_grid, cb.sq, NUM_SQUARES);
        ts.quiet = g.board.quiet;
        ts.acting_side = g.board.side;
        ts.train_policy = g.is_full_search ? 1 : 0;

        uint8_t plen = 0;
        double psum = 0.0;
        for (int a = 0; a < ACTION_SIZE; ++a) {
            if (policy[a] > 0.0f && plen < 80) {
                ts.pol_actions[plen] = static_cast<uint16_t>(a);
                ts.pol_probs[plen] = policy[a];
                psum += static_cast<double>(policy[a]);
                ++plen;
            }
        }
        if (plen == 0) {
            // Fallback legal uniform if root had 0 children (shouldn't happen on live board)
            init_new_game(gidx);
            return;
        }
        ts.pol_len = plen;
        g.trajectory.push_back(ts);

        // Sample canonical action from policy
        std::uniform_real_distribution<double> u01(0.0, psum);
        double target = u01(g.rng);
        double cum = 0.0;
        uint16_t chosen_c_act = ts.pol_actions[0];
        for (int i = 0; i < plen; ++i) {
            cum += static_cast<double>(ts.pol_probs[i]);
            if (cum >= target) {
                chosen_c_act = ts.pol_actions[i];
                break;
            }
        }

        uint16_t real_act = (g.board.side == BLUE) ? chosen_c_act : tables().rotate_action[chosen_c_act];
        g.board.push_action(real_act);
        g.history_hashes.push_back(g.board.hash);

        // Check natural termination: terminal rules, 3-fold repetition, or exact 4-piece TB
        int term = g.board.terminal_result();
        if (term != 0) {
            int bw = (term == 2) ? 0 : term;
            emit_finished_game(g, bw, false);
            init_new_game(gidx);
            return;
        }

        int rep = 0;
        for (uint64_t hh : g.history_hashes) {
            if (hh == g.board.hash) ++rep;
        }
        if (rep >= 3 || g.board.ply >= cfg_.hard_max_plies) {
            emit_finished_game(g, 0, false);
            init_new_game(gidx);
            return;
        }

        if (tb_ && tb_->is_loaded() && g.board.piece_count() <= tb_->max_pieces()) {
            int wdl = tb_->probe_wdl_fast(g.board);
            if (wdl != -99) {
                int bw = (wdl == 0) ? 0 : ((wdl > 0) ? g.board.side : -g.board.side);
                emit_finished_game(g, bw, true);
                init_new_game(gidx);
                return;
            }
        }

        // Reuse chosen child subtree for the next turn!
        if (!g.tree.advance_root(chosen_c_act)) {
            g.tree.reset(g.board, &g.history_hashes);
        }
        setup_next_turn_budget(g);
    }
};

}  // namespace intransitive
