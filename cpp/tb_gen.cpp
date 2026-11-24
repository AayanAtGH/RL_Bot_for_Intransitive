#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "tb_index.hpp"
#include "tb_probe.hpp"

using namespace intransitive;
using namespace intransitive::tb;

namespace {

constexpr uint8_t STATUS_INVALID = 254;
constexpr uint8_t STATUS_UNKNOWN = 255;

struct SliceGenData {
    const SliceInfo* info = nullptr;
    std::vector<std::atomic<uint8_t>> wdl;
    std::vector<std::atomic<uint8_t>> out_degree;
    std::vector<uint8_t> dtz;
    std::vector<uint32_t> frontier_win;
    std::vector<uint32_t> frontier_loss;

    explicit SliceGenData(const SliceInfo& inf)
        : info(&inf), wdl(inf.num_entries), out_degree(inf.num_entries), dtz(inf.num_entries, 0) {
        for (size_t i = 0; i < inf.num_entries; ++i) {
            wdl[i].store(STATUS_UNKNOWN, std::memory_order_relaxed);
            out_degree[i].store(0, std::memory_order_relaxed);
        }
    }
};

// Generate all unique canonical predecessor indices in `pred_slice_id` via quiet unmoves from `child_blue`.
// Remember: `child_blue` has side == BLUE, so the side that just moved to create `child_blue` is RED in `child_blue`.
inline int generate_quiet_predecessors(
    const SliceInfo& child_info,
    const FastBoard& child_blue,
    const SliceInfo& pred_info,
    uint32_t* out_preds
) {
    const auto& t = tables();
    int count = 0;

    FastBoard variants[2];
    int num_variants = 1;
    variants[0] = child_blue;
    if (child_info.use_diag_fold) {
        FastBoard refl = child_blue.reflect_diag();
        if (refl.hash != child_blue.hash || std::memcmp(refl.sq, child_blue.sq, NUM_SQUARES) != 0) {
            variants[1] = refl;
            num_variants = 2;
        }
    }

    for (int v = 0; v < num_variants; ++v) {
        const FastBoard& c = variants[v];
        BB128 occ = c.occ_bb();
        BB128 red = c.red_bb();
        while (red.any()) {
            int sq_now = red.pop_lsb();
            int8_t piece = c.sq[sq_now];  // negative (-1, -2, -3)
            int r_idx = piece_to_bb_idx(piece);
            BB128 candidates = t.king_attacks[sq_now] & (~occ);
            while (candidates.any()) {
                int sq_prev = candidates.pop_lsb();
                if (sq_prev == RED_GOAL) continue;  // Red cannot start from a1 (already terminal)

                FastBoard pred = c;
                pred.bb[r_idx].clear(sq_now);
                pred.bb[r_idx].set(sq_prev);
                pred.sq[sq_now] = EMPTY;
                pred.sq[sq_prev] = piece;
                pred.side = RED;  // Predecessor was Red-to-move (Red moved sq_prev -> sq_now)
                pred.recompute_hash();

                int p_sid = -1;
                uint64_t p_idx = 0;
                if (canonical_slice_and_index(pred, p_sid, p_idx)) {
                    if (p_sid == pred_info.id) {
                        uint32_t pidx32 = static_cast<uint32_t>(p_idx);
                        bool dup = false;
                        for (int k = 0; k < count; ++k) {
                            if (out_preds[k] == pidx32) { dup = true; break; }
                        }
                        if (!dup) out_preds[count++] = pidx32;
                    }
                }
            }
        }
    }
    return count;
}

void write_slice_files(const SliceGenData& sd, const std::string& out_dir) {
    namespace fs = std::filesystem;
    const SliceInfo& info = *sd.info;
    size_t n = info.num_entries;
    size_t wdl_bytes = (n + 3ULL) / 4ULL;
    std::vector<uint8_t> packed(wdl_bytes, 0);

    uint64_t cnt_win = 0, cnt_draw = 0, cnt_loss = 0, cnt_cursed = 0, cnt_blessed = 0, cnt_valid = 0;
    int max_dtz_valid = 0;

    std::vector<uint8_t> dtz_out(n, 0);
    for (size_t i = 0; i < n; ++i) {
        uint8_t st = sd.wdl[i].load(std::memory_order_relaxed);
        uint8_t w = WDL_DRAW;
        uint8_t d = sd.dtz[i];
        if (st == STATUS_INVALID) {
            w = WDL_DRAW;
            d = 0;
        } else {
            ++cnt_valid;
            if (st == WDL_WIN) {
                if (d > QUIET_LIMIT) {
                    w = WDL_CURSED_BLESSED;
                    d = DTZ_CURSED_WIN;
                    ++cnt_cursed;
                } else {
                    w = WDL_WIN;
                    ++cnt_win;
                    if (d > max_dtz_valid) max_dtz_valid = d;
                }
            } else if (st == WDL_LOSS) {
                if (d > QUIET_LIMIT) {
                    w = WDL_CURSED_BLESSED;
                    d = DTZ_BLESSED_LOSS;
                    ++cnt_blessed;
                } else {
                    w = WDL_LOSS;
                    ++cnt_loss;
                    if (d > max_dtz_valid) max_dtz_valid = d;
                }
            } else {
                w = WDL_DRAW;
                d = DTZ_DRAW;
                ++cnt_draw;
            }
        }
        packed[i >> 2] |= static_cast<uint8_t>(w << ((i & 3ULL) * 2ULL));
        dtz_out[i] = d;
    }

    fs::path wdl_tmp = fs::path(out_dir) / (info.name + ".wdl.tmp");
    fs::path wdl_fin = fs::path(out_dir) / (info.name + ".wdl");
    fs::path dtz_tmp = fs::path(out_dir) / (info.name + ".dtz.tmp");
    fs::path dtz_fin = fs::path(out_dir) / (info.name + ".dtz");

    {
        std::ofstream fw(wdl_tmp, std::ios::binary);
        fw.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    }
    {
        std::ofstream fd(dtz_tmp, std::ios::binary);
        fd.write(reinterpret_cast<const char*>(dtz_out.data()), static_cast<std::streamsize>(dtz_out.size()));
    }
    fs::rename(wdl_tmp, wdl_fin);
    fs::rename(dtz_tmp, dtz_fin);

    std::cout << "  [" << info.name << "] valid=" << cnt_valid
              << " | W=" << cnt_win << " D=" << cnt_draw << " L=" << cnt_loss
              << " (cursed=" << cnt_cursed << ", blessed=" << cnt_blessed
              << ", maxDTZ=" << max_dtz_valid << ")\n";
}

void solve_coupled_pair(
    int slice_a_id,
    int slice_b_id,
    const TablebaseManager& prev_tb,
    const std::string& out_dir,
    int num_threads
) {
    const auto& reg = registry();
    bool self_dual = (slice_a_id == slice_b_id);

    std::vector<std::unique_ptr<SliceGenData>> pair_data;
    pair_data.push_back(std::make_unique<SliceGenData>(reg.slices[slice_a_id]));
    if (!self_dual) {
        pair_data.push_back(std::make_unique<SliceGenData>(reg.slices[slice_b_id]));
    }

    auto t0 = std::chrono::steady_clock::now();

    // Pass 0: Unrank, mark invalid slots, seed immediate wins/losses & captures from (N-1)
    for (int s_idx = 0; s_idx < static_cast<int>(pair_data.size()); ++s_idx) {
        SliceGenData& sd = *pair_data[s_idx];
        const SliceInfo& info = *sd.info;
        int dual_local_idx = self_dual ? 0 : (1 - s_idx);
        const SliceInfo& dual_info = *pair_data[dual_local_idx]->info;

        std::vector<std::vector<uint32_t>> local_wins(num_threads);
        std::vector<std::vector<uint32_t>> local_losses(num_threads);

        auto worker_pass0 = [&](int tid) {
            uint64_t chunk = (info.num_entries + num_threads - 1) / num_threads;
            uint64_t start = static_cast<uint64_t>(tid) * chunk;
            uint64_t end = std::min<uint64_t>(info.num_entries, start + chunk);

            FastBoard b;
            uint16_t actions[80];
            uint32_t quiet_children[64];

            for (uint64_t idx = start; idx < end; ++idx) {
                if (!unrank_board(info, idx, b)) {
                    sd.wdl[idx].store(STATUS_INVALID, std::memory_order_relaxed);
                    continue;
                }
                int num_acts = b.gen_pseudo_actions(actions);
                if (num_acts == 0) {
                    // Immediate no-moves loss (DTZ = 0)
                    sd.wdl[idx].store(WDL_LOSS, std::memory_order_relaxed);
                    sd.dtz[idx] = 0;
                    local_losses[tid].push_back(static_cast<uint32_t>(idx));
                    continue;
                }

                bool immediate_win = false;
                bool has_drawing_capture = false;
                int num_qc = 0;
                const auto& t = tables();

                for (int a = 0; a < num_acts; ++a) {
                    uint16_t act = actions[a];
                    int from_sq = act / NUM_DIRECTIONS;
                    int to_sq = t.dir_to_sq[from_sq][act % NUM_DIRECTIONS];
                    if (to_sq == BLUE_GOAL) {
                        immediate_win = true;
                        break;
                    }
                    bool is_cap = (b.sq[to_sq] != EMPTY);
                    UndoInfo u = b.push_action(act);

                    if (!b.has_any_pseudo_move()) {
                        b.pop_action(act, u);
                        immediate_win = true;
                        break;
                    }

                    if (is_cap) {
                        TBProbeResult cr = prev_tb.probe(b);
                        if (!cr.available) {
                            std::cerr << "ERROR: Missing (N-1) table during Pass 0!\n";
                            std::abort();
                        }
                        if (cr.wdl == -1) {
                            b.pop_action(act, u);
                            immediate_win = true;
                            break;
                        } else if (cr.wdl == 0) {
                            has_drawing_capture = true;
                        }
                    } else {
                        int c_sid = -1;
                        uint64_t c_idx = 0;
                        if (canonical_slice_and_index(b, c_sid, c_idx) && c_sid == dual_info.id) {
                            uint32_t c32 = static_cast<uint32_t>(c_idx);
                            bool dup = false;
                            for (int k = 0; k < num_qc; ++k) {
                                if (quiet_children[k] == c32) { dup = true; break; }
                            }
                            if (!dup) quiet_children[num_qc++] = c32;
                        }
                    }
                    b.pop_action(act, u);
                }

                if (immediate_win) {
                    sd.wdl[idx].store(WDL_WIN, std::memory_order_relaxed);
                    sd.dtz[idx] = 1;
                    local_wins[tid].push_back(static_cast<uint32_t>(idx));
                } else if (has_drawing_capture) {
                    sd.out_degree[idx].store(255, std::memory_order_relaxed);
                } else if (num_qc == 0) {
                    // Only legal moves are losing captures -> forced zeroing loss on ply 1!
                    sd.wdl[idx].store(WDL_LOSS, std::memory_order_relaxed);
                    sd.dtz[idx] = 1;
                    local_losses[tid].push_back(static_cast<uint32_t>(idx));
                } else {
                    sd.out_degree[idx].store(static_cast<uint8_t>(num_qc), std::memory_order_relaxed);
                }
            }
        };

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) threads.emplace_back(worker_pass0, t);
        for (auto& th : threads) th.join();

        for (int t = 0; t < num_threads; ++t) {
            sd.frontier_win.insert(sd.frontier_win.end(), local_wins[t].begin(), local_wins[t].end());
            sd.frontier_loss.insert(sd.frontier_loss.end(), local_losses[t].begin(), local_losses[t].end());
        }
    }

    // Iterative Retrograde BFS across coupled slices
    // Separate d=0 losses vs d=1 losses cleanly.
    std::vector<std::vector<uint32_t>> curr_loss(pair_data.size());
    std::vector<std::vector<uint32_t>> curr_win(pair_data.size());
    std::vector<std::vector<uint32_t>> deferred_loss_d1(pair_data.size());

    for (size_t s = 0; s < pair_data.size(); ++s) {
        for (uint32_t idx : pair_data[s]->frontier_loss) {
            if (pair_data[s]->dtz[idx] == 0) curr_loss[s].push_back(idx);
            else deferred_loss_d1[s].push_back(idx);
        }
        curr_win[s].swap(pair_data[s]->frontier_win);
        pair_data[s]->frontier_loss.clear();
    }

    // Step d=0 -> d=1 for immediate no-moves losses (if any)
    for (size_t s = 0; s < pair_data.size(); ++s) {
        if (curr_loss[s].empty()) continue;
        size_t ps = self_dual ? 0 : (1 - s);
        const SliceInfo& c_info = *pair_data[s]->info;
        SliceGenData& pred_sd = *pair_data[ps];
        const SliceInfo& p_info = *pred_sd.info;
        const auto& front = curr_loss[s];
        std::vector<std::vector<uint32_t>> thread_wins(num_threads);
        auto worker_d0 = [&](int tid) {
            size_t chunk = (front.size() + num_threads - 1) / num_threads;
            size_t st = static_cast<size_t>(tid) * chunk;
            size_t en = std::min(front.size(), st + chunk);
            FastBoard cb;
            uint32_t preds[64];
            for (size_t i = st; i < en; ++i) {
                if (!unrank_board(c_info, front[i], cb)) continue;
                int np = generate_quiet_predecessors(c_info, cb, p_info, preds);
                for (int k = 0; k < np; ++k) {
                    uint32_t pidx = preds[k];
                    uint8_t expected = STATUS_UNKNOWN;
                    if (pred_sd.wdl[pidx].compare_exchange_strong(expected, WDL_WIN, std::memory_order_acq_rel)) {
                        pred_sd.dtz[pidx] = 1;
                        thread_wins[tid].push_back(pidx);
                    }
                }
            }
        };
        std::vector<std::thread> ths;
        for (int t = 0; t < num_threads; ++t) ths.emplace_back(worker_d0, t);
        for (auto& th : ths) th.join();
        for (int t = 0; t < num_threads; ++t) {
            curr_win[ps].insert(curr_win[ps].end(), thread_wins[t].begin(), thread_wins[t].end());
        }
    }
    for (size_t s = 0; s < pair_data.size(); ++s) {
        curr_loss[s].swap(deferred_loss_d1[s]);
        deferred_loss_d1[s].clear();
    }

    for (int d = 1; ; ++d) {
        bool any_work = false;
        for (size_t s = 0; s < pair_data.size(); ++s) {
            if (!curr_loss[s].empty() || !curr_win[s].empty()) {
                any_work = true;
                break;
            }
        }
        if (!any_work) break;

        uint8_t next_d = static_cast<uint8_t>(std::min(205, d + 1));
        std::vector<std::vector<uint32_t>> next_win(pair_data.size());
        std::vector<std::vector<uint32_t>> next_loss(pair_data.size());

        // Phase A: Expand curr_loss[s] -> predecessors in dual become WIN(d+1)
        for (size_t s = 0; s < pair_data.size(); ++s) {
            if (curr_loss[s].empty()) continue;
            size_t ps = self_dual ? 0 : (1 - s);
            const SliceInfo& c_info = *pair_data[s]->info;
            SliceGenData& pred_sd = *pair_data[ps];
            const SliceInfo& p_info = *pred_sd.info;
            const auto& front = curr_loss[s];

            std::vector<std::vector<uint32_t>> thread_wins(num_threads);
            auto worker_a = [&](int tid) {
                size_t chunk = (front.size() + num_threads - 1) / num_threads;
                size_t st = static_cast<size_t>(tid) * chunk;
                size_t en = std::min(front.size(), st + chunk);
                FastBoard cb;
                uint32_t preds[64];
                for (size_t i = st; i < en; ++i) {
                    if (!unrank_board(c_info, front[i], cb)) continue;
                    int np = generate_quiet_predecessors(c_info, cb, p_info, preds);
                    for (int k = 0; k < np; ++k) {
                        uint32_t pidx = preds[k];
                        uint8_t expected = STATUS_UNKNOWN;
                        if (pred_sd.wdl[pidx].compare_exchange_strong(expected, WDL_WIN, std::memory_order_acq_rel)) {
                            pred_sd.dtz[pidx] = next_d;
                            thread_wins[tid].push_back(pidx);
                        }
                    }
                }
            };
            std::vector<std::thread> ths;
            for (int t = 0; t < num_threads; ++t) ths.emplace_back(worker_a, t);
            for (auto& th : ths) th.join();
            for (int t = 0; t < num_threads; ++t) {
                next_win[ps].insert(next_win[ps].end(), thread_wins[t].begin(), thread_wins[t].end());
            }
        }

        // Phase B: Expand curr_win[s] -> decrement out_degree -> 0 becomes LOSS(d+1)
        for (size_t s = 0; s < pair_data.size(); ++s) {
            if (curr_win[s].empty()) continue;
            size_t ps = self_dual ? 0 : (1 - s);
            const SliceInfo& c_info = *pair_data[s]->info;
            SliceGenData& pred_sd = *pair_data[ps];
            const SliceInfo& p_info = *pred_sd.info;
            const auto& front = curr_win[s];

            std::vector<std::vector<uint32_t>> thread_losses(num_threads);
            auto worker_b = [&](int tid) {
                size_t chunk = (front.size() + num_threads - 1) / num_threads;
                size_t st = static_cast<size_t>(tid) * chunk;
                size_t en = std::min(front.size(), st + chunk);
                FastBoard cb;
                uint32_t preds[64];
                for (size_t i = st; i < en; ++i) {
                    if (!unrank_board(c_info, front[i], cb)) continue;
                    int np = generate_quiet_predecessors(c_info, cb, p_info, preds);
                    for (int k = 0; k < np; ++k) {
                        uint32_t pidx = preds[k];
                        if (pred_sd.wdl[pidx].load(std::memory_order_relaxed) != STATUS_UNKNOWN) continue;
                        uint8_t cur_deg = pred_sd.out_degree[pidx].load(std::memory_order_relaxed);
                        if (cur_deg == 255 || cur_deg == 0) continue;
                        uint8_t prev_deg = pred_sd.out_degree[pidx].fetch_sub(1, std::memory_order_acq_rel);
                        if (prev_deg == 1) {
                            uint8_t expected = STATUS_UNKNOWN;
                            if (pred_sd.wdl[pidx].compare_exchange_strong(expected, WDL_LOSS, std::memory_order_acq_rel)) {
                                pred_sd.dtz[pidx] = next_d;
                                thread_losses[tid].push_back(pidx);
                            }
                        }
                    }
                }
            };
            std::vector<std::thread> ths;
            for (int t = 0; t < num_threads; ++t) ths.emplace_back(worker_b, t);
            for (auto& th : ths) th.join();
            for (int t = 0; t < num_threads; ++t) {
                next_loss[ps].insert(next_loss[ps].end(), thread_losses[t].begin(), thread_losses[t].end());
            }
        }

        curr_win.swap(next_win);
        curr_loss.swap(next_loss);
    }

    for (size_t s = 0; s < pair_data.size(); ++s) {
        write_slice_files(*pair_data[s], out_dir);
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "  -> Solved slice pair in " << secs << "s\n";
}

}  // namespace

int main(int argc, char** argv) {
    int max_pieces = 4;
    std::string out_dir = "data/tablebase";
    int num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    bool overwrite = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--max-pieces" && i + 1 < argc) max_pieces = std::stoi(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) num_threads = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--overwrite") overwrite = true;
    }

    max_pieces = std::clamp(max_pieces, 2, 4);
    std::filesystem::create_directories(out_dir);

    std::cout << "=== Intransitive Tablebase Generator (max_pieces=" << max_pieces
              << ", threads=" << num_threads << ", out=" << out_dir << ") ===\n";

    auto total_t0 = std::chrono::steady_clock::now();
    const auto& reg = registry();

    for (int n = 2; n <= max_pieces; ++n) {
        std::cout << "\n--- Generating N=" << n << " piece tables ---\n";
        TablebaseManager prev_tb;
        if (n > 2) {
            if (!prev_tb.load(out_dir) || prev_tb.max_pieces() < n - 1) {
                std::cerr << "ERROR: Failed to load N=" << (n - 1) << " tables from " << out_dir << "\n";
                return 1;
            }
        }

        std::vector<bool> done(reg.slices.size(), false);
        for (const auto& slice : reg.slices) {
            if (slice.total_pieces != n || done[slice.id]) continue;
            int a = slice.id;
            int b = slice.dual_id;
            done[a] = true;
            done[b] = true;

            if (!overwrite) {
                namespace fs = std::filesystem;
                bool a_ok = fs::exists(fs::path(out_dir) / (reg.slices[a].name + ".wdl")) &&
                            fs::exists(fs::path(out_dir) / (reg.slices[a].name + ".dtz"));
                bool b_ok = fs::exists(fs::path(out_dir) / (reg.slices[b].name + ".wdl")) &&
                            fs::exists(fs::path(out_dir) / (reg.slices[b].name + ".dtz"));
                if (a_ok && b_ok) {
                    std::cout << "  [SKIP existing] " << reg.slices[a].name
                              << (a == b ? "" : (" + " + reg.slices[b].name)) << "\n";
                    continue;
                }
            }

            solve_coupled_pair(a, b, prev_tb, out_dir, num_threads);
        }
    }

    auto total_t1 = std::chrono::steady_clock::now();
    double total_secs = std::chrono::duration<double>(total_t1 - total_t0).count();
    std::cout << "\nAll N<= " << max_pieces << " tablebases completed in " << total_secs << "s.\n";
    return 0;
}
