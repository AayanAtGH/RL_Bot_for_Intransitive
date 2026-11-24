#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include "tb_index.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace intransitive {
namespace tb {

struct MappedFile {
    const uint8_t* data = nullptr;
    size_t size = 0;

#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = NULL;
#else
    int fd = -1;
#endif

    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& o) noexcept
        : data(o.data), size(o.size)
#ifdef _WIN32
        , hFile(o.hFile), hMap(o.hMap)
#else
        , fd(o.fd)
#endif
    {
        o.data = nullptr;
        o.size = 0;
#ifdef _WIN32
        o.hFile = INVALID_HANDLE_VALUE;
        o.hMap = NULL;
#else
        o.fd = -1;
#endif
    }

    ~MappedFile() { close(); }

    bool open(const std::string& path) {
        close();
#ifdef _WIN32
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(hFile, &sz) || sz.QuadPart <= 0) {
            close();
            return false;
        }
        size = static_cast<size_t>(sz.QuadPart);
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            close();
            return false;
        }
        data = static_cast<const uint8_t*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
        if (!data) {
            close();
            return false;
        }
        return true;
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            close();
            return false;
        }
        size = static_cast<size_t>(st.st_size);
        void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            close();
            return false;
        }
        data = static_cast<const uint8_t*>(p);
        return true;
#endif
    }

    void close() {
#ifdef _WIN32
        if (data) { UnmapViewOfFile(data); data = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = NULL; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
#else
        if (data) { munmap(const_cast<uint8_t*>(data), size); data = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
        size = 0;
    }
};

struct TBProbeResult {
    int8_t wdl = 0;       // +1 win, 0 draw, -1 loss (already respects 200-quiet-ply clock if `respect_quiet`)
    uint8_t dtz = 0;      // 0 draw, 1..200 exact plies, 201 cursed win, 202 blessed loss
    uint8_t raw_wdl = 1;  // raw 2-bit stored WDL (0..3)
    bool available = false;
};

class TablebaseManager {
public:
    bool load(const std::string& dir_path) {
        close();
        namespace fs = std::filesystem;
        if (!fs::exists(dir_path)) return false;

        const auto& reg = registry();
        wdl_files_.resize(reg.slices.size());
        dtz_files_.resize(reg.slices.size());

        int loaded_max = 0;
        for (const auto& slice : reg.slices) {
            std::string wdl_p = (fs::path(dir_path) / (slice.name + ".wdl")).string();
            std::string dtz_p = (fs::path(dir_path) / (slice.name + ".dtz")).string();
            if (!fs::exists(wdl_p) || !fs::exists(dtz_p)) continue;

            if (!wdl_files_[slice.id].open(wdl_p)) continue;
            if (!dtz_files_[slice.id].open(dtz_p)) {
                wdl_files_[slice.id].close();
                continue;
            }
            size_t expected_wdl = (slice.num_entries + 3ULL) / 4ULL;
            if (wdl_files_[slice.id].size < expected_wdl || dtz_files_[slice.id].size < slice.num_entries) {
                wdl_files_[slice.id].close();
                dtz_files_[slice.id].close();
                continue;
            }
            if (slice.total_pieces > loaded_max) loaded_max = slice.total_pieces;
        }
        max_loaded_pieces_ = loaded_max;
        return max_loaded_pieces_ >= 2;
    }

    void close() {
        wdl_files_.clear();
        dtz_files_.clear();
        max_loaded_pieces_ = 0;
    }

    int max_pieces() const { return max_loaded_pieces_; }
    bool is_loaded() const { return max_loaded_pieces_ >= 2; }

    // Probe WDL & DTZ from `b.side`'s perspective.
    // Returns TBProbeResult{available=true, ...} if position is terminal, <=1 piece, or in loaded 2..4 piece TB.
    TBProbeResult probe(const FastBoard& b) const {
        TBProbeResult res;
        int term = b.terminal_result();
        if (term != 0) {
            res.available = true;
            res.dtz = 0;
            if (term == 2) {
                res.wdl = 0;
                res.raw_wdl = WDL_DRAW;
            } else if (term == b.side) {
                res.wdl = 1;
                res.raw_wdl = WDL_WIN;
            } else {
                res.wdl = -1;
                res.raw_wdl = WDL_LOSS;
            }
            return res;
        }

        int n = b.piece_count();
        if (n <= 1) {
            // Only 1 piece total and game is live => opponent has 0 pieces!
            // Any move by side-to-move (always exists on 9x9 with 1 piece) leaves opponent with 0 moves -> Win in 1!
            res.available = true;
            res.wdl = 1;
            res.raw_wdl = WDL_WIN;
            res.dtz = 1;
            return res;
        }
        // If one side has 0 pieces and it's the other side's turn (live): side with piece wins in 1!
        if (b.blue_count() == 0 || b.red_count() == 0) {
            res.available = true;
            res.wdl = 1;
            res.raw_wdl = WDL_WIN;
            res.dtz = 1;
            return res;
        }

        if (n > max_loaded_pieces_) return res;

        int sid = -1;
        uint64_t idx = 0;
        if (!canonical_slice_and_index(b, sid, idx)) return res;
        if (sid < 0 || sid >= static_cast<int>(wdl_files_.size())) return res;
        const auto& wf = wdl_files_[sid];
        const auto& df = dtz_files_[sid];
        if (!wf.data || !df.data) return res;

        uint8_t packed = wf.data[idx >> 2];
        uint8_t raw = (packed >> ((idx & 3ULL) * 2ULL)) & 0x3U;
        uint8_t dtz = df.data[idx];

        res.available = true;
        res.raw_wdl = raw;
        res.dtz = dtz;

        if (raw == WDL_WIN) {
            // Honor 200-quiet-ply clock
            res.wdl = (static_cast<int>(b.quiet) + static_cast<int>(dtz) <= QUIET_LIMIT) ? 1 : 0;
        } else if (raw == WDL_LOSS) {
            res.wdl = (static_cast<int>(b.quiet) + static_cast<int>(dtz) <= QUIET_LIMIT) ? -1 : 0;
        } else {
            res.wdl = 0;  // DRAW, CURSED_WIN, or BLESSED_LOSS -> effective draw under 200-ply rule
        }
        return res;
    }

    // Fast WDL-only probe for MCTS internal nodes (+1 win, 0 draw, -1 loss, -99 unavailable)
    int probe_wdl_fast(const FastBoard& b) const {
        TBProbeResult r = probe(b);
        return r.available ? static_cast<int>(r.wdl) : -99;
    }

    // Exact optimal root move selection using DTZ + quiet-clock awareness.
    // Returns real action index (0..647) or -1 if TB unavailable / position already terminal.
    int probe_best_action(const FastBoard& b, TBProbeResult* root_info = nullptr) const {
        if (root_info) *root_info = probe(b);
        if (b.terminal_result() != 0) return -1;

        uint16_t actions[80];
        int num_actions = b.gen_legal_actions(actions);
        if (num_actions == 0) return -1;

        const auto& t = tables();
        int own_goal = (b.side == BLUE) ? BLUE_GOAL : RED_GOAL;

        // 1. Immediate 1-ply corner win check
        for (int i = 0; i < num_actions; ++i) {
            uint16_t act = actions[i];
            int from_sq = act / NUM_DIRECTIONS;
            int to_sq = t.dir_to_sq[from_sq][act % NUM_DIRECTIONS];
            if (to_sq == own_goal) return static_cast<int>(act);
        }

        // Check if TB covers current piece count (or all capture/quiet children)
        TBProbeResult self_r = probe(b);
        if (!self_r.available) return -1;

        int best_act = -1;
        // Tier: 2 = guaranteed win within quiet clock, 1 = guaranteed draw (or swindle), 0 = loss
        int best_tier = -1;
        int best_subscore = -1000000;

        FastBoard work = b;
        for (int i = 0; i < num_actions; ++i) {
            uint16_t act = actions[i];
            int from_sq = act / NUM_DIRECTIONS;
            int to_sq = t.dir_to_sq[from_sq][act % NUM_DIRECTIONS];
            bool is_capture = (work.sq[to_sq] != EMPTY);

            UndoInfo u = work.push_action(act);
            int child_term = work.terminal_result();

            int tier = 0;
            int subscore = 0;

            if (child_term == b.side) {
                // Immediate no-moves win!
                work.pop_action(act, u);
                return static_cast<int>(act);
            } else if (child_term == 2) {
                // Quiet limit draw
                tier = 1;
                subscore = 0;
            } else if (child_term == -b.side) {
                tier = 0;
                subscore = 0;
            } else {
                TBProbeResult cr = probe(work);
                if (!cr.available) {
                    work.pop_action(act, u);
                    return -1;
                }
                // `cr` is from opponent's perspective (already accounts for work.quiet == b.quiet+1 or 0)
                if (cr.wdl == -1) {
                    // Opponent strictly loses within quiet limit -> winning move!
                    tier = 2;
                    // Prefer zeroing capture (breaking ties by fastest N-1 loss); otherwise minimize dtz
                    subscore = is_capture ? (20000 - static_cast<int>(cr.dtz)) : (10000 - static_cast<int>(cr.dtz));
                } else if (cr.wdl == 0) {
                    // Draw (true draw, cursed/blessed >200, or quiet clock pushed past 200!)
                    tier = 1;
                    // If our side was cursed-winning (raw_wdl==WDL_CURSED_BLESSED & dtz==201), prefer keeping pressure (smallest dtz);
                    // otherwise neutral.
                    subscore = (cr.raw_wdl == WDL_CURSED_BLESSED && cr.dtz == DTZ_BLESSED_LOSS) ? 10 : 0;
                } else {
                    // Opponent wins -> losing move; maximize survival distance
                    tier = 0;
                    int opp_dtz = (cr.raw_wdl == WDL_WIN) ? static_cast<int>(cr.dtz) : 205;
                    // Avoid captures that shorten survival if a slower quiet loss exists
                    subscore = opp_dtz + (is_capture ? 0 : 1);
                }
            }
            work.pop_action(act, u);

            if (tier > best_tier || (tier == best_tier && subscore > best_subscore)) {
                best_tier = tier;
                best_subscore = subscore;
                best_act = static_cast<int>(act);
            }
        }
        return best_act;
    }

private:
    std::vector<MappedFile> wdl_files_;
    std::vector<MappedFile> dtz_files_;
    int max_loaded_pieces_ = 0;
};

}  // namespace tb
}  // namespace intransitive
