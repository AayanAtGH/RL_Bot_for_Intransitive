#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "board.hpp"

namespace intransitive {
namespace tb {

// WDL encoding stored packed (2 bits/entry) in *.wdl:
//   0 = LOSS (strictly loses within <=200 quiet plies)
//   1 = DRAW (true draw, or invalid/unused slot)
//   2 = WIN  (strictly wins within <=200 quiet plies)
//   3 = CURSED_OR_BLESSED (forced win/loss > 200 plies; effective draw at quiet=0; check *.dtz: 201=cursed win, 202=blessed loss)
constexpr uint8_t WDL_LOSS = 0;
constexpr uint8_t WDL_DRAW = 1;
constexpr uint8_t WDL_WIN = 2;
constexpr uint8_t WDL_CURSED_BLESSED = 3;

constexpr uint8_t DTZ_DRAW = 0;
constexpr uint8_t DTZ_CURSED_WIN = 201;
constexpr uint8_t DTZ_BLESSED_LOSS = 202;

struct Sig {
    uint8_t b[3]{};  // R, P, S counts for Blue (side to move)
    uint8_t r[3]{};  // R, P, S counts for Red

    bool operator==(const Sig& o) const {
        return std::memcmp(b, o.b, 3) == 0 && std::memcmp(r, o.r, 3) == 0;
    }
    bool operator<(const Sig& o) const {
        for (int i = 0; i < 3; ++i) if (b[i] != o.b[i]) return b[i] < o.b[i];
        for (int i = 0; i < 3; ++i) if (r[i] != o.r[i]) return r[i] < o.r[i];
        return false;
    }

    int blue_count() const { return b[0] + b[1] + b[2]; }
    int red_count() const { return r[0] + r[1] + r[2]; }
    int total_count() const { return blue_count() + red_count(); }

    // Matches FastBoard::cycle_rps(shift): old pt -> (pt + shift) % 3
    Sig shifted(int shift) const {
        shift = ((shift % 3) + 3) % 3;
        Sig out;
        for (int pt = 0; pt < 3; ++pt) {
            int npt = (pt + shift) % 3;
            out.b[npt] = b[pt];
            out.r[npt] = r[pt];
        }
        return out;
    }

    std::string name() const {
        static constexpr char B_CHARS[3] = {'R', 'P', 'S'};
        static constexpr char R_CHARS[3] = {'r', 'p', 's'};
        std::string s;
        for (int pt = 0; pt < 3; ++pt) {
            for (int k = 0; k < b[pt]; ++k) s.push_back(B_CHARS[pt]);
        }
        s += "_vs_";
        for (int pt = 0; pt < 3; ++pt) {
            for (int k = 0; k < r[pt]; ++k) s.push_back(R_CHARS[pt]);
        }
        return s;
    }
};

// Return canonical C3 shift s in {0, 1, 2} such that sig.shifted(s) is lexicographically minimal
inline int canonical_c3_shift(const Sig& sig, Sig* out_canon = nullptr) {
    int best_s = 0;
    Sig best = sig;
    for (int s = 1; s < 3; ++s) {
        Sig cand = sig.shifted(s);
        if (cand < best) {
            best = cand;
            best_s = s;
        }
    }
    if (out_canon) *out_canon = best;
    return best_s;
}

inline Sig dual_canonical_sig(const Sig& s) {
    Sig flipped;
    for (int i = 0; i < 3; ++i) {
        flipped.b[i] = s.r[i];
        flipped.r[i] = s.b[i];
    }
    Sig out;
    canonical_c3_shift(flipped, &out);
    return out;
}

struct GroupSpec {
    bool is_blue;
    uint8_t pt;     // 0=R, 1=P, 2=S
    uint8_t count;  // 1..3
};

struct IndexTables {
    uint64_t comb[81][5]{};
    uint8_t tri_blue[44]{};      // squares r <= f excluding BLUE_GOAL (80)
    int8_t blue_to_tri[81]{};    // -1 if r > f or sq == 80
    uint8_t tri_red[44]{};       // squares r <= f excluding RED_GOAL (0)
    int8_t red_to_tri[81]{};     // -1 if r > f or sq == 0

    IndexTables() {
        for (int n = 0; n <= 80; ++n) {
            comb[n][0] = 1;
            for (int k = 1; k <= 4; ++k) {
                if (k > n) comb[n][k] = 0;
                else comb[n][k] = comb[n - 1][k - 1] + comb[n - 1][k];
            }
        }
        std::memset(blue_to_tri, -1, sizeof(blue_to_tri));
        std::memset(red_to_tri, -1, sizeof(red_to_tri));
        int ib = 0, ir = 0;
        for (int sq = 0; sq < NUM_SQUARES; ++sq) {
            int r = sq / BOARD_SIZE;
            int f = sq % BOARD_SIZE;
            if (r <= f) {
                if (sq != BLUE_GOAL) {
                    tri_blue[ib] = static_cast<uint8_t>(sq);
                    blue_to_tri[sq] = static_cast<int8_t>(ib);
                    ++ib;
                }
                if (sq != RED_GOAL) {
                    tri_red[ir] = static_cast<uint8_t>(sq);
                    red_to_tri[sq] = static_cast<int8_t>(ir);
                    ++ir;
                }
            }
        }
    }
};

inline const IndexTables& idx_tables() {
    static const IndexTables t;
    return t;
}

struct SliceInfo {
    int id = -1;
    int dual_id = -1;
    int total_pieces = 0;
    Sig sig{};
    std::string name;
    bool use_diag_fold = false;
    int num_groups = 0;
    GroupSpec groups[4]{};
    uint64_t group_radix[4]{};
    uint64_t num_entries = 0;
};

inline void build_groups_for_sig(SliceInfo& info) {
    std::vector<GroupSpec> raw;
    for (int pt = 0; pt < 3; ++pt) {
        if (info.sig.b[pt] > 0) raw.push_back({true, static_cast<uint8_t>(pt), info.sig.b[pt]});
    }
    for (int pt = 0; pt < 3; ++pt) {
        if (info.sig.r[pt] > 0) raw.push_back({false, static_cast<uint8_t>(pt), info.sig.r[pt]});
    }

    int singleton_pos = -1;
    for (int i = 0; i < static_cast<int>(raw.size()); ++i) {
        if (raw[i].count == 1) {
            singleton_pos = i;
            break;
        }
    }

    info.num_groups = static_cast<int>(raw.size());
    const auto& it = idx_tables();
    if (singleton_pos >= 0) {
        info.use_diag_fold = true;
        info.groups[0] = raw[singleton_pos];
        info.group_radix[0] = 44ULL;
        int out_i = 1;
        for (int i = 0; i < static_cast<int>(raw.size()); ++i) {
            if (i == singleton_pos) continue;
            info.groups[out_i] = raw[i];
            info.group_radix[out_i] = it.comb[80][raw[i].count];
            ++out_i;
        }
    } else {
        info.use_diag_fold = false;
        for (int i = 0; i < static_cast<int>(raw.size()); ++i) {
            info.groups[i] = raw[i];
            info.group_radix[i] = it.comb[80][raw[i].count];
        }
    }

    uint64_t total = 1ULL;
    for (int g = 0; g < info.num_groups; ++g) {
        total *= info.group_radix[g];
    }
    info.num_entries = total;
}

class SliceRegistry {
public:
    std::vector<SliceInfo> slices;

    SliceRegistry() {
        for (int n = 2; n <= 4; ++n) {
            std::vector<Sig> sigs;
            for (int b_tot = 1; b_tot < n; ++b_tot) {
                int r_tot = n - b_tot;
                for (int bR = 0; bR <= b_tot; ++bR) {
                    for (int bP = 0; bP <= b_tot - bR; ++bP) {
                        int bS = b_tot - bR - bP;
                        for (int rR = 0; rR <= r_tot; ++rR) {
                            for (int rP = 0; rP <= r_tot - rR; ++rP) {
                                int rS = r_tot - rR - rP;
                                Sig raw{{static_cast<uint8_t>(bR), static_cast<uint8_t>(bP), static_cast<uint8_t>(bS)},
                                        {static_cast<uint8_t>(rR), static_cast<uint8_t>(rP), static_cast<uint8_t>(rS)}};
                                Sig canon;
                                canonical_c3_shift(raw, &canon);
                                sigs.push_back(canon);
                            }
                        }
                    }
                }
            }
            std::sort(sigs.begin(), sigs.end());
            sigs.erase(std::unique(sigs.begin(), sigs.end()), sigs.end());
            for (const Sig& s : sigs) {
                SliceInfo info;
                info.id = static_cast<int>(slices.size());
                info.total_pieces = n;
                info.sig = s;
                info.name = s.name();
                build_groups_for_sig(info);
                slices.push_back(std::move(info));
            }
        }
        for (auto& info : slices) {
            Sig ds = dual_canonical_sig(info.sig);
            info.dual_id = find_id(ds);
        }
    }

    int find_id(const Sig& canon_sig) const {
        for (const auto& s : slices) {
            if (s.sig == canon_sig) return s.id;
        }
        return -1;
    }
};

inline const SliceRegistry& registry() {
    static const SliceRegistry reg;
    return reg;
}

inline Sig extract_sig(const FastBoard& b_blue) {
    Sig s;
    for (int pt = 0; pt < 3; ++pt) {
        s.b[pt] = static_cast<uint8_t>(b_blue.bb[pt].popcount());
        s.r[pt] = static_cast<uint8_t>(b_blue.bb[3 + pt].popcount());
    }
    return s;
}

// Compute group-ordered square tuple for `b_blue` (pre-D1 folded) and check if `vec(B) <= vec(reflect(B))`
inline bool is_diag_canonical(const SliceInfo& info, const FastBoard& b_blue) {
    if (!info.use_diag_fold) return true;
    const auto& t = tables();

    uint8_t orig_sqs[4];
    uint8_t refl_sqs[4];
    int pos = 0;

    for (int g = 0; g < info.num_groups; ++g) {
        const auto& grp = info.groups[g];
        int bb_idx = (grp.is_blue ? 0 : 3) + grp.pt;
        BB128 bb = b_blue.bb[bb_idx];
        uint8_t tmp_o[4]{};
        uint8_t tmp_r[4]{};
        int k = 0;
        while (bb.any() && k < 4) {
            int sq = bb.pop_lsb();
            tmp_o[k] = static_cast<uint8_t>(sq);
            tmp_r[k] = t.reflect_sq[sq];
            ++k;
        }
        // tmp_o is already sorted ascending by pop_lsb(); sort small tmp_r in-place
        for (int i = 1; i < k; ++i) {
            uint8_t key = tmp_r[i];
            int j = i - 1;
            while (j >= 0 && tmp_r[j] > key) {
                tmp_r[j + 1] = tmp_r[j];
                --j;
            }
            tmp_r[j + 1] = key;
        }
        for (int i = 0; i < k && pos < 4; ++i) {
            orig_sqs[pos] = tmp_o[i];
            refl_sqs[pos] = tmp_r[i];
            ++pos;
        }
    }

    for (int i = 0; i < pos; ++i) {
        if (orig_sqs[i] < refl_sqs[i]) return true;
        if (orig_sqs[i] > refl_sqs[i]) return false;
    }
    return true;  // self-symmetric across main diagonal
}

// Canonicalize arbitrary non-terminal board (piece_count in 2..4) and return (slice_id, index).
// Returns false if board is terminal (Blue on i9 or Red on a1) or piece count not in 2..4.
inline bool canonical_slice_and_index(const FastBoard& b, int& out_slice_id, uint64_t& out_idx) {
    if (b.sq[BLUE_GOAL] > 0 || b.sq[RED_GOAL] < 0) return false;
    int n = b.piece_count();
    if (n < 2 || n > 4) return false;

    FastBoard cb = b.canonical_copy();  // Blue to move
    Sig raw = extract_sig(cb);
    Sig canon_sig;
    int shift = canonical_c3_shift(raw, &canon_sig);
    if (shift != 0) cb = cb.cycle_rps(shift);

    int sid = registry().find_id(canon_sig);
    if (sid < 0) return false;
    const SliceInfo& info = registry().slices[sid];

    if (info.use_diag_fold && !is_diag_canonical(info, cb)) {
        cb = cb.reflect_diag();
    }

    const auto& it = idx_tables();
    uint64_t idx = 0;
    for (int g = 0; g < info.num_groups; ++g) {
        const auto& grp = info.groups[g];
        int bb_idx = (grp.is_blue ? 0 : 3) + grp.pt;
        BB128 bb = cb.bb[bb_idx];
        uint64_t g_rank = 0;
        if (g == 0 && info.use_diag_fold) {
            int sq = bb.lsb();
            int tri = grp.is_blue ? it.blue_to_tri[sq] : it.red_to_tri[sq];
            if (tri < 0) return false;
            g_rank = static_cast<uint64_t>(tri);
        } else {
            int k = 0;
            while (bb.any()) {
                int sq = bb.pop_lsb();
                int dom = grp.is_blue ? sq : (sq - 1);
                g_rank += it.comb[dom][k + 1];
                ++k;
            }
        }
        idx = idx * info.group_radix[g] + g_rank;
    }

    out_slice_id = sid;
    out_idx = idx;
    return true;
}

// Invert index -> canonical FastBoard (Blue to move). Returns false if invalid (overlapping pieces or non-canonical D1).
inline bool unrank_board(const SliceInfo& info, uint64_t idx, FastBoard& out_b) {
    const auto& it = idx_tables();
    uint64_t g_ranks[4]{};
    for (int g = info.num_groups - 1; g >= 0; --g) {
        g_ranks[g] = idx % info.group_radix[g];
        idx /= info.group_radix[g];
    }

    out_b.clear();
    out_b.side = BLUE;
    BB128 occ{};

    for (int g = 0; g < info.num_groups; ++g) {
        const auto& grp = info.groups[g];
        int bb_idx = (grp.is_blue ? 0 : 3) + grp.pt;
        int8_t piece = static_cast<int8_t>(bb_idx_to_piece(bb_idx));

        if (g == 0 && info.use_diag_fold) {
            int sq = grp.is_blue ? it.tri_blue[g_ranks[0]] : it.tri_red[g_ranks[0]];
            if (occ.test(sq)) return false;
            occ.set(sq);
            out_b.bb[bb_idx].set(sq);
            out_b.sq[sq] = piece;
        } else {
            uint64_t rem = g_ranks[g];
            int upper = 79;
            for (int k = grp.count; k >= 1; --k) {
                int d = k - 1;
                // Binary / linear search for largest d <= upper with comb[d][k] <= rem
                int lo = k - 1, hi = upper;
                while (lo <= hi) {
                    int mid = (lo + hi) >> 1;
                    if (it.comb[mid][k] <= rem) {
                        d = mid;
                        lo = mid + 1;
                    } else {
                        hi = mid - 1;
                    }
                }
                rem -= it.comb[d][k];
                upper = d - 1;
                int sq = grp.is_blue ? d : (d + 1);
                if (occ.test(sq)) return false;
                occ.set(sq);
                out_b.bb[bb_idx].set(sq);
                out_b.sq[sq] = piece;
            }
        }
    }

    if (info.use_diag_fold && !is_diag_canonical(info, out_b)) {
        return false;
    }
    out_b.recompute_hash();
    return true;
}

}  // namespace tb
}  // namespace intransitive
