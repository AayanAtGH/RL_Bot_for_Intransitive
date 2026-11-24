#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "bitboard.hpp"

namespace intransitive {

struct UndoInfo {
    int8_t captured;
    uint8_t prev_quiet;
    uint64_t prev_hash;
};

struct FastBoard {
    // 0=B_R, 1=B_P, 2=B_S, 3=R_R, 4=R_P, 5=R_S
    BB128 bb[6]{};
    int8_t sq[NUM_SQUARES]{};
    int8_t side = BLUE;
    uint8_t quiet = 0;
    uint16_t ply = 0;
    uint64_t hash = 0;

    inline void clear() {
        for (int i = 0; i < 6; ++i) bb[i] = BB128{};
        std::memset(sq, 0, sizeof(sq));
        side = BLUE;
        quiet = 0;
        ply = 0;
        hash = 0;
    }

    inline void recompute_hash() {
        const auto& t = tables();
        uint64_t h = (side == RED) ? t.zobrist_side : 0ULL;
        for (int s = 0; s < NUM_SQUARES; ++s) {
            int8_t p = sq[s];
            if (p != EMPTY) {
                h ^= t.zobrist_piece[piece_to_bb_idx(p)][s];
            }
        }
        hash = h;
    }

    inline void from_array(const int8_t grid[NUM_SQUARES], int8_t side_in, uint8_t quiet_in, uint16_t ply_in) {
        clear();
        side = side_in;
        quiet = quiet_in;
        ply = ply_in;
        for (int s = 0; s < NUM_SQUARES; ++s) {
            int8_t p = grid[s];
            sq[s] = p;
            if (p != EMPTY) {
                bb[piece_to_bb_idx(p)].set(s);
            }
        }
        recompute_hash();
    }

    inline void to_array(int8_t grid[NUM_SQUARES]) const {
        std::memcpy(grid, sq, NUM_SQUARES);
    }

    inline BB128 blue_bb() const { return bb[0] | bb[1] | bb[2]; }
    inline BB128 red_bb() const { return bb[3] | bb[4] | bb[5]; }
    inline BB128 occ_bb() const { return blue_bb() | red_bb(); }
    inline BB128 empty_bb() const { return ~occ_bb(); }

    inline int piece_count() const {
        return occ_bb().popcount();
    }

    inline int blue_count() const { return blue_bb().popcount(); }
    inline int red_count() const { return red_bb().popcount(); }

    inline bool has_any_pseudo_move() const {
        const auto& t = tables();
        const BB128 emp = empty_bb();
        if (side == BLUE) {
            // R(0) beats S(5), P(1) beats R(3), S(2) beats P(4)
            static constexpr int PREY[3] = {5, 3, 4};
            for (int pt = 0; pt < 3; ++pt) {
                BB128 p = bb[pt];
                if (p.empty()) continue;
                BB128 targets = emp | bb[PREY[pt]];
                while (p.any()) {
                    int s = p.pop_lsb();
                    if ((t.king_attacks[s] & targets).any()) return true;
                }
            }
        } else {
            static constexpr int PREY[3] = {2, 0, 1};
            for (int pt = 0; pt < 3; ++pt) {
                BB128 p = bb[3 + pt];
                if (p.empty()) continue;
                BB128 targets = emp | bb[PREY[pt]];
                while (p.any()) {
                    int s = p.pop_lsb();
                    if ((t.king_attacks[s] & targets).any()) return true;
                }
            }
        }
        return false;
    }

    // Exact precedence matching rules.py::game_result:
    // 0 = live, +1 = BLUE win, -1 = RED win, 2 = DRAW (quiet limit)
    inline int terminal_result() const {
        if (sq[BLUE_GOAL] > 0) return BLUE;
        if (sq[RED_GOAL] < 0) return RED;
        if (!has_any_pseudo_move()) return -side;
        if (quiet >= QUIET_LIMIT) return 2;
        return 0;
    }

    // Generate pseudo-legal real action indices (0..647), sorted by from_sq then dir_idx
    inline int gen_pseudo_actions(uint16_t* out) const {
        const auto& t = tables();
        const BB128 emp = empty_bb();
        int count = 0;
        const int opp_base = (side == BLUE) ? 3 : 0;
        // R(0)->S(2), P(1)->R(0), S(2)->P(1)
        static constexpr int PREY_REL[3] = {2, 0, 1};

        BB128 own = (side == BLUE) ? blue_bb() : red_bb();
        while (own.any()) {
            int from_sq = own.pop_lsb();
            int pt = std::abs(sq[from_sq]) - 1;
            BB128 targets = emp | bb[opp_base + PREY_REL[pt]];
            for (int d = 0; d < NUM_DIRECTIONS; ++d) {
                int8_t to_sq = t.dir_to_sq[from_sq][d];
                if (to_sq >= 0 && targets.test(to_sq)) {
                    out[count++] = static_cast<uint16_t>(from_sq * NUM_DIRECTIONS + d);
                }
            }
        }
        return count;
    }

    inline int gen_legal_actions(uint16_t* out) const {
        if (terminal_result() != 0) return 0;
        return gen_pseudo_actions(out);
    }

    inline UndoInfo push_action(uint16_t action) {
        const auto& t = tables();
        int from_sq = action / NUM_DIRECTIONS;
        int dir_idx = action % NUM_DIRECTIONS;
        int to_sq = t.dir_to_sq[from_sq][dir_idx];

        int8_t mover = sq[from_sq];
        int8_t cap = sq[to_sq];
        UndoInfo u{cap, quiet, hash};

        int mover_idx = piece_to_bb_idx(mover);
        bb[mover_idx].clear(from_sq);
        bb[mover_idx].set(to_sq);
        sq[from_sq] = EMPTY;
        sq[to_sq] = mover;

        uint64_t h = hash ^ t.zobrist_piece[mover_idx][from_sq] ^ t.zobrist_piece[mover_idx][to_sq] ^ t.zobrist_side;
        if (cap != EMPTY) {
            int cap_idx = piece_to_bb_idx(cap);
            bb[cap_idx].clear(to_sq);
            h ^= t.zobrist_piece[cap_idx][to_sq];
            quiet = 0;
        } else {
            if (quiet < 255) quiet += 1;
        }
        hash = h;
        side = -side;
        ply += 1;
        return u;
    }

    inline void pop_action(uint16_t action, const UndoInfo& u) {
        const auto& t = tables();
        int from_sq = action / NUM_DIRECTIONS;
        int dir_idx = action % NUM_DIRECTIONS;
        int to_sq = t.dir_to_sq[from_sq][dir_idx];

        int8_t mover = sq[to_sq];
        int mover_idx = piece_to_bb_idx(mover);
        bb[mover_idx].clear(to_sq);
        bb[mover_idx].set(from_sq);
        sq[from_sq] = mover;
        sq[to_sq] = u.captured;

        if (u.captured != EMPTY) {
            bb[piece_to_bb_idx(u.captured)].set(to_sq);
        }

        side = -side;
        quiet = u.prev_quiet;
        ply -= 1;
        hash = u.prev_hash;
    }

    // Canonical hash from side-to-move's Blue perspective + quiet bucket
    inline uint64_t canonical_hash() const {
        const auto& t = tables();
        uint64_t h = 0;
        if (side == BLUE) {
            // XOR out side bit from incremental hash if any (for BLUE, side bit isn't set anyway)
            h = hash;
        } else {
            for (int idx = 0; idx < 6; ++idx) {
                int c_idx = (idx < 3) ? (idx + 3) : (idx - 3);
                BB128 b = bb[idx];
                while (b.any()) {
                    int s = b.pop_lsb();
                    h ^= t.zobrist_piece[c_idx][t.rotate_sq[s]];
                }
            }
        }
        // Include quiet bucket near draw limit so 200-ply clock nuances aren't aliased
        uint64_t q_bucket = (quiet >= 150) ? static_cast<uint64_t>(quiet) : 0ULL;
        uint64_t mix = q_bucket * 0x9e3779b97f4a7c15ULL;
        return h ^ mix;
    }

    // Return canonical FastBoard (always Blue to move, 180-deg rotated if Red was to move)
    inline FastBoard canonical_copy() const {
        if (side == BLUE) return *this;
        const auto& t = tables();
        FastBoard cb;
        cb.side = BLUE;
        cb.quiet = quiet;
        cb.ply = ply;
        for (int idx = 0; idx < 6; ++idx) {
            int c_idx = (idx < 3) ? (idx + 3) : (idx - 3);
            int8_t c_piece = static_cast<int8_t>(bb_idx_to_piece(c_idx));
            BB128 b = bb[idx];
            while (b.any()) {
                int s = b.pop_lsb();
                int rs = t.rotate_sq[s];
                cb.bb[c_idx].set(rs);
                cb.sq[rs] = c_piece;
            }
        }
        cb.recompute_hash();
        return cb;
    }

    // Write 11 canonical float32 planes (11 x 81 = 891 floats)
    inline void encode_features_11ch(float* out) const {
        const auto& t = tables();
        std::memset(out, 0, 11 * NUM_SQUARES * sizeof(float));

        BB128 own_bb[3];
        BB128 opp_bb[3];

        if (side == BLUE) {
            own_bb[0] = bb[0]; own_bb[1] = bb[1]; own_bb[2] = bb[2];
            opp_bb[0] = bb[3]; opp_bb[1] = bb[4]; opp_bb[2] = bb[5];
        } else {
            // Rotate 180 degrees and swap Blue/Red
            for (int pt = 0; pt < 3; ++pt) {
                BB128 ob = bb[3 + pt];
                while (ob.any()) {
                    own_bb[pt].set(t.rotate_sq[ob.pop_lsb()]);
                }
                BB128 eb = bb[pt];
                while (eb.any()) {
                    opp_bb[pt].set(t.rotate_sq[eb.pop_lsb()]);
                }
            }
        }

        // Ch 0..2: own R, P, S; Ch 3..5: opp R, P, S
        for (int pt = 0; pt < 3; ++pt) {
            BB128 b0 = own_bb[pt];
            while (b0.any()) out[pt * NUM_SQUARES + b0.pop_lsb()] = 1.0f;
            BB128 b1 = opp_bb[pt];
            while (b1.any()) out[(3 + pt) * NUM_SQUARES + b1.pop_lsb()] = 1.0f;
        }

        // Ch 6: normalized quiet clock
        float q_norm = std::min(1.0f, static_cast<float>(quiet) / static_cast<float>(QUIET_LIMIT));
        float* ch6 = out + 6 * NUM_SQUARES;
        float* ch7 = out + 7 * NUM_SQUARES;
        float* ch8 = out + 8 * NUM_SQUARES;
        for (int s = 0; s < NUM_SQUARES; ++s) {
            ch6[s] = q_norm;
            ch7[s] = 1.0f - static_cast<float>(t.dist_to_blue_goal[s]) / 8.0f;
            ch8[s] = 1.0f - static_cast<float>(t.dist_to_red_goal[s]) / 8.0f;
        }

        // Ch 9: own pieces threatened by adjacent enemy nemesis
        // Ch 10: enemy pieces capturable by adjacent own piece
        BB128 dil_own_R = dilate_king(own_bb[0]);
        BB128 dil_own_P = dilate_king(own_bb[1]);
        BB128 dil_own_S = dilate_king(own_bb[2]);

        BB128 dil_opp_R = dilate_king(opp_bb[0]);
        BB128 dil_opp_P = dilate_king(opp_bb[1]);
        BB128 dil_opp_S = dilate_king(opp_bb[2]);

        BB128 own_threatened = (own_bb[0] & dil_opp_P) | (own_bb[1] & dil_opp_S) | (own_bb[2] & dil_opp_R);
        BB128 opp_capturable = (opp_bb[2] & dil_own_R) | (opp_bb[0] & dil_own_P) | (opp_bb[1] & dil_own_S);

        float* ch9 = out + 9 * NUM_SQUARES;
        float* ch10 = out + 10 * NUM_SQUARES;
        while (own_threatened.any()) ch9[own_threatened.pop_lsb()] = 1.0f;
        while (opp_capturable.any()) ch10[opp_capturable.pop_lsb()] = 1.0f;
    }

    // Apply cyclic RPS permutation s in {0, 1, 2} where 0->(0+s)%3 (R->P->S->R for s=1)
    inline FastBoard cycle_rps(int shift) const {
        shift = ((shift % 3) + 3) % 3;
        if (shift == 0) return *this;
        FastBoard out;
        out.side = side;
        out.quiet = quiet;
        out.ply = ply;
        for (int pt = 0; pt < 3; ++pt) {
            int npt = (pt + shift) % 3;
            out.bb[npt] = bb[pt];
            out.bb[3 + npt] = bb[3 + pt];
        }
        for (int s = 0; s < NUM_SQUARES; ++s) {
            int8_t p = sq[s];
            if (p == EMPTY) {
                out.sq[s] = EMPTY;
            } else {
                int abs_pt = std::abs(p) - 1;
                int nabs = ((abs_pt + shift) % 3) + 1;
                out.sq[s] = static_cast<int8_t>((p > 0) ? nabs : -nabs);
            }
        }
        out.recompute_hash();
        return out;
    }

    // Reflect across main diagonal a1-i9
    inline FastBoard reflect_diag() const {
        const auto& t = tables();
        FastBoard out;
        out.side = side;
        out.quiet = quiet;
        out.ply = ply;
        for (int idx = 0; idx < 6; ++idx) {
            int8_t p = static_cast<int8_t>(bb_idx_to_piece(idx));
            BB128 b = bb[idx];
            while (b.any()) {
                int rs = t.reflect_sq[b.pop_lsb()];
                out.bb[idx].set(rs);
                out.sq[rs] = p;
            }
        }
        out.recompute_hash();
        return out;
    }
};

}  // namespace intransitive
