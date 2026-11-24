#pragma once

#include <array>
#include <cstdint>

namespace intransitive {

constexpr int BOARD_SIZE = 9;
constexpr int NUM_SQUARES = 81;
constexpr int NUM_DIRECTIONS = 8;
constexpr int ACTION_SIZE = NUM_SQUARES * NUM_DIRECTIONS;  // 648

constexpr int BLUE = 1;
constexpr int RED = -1;

constexpr int EMPTY = 0;
constexpr int ROCK = 1;
constexpr int PAPER = 2;
constexpr int SCISSORS = 3;

constexpr int BLUE_CORNER = 0;
constexpr int RED_CORNER = 80;
constexpr int BLUE_GOAL = 80;  // i9
constexpr int RED_GOAL = 0;    // a1

constexpr int QUIET_LIMIT = 200;

constexpr uint64_t HI_MASK = 0x1FFFFULL;  // 17 bits (64..80)

struct BB128 {
    uint64_t lo = 0;
    uint64_t hi = 0;

    constexpr BB128() = default;
    constexpr BB128(uint64_t l, uint64_t h) : lo(l), hi(h & HI_MASK) {}

    static constexpr BB128 from_square(int sq) {
        if (sq < 64) {
            return BB128{1ULL << sq, 0ULL};
        }
        return BB128{0ULL, 1ULL << (sq - 64)};
    }

    constexpr bool empty() const { return (lo | (hi & HI_MASK)) == 0ULL; }
    constexpr bool any() const { return !empty(); }

    constexpr bool test(int sq) const {
        if (sq < 64) return ((lo >> sq) & 1ULL) != 0ULL;
        return ((hi >> (sq - 64)) & 1ULL) != 0ULL;
    }

    inline void set(int sq) {
        if (sq < 64) lo |= (1ULL << sq);
        else hi |= (1ULL << (sq - 64));
    }

    inline void clear(int sq) {
        if (sq < 64) lo &= ~(1ULL << sq);
        else hi &= ~(1ULL << (sq - 64));
    }

    inline int popcount() const {
        return __builtin_popcountll(lo) + __builtin_popcountll(hi & HI_MASK);
    }

    inline int lsb() const {
        if (lo != 0ULL) return __builtin_ctzll(lo);
        return 64 + __builtin_ctzll(hi & HI_MASK);
    }

    inline int pop_lsb() {
        if (lo != 0ULL) {
            int sq = __builtin_ctzll(lo);
            lo &= (lo - 1ULL);
            return sq;
        }
        uint64_t h = hi & HI_MASK;
        int sq = 64 + __builtin_ctzll(h);
        hi = h & (h - 1ULL);
        return sq;
    }

    constexpr BB128 operator&(const BB128& o) const { return BB128{lo & o.lo, hi & o.hi}; }
    constexpr BB128 operator|(const BB128& o) const { return BB128{lo | o.lo, hi | o.hi}; }
    constexpr BB128 operator^(const BB128& o) const { return BB128{lo ^ o.lo, hi ^ o.hi}; }
    constexpr BB128 operator~() const { return BB128{~lo, (~hi) & HI_MASK}; }

    inline BB128& operator|=(const BB128& o) { lo |= o.lo; hi = (hi | o.hi) & HI_MASK; return *this; }
    inline BB128& operator&=(const BB128& o) { lo &= o.lo; hi &= o.hi; return *this; }
    constexpr bool operator==(const BB128& o) const { return lo == o.lo && (hi & HI_MASK) == (o.hi & HI_MASK); }
    constexpr bool operator!=(const BB128& o) const { return !(*this == o); }
};

// Clockwise direction deltas matching constants.py
constexpr int DIR_DR[8] = {0, 1, 1, 1, 0, -1, -1, -1};
constexpr int DIR_DF[8] = {1, 1, 0, -1, -1, -1, 0, 1};

// 180-degree rotation and main-diagonal (a1-i9) reflection mappings
constexpr int ROTATE_DIR[8] = {4, 5, 6, 7, 0, 1, 2, 3};
constexpr int REFLECT_DIR[8] = {2, 1, 0, 7, 6, 5, 4, 3};

// Piece type capture table: BEATS_TYPE[ROCK(1)] = SCISSORS(3), PAPER(2)->ROCK(1), SCISSORS(3)->PAPER(2)
constexpr int BEATS_TYPE[4] = {0, SCISSORS, ROCK, PAPER};
// Reverse: NEMESIS_TYPE[ROCK(1)] = PAPER(2), PAPER(2)->SCISSORS(3), SCISSORS(3)->ROCK(1)
constexpr int NEMESIS_TYPE[4] = {0, PAPER, SCISSORS, ROCK};

struct Tables {
    BB128 king_attacks[NUM_SQUARES]{};
    int8_t dir_to_sq[NUM_SQUARES][NUM_DIRECTIONS]{};
    int8_t sq_delta_to_dir[NUM_SQUARES][NUM_SQUARES]{};
    uint8_t dist_to_blue_goal[NUM_SQUARES]{};
    uint8_t dist_to_red_goal[NUM_SQUARES]{};
    uint8_t rotate_sq[NUM_SQUARES]{};
    uint8_t reflect_sq[NUM_SQUARES]{};
    uint16_t rotate_action[ACTION_SIZE]{};
    uint16_t reflect_action[ACTION_SIZE]{};
    uint64_t zobrist_piece[6][NUM_SQUARES]{};
    uint64_t zobrist_side = 0;

    static constexpr uint64_t splitmix64(uint64_t& x) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    Tables() {
        for (int s1 = 0; s1 < NUM_SQUARES; ++s1) {
            for (int s2 = 0; s2 < NUM_SQUARES; ++s2) {
                sq_delta_to_dir[s1][s2] = -1;
            }
        }
        for (int sq = 0; sq < NUM_SQUARES; ++sq) {
            int r = sq / BOARD_SIZE;
            int f = sq % BOARD_SIZE;
            dist_to_blue_goal[sq] = static_cast<uint8_t>(std::max(8 - r, 8 - f));
            dist_to_red_goal[sq] = static_cast<uint8_t>(std::max(r, f));
            rotate_sq[sq] = static_cast<uint8_t>((NUM_SQUARES - 1) - sq);
            reflect_sq[sq] = static_cast<uint8_t>(f * BOARD_SIZE + r);

            BB128 atk{};
            for (int d = 0; d < NUM_DIRECTIONS; ++d) {
                int nr = r + DIR_DR[d];
                int nf = f + DIR_DF[d];
                if (nr >= 0 && nr < BOARD_SIZE && nf >= 0 && nf < BOARD_SIZE) {
                    int nsq = nr * BOARD_SIZE + nf;
                    dir_to_sq[sq][d] = static_cast<int8_t>(nsq);
                    sq_delta_to_dir[sq][nsq] = static_cast<int8_t>(d);
                    atk.set(nsq);
                } else {
                    dir_to_sq[sq][d] = -1;
                }
                int act = sq * NUM_DIRECTIONS + d;
                rotate_action[act] = static_cast<uint16_t>(rotate_sq[sq] * NUM_DIRECTIONS + ROTATE_DIR[d]);
                reflect_action[act] = static_cast<uint16_t>(reflect_sq[sq] * NUM_DIRECTIONS + REFLECT_DIR[d]);
            }
            king_attacks[sq] = atk;
        }

        uint64_t seed = 0x123456789ABCDEF0ULL;
        for (int p = 0; p < 6; ++p) {
            for (int sq = 0; sq < NUM_SQUARES; ++sq) {
                zobrist_piece[p][sq] = splitmix64(seed);
            }
        }
        zobrist_side = splitmix64(seed);
    }
};

inline const Tables& tables() {
    static const Tables t;
    return t;
}

// Helper: map signed piece {-3,-2,-1, +1,+2,+3} to bitboard index 0..5:
// 0=B_R(+1), 1=B_P(+2), 2=B_S(+3), 3=R_R(-1), 4=R_P(-2), 5=R_S(-3)
inline constexpr int piece_to_bb_idx(int piece) {
    return (piece > 0) ? (piece - 1) : (3 + (-piece) - 1);
}

inline constexpr int bb_idx_to_piece(int idx) {
    return (idx < 3) ? (idx + 1) : -(idx - 2);
}

// Neighbor union of all bits set in `b` (king-move dilation)
inline BB128 dilate_king(const BB128& b) {
    const auto& t = tables();
    BB128 out{};
    BB128 tmp = b;
    while (tmp.any()) {
        int sq = tmp.pop_lsb();
        out |= t.king_attacks[sq];
    }
    return out;
}

}  // namespace intransitive
