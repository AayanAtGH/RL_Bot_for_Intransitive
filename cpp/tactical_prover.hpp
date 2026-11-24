#pragma once

#include <cstdint>
#include "board.hpp"

namespace intransitive {

// Bitmasks of squares within Chebyshev distance <= 2 of Blue goal (80) and Red goal (0)
inline BB128 blue_goal_zone2() {
    static const BB128 mask = []() {
        BB128 m{};
        const auto& t = tables();
        for (int s = 0; s < NUM_SQUARES; ++s) {
            if (t.dist_to_blue_goal[s] <= 2) m.set(s);
        }
        return m;
    }();
    return mask;
}

inline BB128 red_goal_zone2() {
    static const BB128 mask = []() {
        BB128 m{};
        const auto& t = tables();
        for (int s = 0; s < NUM_SQUARES; ++s) {
            if (t.dist_to_red_goal[s] <= 2) m.set(s);
        }
        return m;
    }();
    return mask;
}

inline BB128 blue_goal_zone1() {
    return tables().king_attacks[BLUE_GOAL];
}

inline BB128 red_goal_zone1() {
    return tables().king_attacks[RED_GOAL];
}

// Fast O(1) check whether `side_to_check` has an immediate 1-step move reaching its goal corner.
inline bool has_immediate_corner_win_for(const FastBoard& b, int side_to_check) {
    const auto& t = tables();
    int goal = (side_to_check == BLUE) ? BLUE_GOAL : RED_GOAL;
    BB128 own = (side_to_check == BLUE) ? b.blue_bb() : b.red_bb();
    BB128 adj = t.king_attacks[goal] & own;
    if (adj.empty()) return false;

    int8_t victim = b.sq[goal];
    if (victim == EMPTY) return true;
    // Own piece on own goal shouldn't happen on live board, but guard anyway
    if ((victim > 0) == (side_to_check > 0)) return false;

    int victim_type = std::abs(victim);
    while (adj.any()) {
        int s = adj.pop_lsb();
        int att_type = std::abs(b.sq[s]);
        if (BEATS_TYPE[att_type] == victim_type) return true;
    }
    return false;
}

inline bool has_immediate_corner_win(const FastBoard& b) {
    return has_immediate_corner_win_for(b, b.side);
}

// Find any immediate 1-ply win action (corner reach OR leaving opponent with 0 moves).
// Returns real action index (0..647) or -1.
inline int find_immediate_win_action(FastBoard& b) {
    const auto& t = tables();
    uint16_t acts[80];
    int n = b.gen_pseudo_actions(acts);
    if (n == 0) return -1;

    int own_goal = (b.side == BLUE) ? BLUE_GOAL : RED_GOAL;
    for (int i = 0; i < n; ++i) {
        int from_sq = acts[i] / NUM_DIRECTIONS;
        int to_sq = t.dir_to_sq[from_sq][acts[i] % NUM_DIRECTIONS];
        if (to_sq == own_goal) return static_cast<int>(acts[i]);
    }

    for (int i = 0; i < n; ++i) {
        UndoInfo u = b.push_action(acts[i]);
        bool no_opp_moves = !b.has_any_pseudo_move();
        b.pop_action(acts[i], u);
        if (no_opp_moves) return static_cast<int>(acts[i]);
    }
    return -1;
}

// Exact bounded 3-ply corner-sprint prover:
// Checks if `b.side` has a move m1 such that:
//   - m1 does not let opponent win in 1 ply, AND
//   - for EVERY legal opponent reply m2, `b.side` has an immediate 1-ply corner win on ply 3!
// Returns proven winning real action index (0..647) or -1. Zero heap allocations.
inline int prove_corner_win_3ply(FastBoard& b) {
    if (b.quiet >= QUIET_LIMIT - 2) {
        // Near 200-quiet-ply boundary, quiet moves might hit draw before ply 3;
        // let standard terminal check / MCTS handle it safely.
        return -1;
    }

    int orig_side = b.side;
    BB128 own = (orig_side == BLUE) ? b.blue_bb() : b.red_bb();
    BB128 z2 = (orig_side == BLUE) ? blue_goal_zone2() : red_goal_zone2();
    if ((own & z2).empty()) return -1;

    BB128 z1 = (orig_side == BLUE) ? blue_goal_zone1() : red_goal_zone1();
    const auto& t = tables();

    uint16_t m1_acts[80];
    int n1 = b.gen_pseudo_actions(m1_acts);
    if (n1 == 0) return -1;

    uint16_t m2_acts[80];

    for (int i = 0; i < n1; ++i) {
        uint16_t a1 = m1_acts[i];
        int from1 = a1 / NUM_DIRECTIONS;
        int to1 = t.dir_to_sq[from1][a1 % NUM_DIRECTIONS];

        // Only consider moves into zone1 or moves when we already have a piece in zone1
        BB128 own_after_z1 = ((orig_side == BLUE) ? b.blue_bb() : b.red_bb()) & z1;
        if (own_after_z1.empty() && !z1.test(to1)) {
            continue;
        }

        UndoInfo u1 = b.push_action(a1);
        if (!b.has_any_pseudo_move()) {
            // Immediate no-moves win on ply 1!
            b.pop_action(a1, u1);
            return static_cast<int>(a1);
        }
        if (has_immediate_corner_win(b)) {
            // Opponent wins immediately on ply 2!
            b.pop_action(a1, u1);
            continue;
        }

        // Fast necessary condition: `orig_side` must now have at least one piece in zone1
        BB128 cur_own = (orig_side == BLUE) ? b.blue_bb() : b.red_bb();
        if ((cur_own & z1).empty()) {
            b.pop_action(a1, u1);
            continue;
        }

        int n2 = b.gen_pseudo_actions(m2_acts);
        bool all_replies_lose = (n2 > 0);
        for (int j = 0; j < n2; ++j) {
            uint16_t a2 = m2_acts[j];
            UndoInfo u2 = b.push_action(a2);

            // Check if opponent's move m2 escaped the corner threat or left orig_side with no moves
            bool orig_wins_ply3 = false;
            if (b.has_any_pseudo_move() && b.quiet < QUIET_LIMIT) {
                orig_wins_ply3 = has_immediate_corner_win(b);
            }
            b.pop_action(a2, u2);

            if (!orig_wins_ply3) {
                all_replies_lose = false;
                break;
            }
        }

        b.pop_action(a1, u1);
        if (all_replies_lose) {
            return static_cast<int>(a1);
        }
    }
    return -1;
}

}  // namespace intransitive
