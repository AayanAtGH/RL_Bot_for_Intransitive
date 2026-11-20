# Intransitive (RPS V6) — Game Engine & AlphaZero RL Bot

A complete Python implementation of the abstract strategy war game **Intransitive** (originally by *webgoatguy* and hosted on *meaf.us/rps2/*), along with baseline bots (Random, Greedy Heuristic, Alpha-Beta Minimax) and an **AlphaZero Reinforcement Learning engine** (MCTS + Dual-Headed Policy-Value ResNet).

## Game Rules Summary
- **Board**: $9 \times 9$ grid (`a1` at bottom-left, `i9` at top-right).
- **Pieces**: Rock (`R`/`r`), Paper (`P`/`p`), Scissors (`S`/`s`). Blue (uppercase) moves first from `a1`; Red (lowercase) moves second from `i9`.
- **Movement**: Any piece moves 1 step in any of 8 directions (king move in chess).
- **Captures**: Intransitive hierarchy — Rock beats Scissors (`R > S`), Scissors beats Paper (`S > P`), Paper beats Rock (`P > R`).
- **Winning**:
  1. **Corner Reach**: Blue reaches `i9` or Red reaches `a1`.
  2. **Stalemate / Elimination**: Opponent has no legal moves on their turn.
  3. **Draw**: 200 quiet plies (half-moves without a capture).

## Quick Commands

### Run Unit Tests
```bash
pytest tests/
```

### Train AlphaZero Self-Play (e.g. 2 Epochs)
```bash
python -m intransitive.rl.trainer --epochs 2 --episodes 4 --simulations 30
```

### Run Benchmark Arena
```bash
python -m intransitive.arena.benchmark --games 10
```

### Play Interactively in Terminal
```bash
python -m intransitive.cli.play --opponent greedy
```
