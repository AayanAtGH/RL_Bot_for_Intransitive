# Intransitive (RPS V6) — C++17 Game Engine, 4-Piece Tablebase & AlphaZero v2 RL Bot

A high-performance implementation of the abstract strategy war game **Intransitive** (originally by *webgoatguy*, hosted on *meaf.us/rps2/*), featuring:
- A unified **C++17 128-bit bitboard core** (`ctypes` zero-copy bridge, ~500× faster than pure Python).
- An exact **Syzygy-style 2–4 piece WDL + DTZ Endgame Tablebase** generator and memory-mapped prober (`47` canonical slices, `437.5 MB`, `maxDTZ = 50`, verified 100% against an independent brute-force Python solver).
- **AlphaZero v2** (tabula-rasa RL): a **6 × 128 SE-ResNet** (`1.93M` params) with KataGo-style Global Pooling, a spatially equivariant convolutional `(8, 9, 9)` policy head, a 3-class WDL value head, and **11-channel 6× symmetry ($D_1 \times C_3$)** features.
- **Deep Vectorized C++ MCTS**: Dynamic FPU, subtree reuse, Zobrist NN evaluation cache, 2-fold search-path repetition avoidance, **MCTS-Solver**, a **3-ply exact corner-sprint prover**, adaptive PV-depth extension, and `128`-game concurrent batched self-play with Playout Cap Randomization.
- Baseline bots (`RandomBot`, `GreedyBot`, `Minimax`), an Arena Elo benchmark, UCI stdio bridge, and a **meaf.us-style Tkinter GUI**.

---

## Game Rules Summary
- **Board**: $9 \times 9$ grid (`a1` at bottom-left, `i9` at top-right).
- **Pieces**: Rock (`R`/`r`), Paper (`P`/`p`), Scissors (`S`/`s`). Blue (uppercase, 10 pieces) moves first aiming for `i9`; Red (lowercase, 10 pieces) moves second aiming for `a1`.
- **Movement**: Any piece moves 1 step in any of 8 directions (chess King move).
- **Captures**: Intransitive hierarchy — Rock beats Scissors (`R > S`), Scissors beats Paper (`S > P`), Paper beats Rock (`P > R`).
- **Winning & Draw**:
  1. **Corner Reach**: Blue reaches `i9` or Red reaches `a1`.
  2. **Stalemate / Elimination**: Opponent has no legal moves on their turn.
  3. **Draw**: 200 quiet plies (half-moves without a capture).

---

## Setup & Native Build

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Compile the C++17 Native Core & Tablebase Generator
Requires `g++` with C++17 support (works on Windows via MSYS2 UCRT64 `g++` and natively on Linux / Google Colab):
```bash
python scripts/build_cpp.py --target all
```
This builds:
- `intransitive/native/libintransitive_core.dll` (Windows) or `.so` (Linux)
- `bin/tb_gen.exe` (Windows) or `bin/tb_gen` (Linux)

*(Use `--portable` on cloud/Colab builds to omit `-march=native`.)*

### 3. Generate the 2–4 Piece Endgame Tablebase (Optional, Recommended)
```bash
./bin/tb_gen --max-pieces 4 --out data/tablebase
```
- Solves all **47 canonical slices** (`3` for 2-piece, `12` for 3-piece, `32` for 4-piece) via multi-threaded retrograde BFS in ~5–9 minutes on an 8-thread CPU (< 250 MB peak RAM).
- Produces **94 files** (`47 .wdl` + `47 .dtz`, **437.5 MB** total) in `data/tablebase/`.
- Automatically memory-mapped and used by MCTS, self-play, CLI, and GUI whenever present.

---

## Quick Commands

### Run Full Test Suite (31 Tests)
```bash
pytest tests/ -v
```
Includes a 2,000-game C++ vs. Python rules/features fuzz test, 6× symmetry checks, exhaustive 113,778-state 2-piece tablebase exactness verification, 500 automated 2–4 piece TB-vs-TB games, 3-ply corner-sprint prover tests, FPU/PV-depth & repetition tests, and batched self-play tests.

### Train AlphaZero v2 (Tabula-Rasa Self-Play)
```bash
python -m intransitive.rl.trainer \
  --iterations 20 \
  --blocks 6 --channels 128 \
  --parallel-games 128 --sp-batch 256 \
  --full-sims 320 --fast-sims 64 \
  --tb-dir data/tablebase
```
Checkpoints save to `checkpoints/alphazero_v2_latest.pt` and `checkpoints/alphazero_v2_best.pt` (legacy v1 checkpoints remain supported via `load_checkpoint()`).

### Train on Google Colab (T4 GPU)
Open [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) in Google Colab (T4 GPU runtime) or run:
```bash
python scripts/colab_train.py \
  --drive-dir /content/drive/MyDrive/intransitive_az \
  --duration-minutes 480 \
  --build-tb-if-missing
```
Checkpoints and `replay_buffer.npz` are persisted to Google Drive so runs resume seamlessly across sessions.

### Play in the Tkinter GUI (meaf.us Style)
```bash
python play_gui.py
```
Displays the live win-probability bar, PV depth, and exact tablebase evaluations (`TB: Win in N` / `TB: Draw`) at $\le 4$ pieces.

### Play in the Terminal
```bash
python -m intransitive.cli.play --blue human --red alphazero --simulations 400
```

### Run Benchmark Arena (v2 vs. v1 & Baselines)
```bash
python -m intransitive.arena.benchmark --games 10 --simulations 100
```

---

## Project Layout
```text
cpp/                    C++17 bitboard core, retrograde TB generator/prober, tactical prover & batched MCTS
intransitive/
  core/                 Board, rules, move notation, and Python TablebaseManager
  env/                  Canonical 7-ch (v1) & 11-ch (v2) encoders, 6x D1xC3 symmetries, Gymnasium env
  native/               Zero-copy ctypes bridge to libintransitive_core
  rl/                   IntransitiveNetV1/V2, deep MCTS, compact 6x ReplayBuffer, batched self-play & trainer
  baselines/            Random, Greedy, and Alpha-Beta Minimax players
  arena/                Match runner & Elo round-robin benchmark
  cli/                  Tkinter GUI, terminal play, and UCI stdio bridge
scripts/                build_cpp.py and colab_train.py
notebooks/              train_colab.ipynb (Google Colab T4 workflow)
tests/                  Rules, FEN, MCTS v1/v2, native fuzz, and tablebase verification suites
```
