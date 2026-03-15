# Blokus AlphaZero — Developer Guide

This file documents everything a developer (or AI agent) needs to know to work effectively in this repo.

## Project Overview

Building an AlphaZero-inspired self-play RL agent for [Blokus](https://en.wikipedia.org/wiki/Blokus). The game engine is complete (Phase 1). Future phases will add neural networks, MCTS, self-play training, and interactive play.

The core engine is in `blokus/engine/`. It provides an immutable `GameState` class designed for tree search and RL. The original codebase is preserved in `legacy/` for reference but is not used by the new engine.

## Repo Structure

```
blokus/                          # Main Python package
  __init__.py
  engine/                        # Game engine (Phase 1 — complete)
    __init__.py                  # Re-exports key symbols (GameState, encode/decode_action, etc.)
    game_state.py                # Core GameState class, legal move gen, action encoding
    piece.py                     # Piece class: (x,y) tuples, flip/rotate/translate, orientations
    board.py                     # Board class: numpy grid, move validation, pygame/seaborn rendering
    heuristics.py                # Flood-fill territory estimation heuristics
  agents/                        # RL agents (Phase 2+ — empty)
    __init__.py
  mcts/                          # Monte Carlo Tree Search (Phase 3 — empty)
    __init__.py
  nn/                            # Neural network models (Phase 2 — empty)
    __init__.py
scripts/                         # Utility scripts (not part of the package)
  generate_video.py              # Plays random games and saves animated GIFs to output/
tests/                           # pytest test suite
  __init__.py
  test_game_state.py             # 61 tests for the game engine
data/                            # Data files (piece definitions, future: trained models)
  blokus_pieces.pkl              # 21 standard Blokus pieces (pickled Piece objects)
legacy/                          # Original codebase — reference only, not imported
  Game.py                        # Original game orchestrator (turns, scoring)
  Player.py                      # Original player logic (random, heuristic, manual strategies)
  RL_agent.py                    # Legacy Q-learning attempt with ResNet (non-functional)
  BlokusEnv_gym.py               # Stub OpenAI Gym environment (not implemented)
  Blokus_RL_gym.py               # Stub Gym test script
  heuristic_buffer.cpkl          # Saved replay buffer from heuristic training
  current_buffer.cpkl            # Saved replay buffer snapshot
  Piece Images.JPG               # Reference image of all Blokus pieces
output/                          # Generated artifacts (gitignored — regenerate via scripts)
slurm/                           # VACC Slurm job scripts
  train_alphazero.sh             # GPU training (gpu-preempt, 1 GPU, 10 CPUs)
configs/                         # Training configs (Phase 4 — empty)
readme_ims/                      # Images used in README.md
pyproject.toml                   # Project metadata, dependencies, pytest config
CLAUDE.md                        # This file
README.md                        # User-facing project overview
.gitignore                       # Ignores output/, __pycache__/, *.egg-info/, etc.
```

## Build / Run / Test

### Installation

```bash
pip install -e ".[all]"     # Editable install with all optional deps (viz + dev)
pip install -e "."          # Core only: numpy, Pillow
pip install -e ".[dev]"     # Core + pytest
pip install -e ".[viz]"     # Core + matplotlib, seaborn, pygame
```

Python 3.9+ required. Defined in `pyproject.toml`.

### Running tests

```bash
python -m pytest tests/ -v
```

61 tests covering: piece loading, action encoding, first move rules, move validation (overlap, diagonal adjacency, orthogonal adjacency), state transitions (immutability, passes, scores), terminal detection, both game modes, NN state shape, rendering, edge cases.

**Always run tests before committing.**

### Running a game

```bash
# Quick test from the command line
python -c "from blokus.engine import play_random_game; play_random_game('standard', seed=42, verbose=True)"
```

### Generating visualizations

```bash
python scripts/generate_video.py
# Creates: output/game_standard.gif, output/game_dual.gif
```

### Importing in code

```python
# Preferred: import from the engine subpackage (re-exports key symbols)
from blokus.engine import GameState, encode_action, decode_action, play_random_game

# For constants and utilities, import directly from the module
from blokus.engine.game_state import (
    BOARD_SIZE, NUM_PIECES, MAX_ORIENTATIONS, NUM_COLORS,
    ACTION_SPACE_SIZE, COLOR_CORNERS, COLOR_NAMES, COLOR_RGB,
    load_pieces, clear_piece_cache,
)
```

## Game Engine Architecture

### GameState — the core class

`GameState` is an **immutable** game state object. All transitions return a **new** state; the original is never mutated. This makes it safe for MCTS and parallel search.

**Key attributes** (access directly, all are public):
- `board` — `np.ndarray` shape `(20, 20)`, dtype `int8`. Values: 0=empty, 1-4=colors.
- `pieces_remaining` — tuple of 4 `frozenset`s, each containing remaining piece IDs (0-20) for that color.
- `current_color` — int 0-3, which color moves next.
- `consecutive_passes` — int, how many colors have passed in a row (game ends at 4).
- `game_mode` — `'standard'` or `'dual'`.
- `pieces` — list of 21 `PieceInfo` objects (shared reference, never copied).

**Key methods:**

| Method | Description |
|---|---|
| `new_game(mode, pickle_path=None)` | Factory: create a fresh game |
| `get_legal_actions()` | List of valid action ints for current color |
| `get_legal_actions_mask()` | Binary `ndarray` of shape `(67200,)` |
| `apply_action(action)` | Place a piece, return new state (resets passes to 0) |
| `pass_turn()` | Skip turn, return new state (increments consecutive_passes) |
| `is_terminal()` | True if 4 consecutive passes or any color placed all 21 pieces |
| `get_current_player()` | Current color index (0-3) |
| `get_current_agent()` | Current agent index (0-3 standard, 0-1 dual) |
| `get_num_agents()` | 4 (standard) or 2 (dual) |
| `get_scores()` | `{color_idx: num_squares_on_board}` |
| `get_rewards()` | `{agent_idx: float in {-1, 0, +1}}` win/loss/draw rewards |
| `get_nn_state()` | `(5, 20, 20)` float32 tensor for NN input |
| `render_text()` | ASCII board string |
| `render_image(cell_size=24)` | RGB `(H, W, 3)` uint8 numpy array |
| `copy()` | Independent copy (new board array, shared piece data) |

### Game modes

**Standard (1v1v1v1):** 4 agents, agent `i` controls color `i`. `get_current_agent() == get_current_player()`.

**Dual (1v1):** 2 agents. Agent 0 controls colors 0 and 2; Agent 1 controls colors 1 and 3. Turn order still cycles 0→1→2→3. `get_current_agent() == current_color % 2`. Rewards combine both colors per agent.

Both modes use the same 20x20 board with 4 colors and 21 pieces per color.

### Action encoding

Flat integer: `piece_id * 3200 + orientation_id * 400 + row * 20 + col`

- `piece_id`: 0-20 (21 standard Blokus pieces)
- `orientation_id`: 0-7 (up to 8 unique orientations per piece; many pieces have fewer)
- `row`, `col`: 0-19 (translation offset — top-left of piece after shift_min)
- Total space: 67,200. Most are invalid at any state. Use `get_legal_actions()` or `get_legal_actions_mask()`.

Use `encode_action()` / `decode_action()` to convert between flat ints and `(piece_id, orientation_id, row, col)` tuples.

### Legal move generation algorithm

1. **Anchor points**: Empty cells diagonally adjacent to an existing same-color piece. For the first move: the assigned board corner cell.
2. **Forbidden mask**: Cells orthogonally adjacent to same-color pieces (numpy dilation).
3. **Candidate enumeration**: For each remaining piece/orientation, align each piece cell with each anchor point to get candidate translations.
4. **Validation**: Check all cells are in-bounds, not overlapping any piece, not on the forbidden mask.

Performance: ~0.3s for a full random game (~60 moves). Legal move gen is recomputed from scratch each call (no incremental state). This is fast enough for RL training but may need optimization for large-scale MCTS.

### Piece data and the pickle file

`data/blokus_pieces.pkl` contains 21 `Piece` objects serialized from the original codebase. The `Piece` class is in `blokus/engine/piece.py`. When unpickling, a custom `_PieceUnpickler` redirects the old module path (`Piece.Piece`) to the new location (`blokus.engine.piece.Piece`).

Each piece has:
- `occupied`: list of (row, col) tuples for squares the piece covers
- `corners`: "endpoint" cells of the piece shape
- `adjacents`: orthogonally adjacent empty cells
- `diag_adjacents`: diagonally adjacent empty cells (excluding adjacents)
- `flip()`, `rotate()`, `translate()`, `shift_min()`: geometric transforms (mutate in place)
- `get_orientations()`: returns all unique orientations (up to 8: 4 rotations x 2 flips, deduplicated)

At load time, `load_pieces()` calls `get_orientations()` on each piece and stores the results as lightweight `OrientationData` objects (just the occupied tuple and bounding box). This is cached globally.

### Neural network state representation

`get_nn_state()` returns a `(5, 20, 20)` float32 array:

| Channel | Content |
|---|---|
| 0 | Current player's pieces (binary) |
| 1 | Next player's pieces in turn order (binary) |
| 2 | Player +2 in turn order (binary) |
| 3 | Player +3 (previous) in turn order (binary) |
| 4 | Empty squares (binary) |

The representation is always from the perspective of the player to move. This means the NN sees a consistent view regardless of which color it's playing.

### Board conventions

- Board is a `(20, 20)` numpy array, dtype `int8`.
- Value 0 = empty. Values 1-4 = the four colors.
- Color index `i` (0-based, used in GameState) maps to board value `i + 1`.
- Corner assignments: color 0 = (0,0), color 1 = (19,19), color 2 = (0,19), color 3 = (19,0).
- Color names: 0=Yellow, 1=Blue, 2=Red, 3=Green.

### Rendering

- `render_text()` — ASCII art with `.` for empty, `Y`/`B`/`R`/`G` for colors.
- `render_image(cell_size=24)` — RGB numpy array suitable for PIL/matplotlib display or video creation.
- `scripts/generate_video.py` — Uses `render_image()` + PIL to generate animated GIFs frame by frame.
- Legacy rendering (in `blokus/engine/board.py`): `display_pygame()`, `display2()` (seaborn), `display()` (print numpy).

## Blokus Rules Reference

1. **Board**: 20x20 grid, 4 colors.
2. **Pieces**: 21 polyominoes per color (1 monomino, 1 domino, 2 triominoes, 5 tetrominoes, 12 pentominoes = 89 total squares).
3. **First move**: Must cover the player's assigned corner cell.
4. **Subsequent moves**: At least one cell of the new piece must be **diagonally adjacent** to an existing same-color cell. No cell may be **orthogonally adjacent** to same-color. No overlap with any color.
5. **Passing**: If a color has no legal moves, it passes. A color that passes may become able to play again later (if other players' moves open new adjacencies — though this is rare in practice).
6. **Game end**: All 4 colors pass consecutively, or any color places all 21 pieces.
7. **Scoring**: Number of squares placed on the board per color (max 89 per color).

## Development Conventions

### Code style
- Use **type hints** in all new code
- Add comments for non-obvious logic; don't over-comment obvious code
- Prefer **immutable data structures** (frozenset, tuple) in GameState internals
- Avoid `deepcopy` in hot paths — use `ndarray.copy()` for boards, share immutable refs

### Imports
- Always import from `blokus.engine` or `blokus.engine.game_state`, never via `sys.path` hacks
- Scripts in `scripts/` may add repo root to `sys.path` for convenience (see `generate_video.py`)
- Tests import directly from package paths

### Git practices
- Clear, descriptive commit messages
- Run `python -m pytest tests/ -v` before every commit
- Keep generated files out of git (they go in `output/`, which is gitignored)

### Where to put new code
- **New RL agents** → `blokus/agents/`
- **Neural networks** → `blokus/nn/`
- **MCTS implementation** → `blokus/mcts/`
- **Training scripts** → `scripts/`
- **Training configs** → `configs/`
- **Trained models / checkpoints** → `data/` (consider gitignoring large files)
- **Tests** → `tests/` (one test file per module, named `test_<module>.py`)

### Performance notes
- Legal move gen is ~5ms per call, ~0.3s per full game. Good enough for training; may need C extension for large-scale MCTS.
- `apply_action()` copies the board array (~1.6KB for 20x20 int8). Piece data is a shared reference (never copied).
- `pass_turn()` does NOT copy the board (just wraps same array in new GameState) — safe because the board is unchanged.
- Piece data is loaded once and cached globally. Call `clear_piece_cache()` to force reload (useful in tests).

## Neural Network Architecture

`blokus/nn/network.py` — AlphaZero-style dual-headed ResNet.

**Input:** `(batch, 5, 20, 20)` from `get_nn_state()` + `(batch, 84)` piece-remaining vector + `(batch, 67200)` legal actions mask.

**Architecture:**
- Initial conv: 5 → 128 filters (3×3), batch norm, ReLU
- 5 residual blocks (configurable): each = two 3×3 convs with batch norm + skip connection
- **Policy head (convolutional):** 128 → 168 (1×1) conv + batch norm + piece-remaining bias (84 → 168 FC, broadcast over spatial dims) + ReLU → flatten to 67,200 logits → masked log-softmax
- **Value head:** 128 → 1 (1×1) conv + batch norm + ReLU → flatten + concat piece-remaining → FC(484, 256) → ReLU → FC(256, 1) → tanh

**Key function:** `make_pieces_remaining_vector(state)` builds the 84-dim binary vector (21 pieces × 4 colors in turn order).

**Parameter count:** ~1.6M (5 blocks, 128 channels).

### Importing

```python
from blokus.nn import BlokusNetwork, make_pieces_remaining_vector
from blokus.mcts import MCTS
from blokus.agents import AlphaZeroAgent, self_play_game
```

## MCTS

`blokus/mcts/mcts.py` — PUCT-based Monte Carlo Tree Search (AlphaZero-style).

- Selection: PUCT formula `a = argmax(Q + c_puct * P * sqrt(N_parent) / (1 + N))`
- Expansion: NN evaluation for (policy, value) at leaf nodes
- Backup: propagate value up tree, negated for opponents
- Root noise: Dirichlet noise for exploration
- Temperature-based action selection from visit counts

## Self-Play & Training

- `blokus/agents/alpha_zero.py` — `AlphaZeroAgent` (NN + MCTS) and `self_play_game()` function
- `scripts/train.py` — Full training loop with multi-process self-play and W&B logging

### Running training

```bash
# Quick test run (small network, few games, sequential)
python scripts/train.py --iterations 5 --games-per-iter 2 --sims 25 --num-blocks 2 --channels 32 --num-workers 1

# Full training with 4 parallel self-play workers + W&B logging
python scripts/train.py --iterations 100 --games-per-iter 10 --sims 100 --num-workers 4 --wandb

# Resume from checkpoint
python scripts/train.py --iterations 50 --resume data/checkpoints/checkpoint_0100.pt --wandb
```

### Multi-process self-play

`--num-workers N` runs N self-play games in parallel using `torch.multiprocessing` (spawn). Each worker gets its own model copy. Default is 4 workers. Use `--num-workers 1` for sequential mode.

### W&B logging

`--wandb` enables Weights & Biases tracking. Logs per-iteration losses, throughput (games/hr, examples/hr), cumulative totals, and all config params. Requires `pip install wandb`. Project name configurable via `--wandb-project`.

### VACC training

```bash
# Default: 100 iters, 10 games/iter, 100 sims, 4 workers, W&B
sbatch slurm/train_alphazero.sh

# Custom args
TRAIN_ARGS="--iterations 200 --sims 200 --num-workers 8" sbatch slurm/train_alphazero.sh
```

Checkpoints saved to `data/checkpoints/` (gitignored).

## Development Roadmap

| Phase | Description | Status |
|---|---|---|
| **1. Engine** | GameState, two game modes, tests, visualization | **Done** |
| **2. Neural Network** | ResNet with convolutional policy + value heads | **Done** |
| **3. MCTS** | PUCT-based tree search with NN-guided evaluation | **Done** |
| **4. Self-Play** | Data generation loop + training pipeline | **Done** |
| **5. Interactive** | Human vs agent play mode | Planned |
