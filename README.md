# Blokus AlphaZero

An AlphaZero-inspired self-play reinforcement learning agent for [Blokus](https://en.wikipedia.org/wiki/Blokus), the classic abstract strategy board game where players compete to place as many of their colored pieces on a shared grid as possible.

Forked from [DerekGloudemans/Blokus-Reinforcement-Learning](https://github.com/DerekGloudemans/Blokus-Reinforcement-Learning).

![](readme_ims/34.png)

## About Blokus

Blokus is played on a 20x20 grid with 4 colors. Each color has 21 polyomino pieces (1 to 5 squares each). Players take turns placing one piece at a time. The placement rules are:

1. The piece must fit entirely within the board
2. It must not overlap any existing piece (of any color)
3. It must touch at least one existing piece of the **same** color **diagonally**
4. It must **not** touch any existing piece of the same color **orthogonally** (edge-to-edge)
5. On the first move, the piece must cover the player's assigned board corner

The game ends when no player can place a piece. Score = number of squares placed on the board.

## Project Status

| Component | Status |
|---|---|
| Game engine (`blokus.engine`) | Done |
| Two game modes (standard + dual-color) | Done |
| Test suite (61 tests) | Done |
| Visualization (animated GIFs) | Done |
| Neural network (`blokus.nn`) | Not started |
| MCTS (`blokus.mcts`) | Not started |
| Self-play training pipeline | Not started |
| Interactive human play | Not started |

## Installation

Requires **Python 3.9+**.

```bash
# Clone the repo
git clone <repo-url>
cd Blokus-Reinforcement-Learning

# Install in editable mode with all dependencies
pip install -e ".[all]"

# Or install only what you need:
pip install -e "."          # Core only (numpy, Pillow)
pip install -e ".[dev]"     # Core + pytest
pip install -e ".[viz]"     # Core + matplotlib, seaborn, pygame
```

## Quick Start

### Run a random game

```python
from blokus.engine import GameState, play_random_game

# Play a full random game and print results
final_state, history = play_random_game('standard', seed=42, verbose=True)
print(final_state.get_scores())   # {0: 61, 1: 75, 2: 55, 3: 62}
print(final_state.get_rewards())  # normalized rewards per agent
```

### Custom game loop

```python
from blokus.engine import GameState

state = GameState.new_game('standard')  # or 'dual'

while not state.is_terminal():
    legal_actions = state.get_legal_actions()
    if not legal_actions:
        state = state.pass_turn()
        continue

    # Your agent picks an action from the legal set
    action = your_agent.choose(state, legal_actions)
    state = state.apply_action(action)

scores = state.get_scores()    # {color_idx: num_squares}
rewards = state.get_rewards()  # {agent_idx: float in [-1, 1]}
```

### Run tests

```bash
python -m pytest tests/ -v
```

### Generate visualization GIFs

```bash
python scripts/generate_video.py
# Output: output/game_standard.gif, output/game_dual.gif
```

## Game Modes

### Standard (1v1v1v1)

4 agents, each controlling one color. Each agent sees the board and decides moves for their single color. Standard Blokus scoring: most squares on the board wins.

- Agent 0 = Color 0 (Yellow), starting corner (0, 0)
- Agent 1 = Color 1 (Blue), starting corner (19, 19)
- Agent 2 = Color 2 (Red), starting corner (0, 19)
- Agent 3 = Color 3 (Green), starting corner (19, 0)

### Dual-Color (1v1)

2 agents on a standard 20x20 board with all 4 colors. Each agent controls two colors:

- **Agent 0** controls Colors 0 and 2 (Yellow and Red)
- **Agent 1** controls Colors 1 and 3 (Blue and Green)

Turn order still cycles through all 4 colors: 0 -> 1 -> 2 -> 3 -> 0 -> ...
Each agent makes decisions when any of their colors is active. Rewards combine the scores of both colors controlled by each agent.

This mode is useful for training since it reduces the number of agents to 2 while keeping the full 4-color board dynamics.

## GameState API

| Method | Returns | Description |
|---|---|---|
| `GameState.new_game(mode)` | `GameState` | Create a fresh game (`'standard'` or `'dual'`) |
| `state.get_legal_actions()` | `List[int]` | All legal action indices for the current color |
| `state.get_legal_actions_mask()` | `ndarray` | Binary mask over full action space (67,200) |
| `state.apply_action(action)` | `GameState` | Place a piece; returns **new** state |
| `state.pass_turn()` | `GameState` | Pass (no legal moves); returns **new** state |
| `state.is_terminal()` | `bool` | True if game is over |
| `state.get_current_player()` | `int` | Current color index (0-3) |
| `state.get_current_agent()` | `int` | Current agent index (mode-aware) |
| `state.get_num_agents()` | `int` | 4 (standard) or 2 (dual) |
| `state.get_scores()` | `Dict[int, int]` | Squares placed per color |
| `state.get_rewards()` | `Dict[int, float]` | Normalized reward per agent in [-1, 1] |
| `state.get_nn_state()` | `ndarray` | (5, 20, 20) float32 tensor for neural network |
| `state.render_text()` | `str` | ASCII board representation |
| `state.render_image(cell_size)` | `ndarray` | RGB (H, W, 3) uint8 image of the board |
| `state.copy()` | `GameState` | Independent copy of the state |
| `encode_action(pid, oid, r, c)` | `int` | Encode move as flat integer |
| `decode_action(action)` | `tuple` | Decode to (piece_id, orientation_id, row, col) |

All state transitions (`apply_action`, `pass_turn`) are **immutable** — they return a new `GameState` and never modify the original. This makes them safe for tree search algorithms like MCTS.

## Action Space

Actions are encoded as flat integers: `piece_id * 3200 + orientation_id * 400 + row * 20 + col`

- **21 pieces** (indexed 0-20), sizes 1 to 5 squares
- **Up to 8 orientations** per piece (4 rotations x 2 flips, deduplicated)
- **20x20 board positions** (row, col translation offset)
- **Total action space: 67,200** — most are invalid at any given state

`get_legal_actions()` returns only the valid subset. `get_legal_actions_mask()` returns a full 67,200-length binary vector suitable as a policy network mask.

## Neural Network State Representation

`state.get_nn_state()` returns a `(5, 20, 20)` float32 tensor:

| Channel | Content |
|---|---|
| 0 | Current player's pieces (binary) |
| 1 | Next player's pieces (binary) |
| 2 | Player after that (binary) |
| 3 | Previous player's pieces (binary) |
| 4 | Empty squares (binary) |

The current player is always in channel 0, so the representation is always "from the perspective of the player to move." Other colors rotate into channels 1-3 in turn order.

## Directory Structure

```
blokus/                          # Python package
  __init__.py
  engine/                        # Game engine
    __init__.py                  # Re-exports: GameState, encode/decode_action, etc.
    game_state.py                # Core GameState class, action encoding, legal move gen
    piece.py                     # Piece class: occupied squares, transforms, orientations
    board.py                     # Board class: validation, rendering (seaborn/pygame)
    heuristics.py                # Flood-fill territory heuristics
  agents/__init__.py             # RL agents (Phase 2+)
  mcts/__init__.py               # Monte Carlo Tree Search (Phase 3)
  nn/__init__.py                 # Neural network models (Phase 2)
scripts/
  generate_video.py              # Generate animated GIF visualizations of random games
tests/
  __init__.py
  test_game_state.py             # 61 tests: pieces, encoding, moves, modes, edge cases
data/
  blokus_pieces.pkl              # 21 standard Blokus piece definitions (pickled)
legacy/                          # Original codebase preserved for reference
  Game.py, Player.py, RL_agent.py, BlokusEnv_gym.py, Blokus_RL_gym.py,
  heuristic_buffer.cpkl, current_buffer.cpkl, Piece Images.JPG
output/                          # Generated artifacts (gitignored)
configs/                         # Training configs (Phase 4)
readme_ims/                      # Images for README
pyproject.toml                   # Project metadata and dependencies
CLAUDE.md                        # Developer guide and conventions
```

## Development Roadmap

| Phase | Description | Status |
|---|---|---|
| **1. Engine** | Clean GameState with two game modes, tests, visualization | Done |
| **2. Neural Network** | ResNet or similar for joint policy + value estimation | Planned |
| **3. MCTS** | Monte Carlo Tree Search with neural network guidance | Planned |
| **4. Self-Play** | Training loop: self-play data generation + network training | Planned |
| **5. Interactive** | Human-playable mode, agent vs human matchups | Planned |

## Further Reading

- [CLAUDE.md](CLAUDE.md) — developer guide, conventions, and detailed architectural notes
- [Blokus on Wikipedia](https://en.wikipedia.org/wiki/Blokus)
- [AlphaZero paper](https://arxiv.org/abs/1712.01815) — Mastering Chess and Shogi by Self-Play with a General RL Algorithm

![](readme_ims/game_heuristic_vs_random.gif)
*Heuristic agents (blue/yellow) vs random agents (red/grey)*
