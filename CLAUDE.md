# Blokus AlphaZero — CLAUDE.md

## Project Overview
Building an AlphaZero-inspired self-play reinforcement learning agent for the board game [Blokus](https://en.wikipedia.org/wiki/Blokus). Forked from DerekGloudemans's Blokus implementation, which provides a working game engine. The goal is to train an agent that can compete with (and eventually beat) strong human players.

## Repo Structure
```
blokus/                          # Main Python package
  __init__.py
  engine/                        # Game engine
    __init__.py                  # Re-exports key symbols from game_state
    game_state.py                # Clean immutable GameState for RL
    piece.py                     # Piece representation (transforms, orientations)
    board.py                     # Board state, move validation, rendering
    heuristics.py                # Space-filling heuristics (flood-fill)
  agents/                        # RL agents (future Phase 2+)
    __init__.py
  mcts/                          # MCTS (future Phase 3)
    __init__.py
  nn/                            # Neural network models (future Phase 2)
    __init__.py
scripts/                         # CLI utilities
  generate_video.py              # Generate animated GIF visualizations
tests/                           # pytest suite
  __init__.py
  test_game_state.py             # 61 tests for the game engine
data/                            # Data files
  blokus_pieces.pkl              # Pickled piece definitions (21 standard pieces)
legacy/                          # Old code kept for reference only
  Game.py                        # Original game orchestrator
  Player.py                      # Original player logic with strategies
  RL_agent.py                    # Legacy Q-learning attempt (non-functional)
  BlokusEnv_gym.py               # Stub OpenAI Gym environment
  Blokus_RL_gym.py               # Stub Gym test script
  heuristic_buffer.cpkl          # Saved replay buffer
  current_buffer.cpkl            # Saved replay buffer snapshot
  Piece Images.JPG               # Reference image of all Blokus pieces
output/                          # Generated artifacts (gitignored)
configs/                         # Training configs (future Phase 4)
readme_ims/                      # README images
pyproject.toml                   # Project config and dependencies
CLAUDE.md
README.md
.gitignore
```

## Build / Run / Test

### Dependencies
```bash
pip install -e ".[all]"          # Install package with all optional deps
pip install -e ".[dev]"          # Install with just test deps
```
Python 3.9+ recommended.

### Running a game
```bash
python -c "from blokus.engine import GameState, play_random_game; play_random_game('standard', verbose=True)"
```

### Running tests
```bash
python -m pytest tests/ -v       # 61 tests covering pieces, moves, state transitions, both modes
```

### Generating visualizations
```bash
python scripts/generate_video.py # Generates output/game_standard.gif and output/game_dual.gif
```

### Importing in code
```python
from blokus.engine import GameState, encode_action, decode_action, play_random_game
from blokus.engine.game_state import BOARD_SIZE, NUM_PIECES, ACTION_SPACE_SIZE
```

## Key Architectural Notes

### Game Engine (`blokus.engine`)
The engine provides a clean, immutable `GameState` class designed for AlphaZero-style RL:

- **Immutable**: `apply_action()` and `pass_turn()` return new states; originals are never mutated
- **Two game modes**:
  - `'standard'`: 4 agents, each controlling one color (1v1v1v1)
  - `'dual'`: 2 agents, each controlling two colors. Turn order: color 0→1→2→3. Agent 0 plays colors 0,2; Agent 1 plays colors 1,3.
- **Canonical action encoding**: `encode_action(piece_id, orientation_id, row, col) -> int` (67,200 total action space)
- **NN-ready state**: `get_nn_state()` returns (5, 20, 20) float32 tensor — current player channel first, then others, then empty
- **Fast legal move generation**: numpy-based anchor/forbidden mask computation, ~0.3s for a full random game
- **20x20 board** with 4 colors and 21 pieces per color in both modes

### Action Space
Flat integer encoding: `piece_id * 3200 + orientation_id * 400 + row * 20 + col`
- 21 pieces × 8 max orientations × 20 rows × 20 cols = 67,200 possible actions
- `get_legal_actions()` returns only valid action indices
- `get_legal_actions_mask()` returns full binary mask for policy network output

### Move Validation Rules (Blokus)
1. All piece squares must be within board bounds
2. At least one piece square must be diagonally adjacent to an existing piece of the same color (or cover the assigned board corner on first move)
3. No piece square may be orthogonally adjacent to an existing piece of the same color
4. No piece square may overlap an occupied square

### Piece Data
Piece definitions are stored in `data/blokus_pieces.pkl` (pickled from the original codebase). The pickle contains objects of the original `Piece` class; a custom unpickler in `game_state.py` handles the module path redirect from the legacy `Piece` module to `blokus.engine.piece`.

### State Representation for NN
5-channel (20×20) float32 tensor:
- Channel 0: current player's pieces (binary)
- Channels 1-3: other colors in turn order (binary)
- Channel 4: empty squares (binary)

### Legacy Code
The original `Game.py`/`Board.py`/`Player.py` code is preserved in `legacy/` for reference. It is not imported by the new engine.

## Development Conventions
- Use clear, descriptive commit messages
- Add thorough comments explaining non-obvious logic
- New RL/AlphaZero code goes in the appropriate `blokus/` subpackage
- Use type hints in new code
- Run `python -m pytest tests/ -v` before committing
- Import from `blokus.engine` (not relative paths or sys.path hacks)
