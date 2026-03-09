# Blokus AlphaZero — CLAUDE.md

## Project Overview
Building an AlphaZero-inspired self-play reinforcement learning agent for the board game [Blokus](https://en.wikipedia.org/wiki/Blokus). Forked from DerekGloudemans's Blokus implementation, which provides a working game engine. The goal is to train an agent that can compete with (and eventually beat) strong human players.

## Repo Structure
```
├── game_state.py             # [NEW] Clean immutable GameState for RL (supports standard + dual mode)
├── generate_video.py         # [NEW] Generates animated GIF visualizations of random games
├── tests/
│   └── test_game_state.py    # [NEW] Comprehensive pytest suite (61 tests)
├── requirements.txt          # [NEW] Python dependencies
├── game_standard.gif         # [NEW] Visualization of a standard 4-player game
├── game_dual.gif             # [NEW] Visualization of a 1v1 dual-color game
├── Piece.py                  # Piece representation (occupied squares, corners, adjacents, transforms)
├── Board.py                  # Board state, move validation, rendering (numpy/seaborn/pygame)
├── Player.py                 # Player logic: valid move generation, move selection (random/heuristic/manual)
├── Game.py                   # Game orchestration: turns, scoring, piece loading, move enumeration
├── heuristics.py             # Space-filling heuristics (flood-fill territory estimation)
├── RL_agent.py               # [Legacy] Q-learning attempt with ResNet — does not work
├── BlokusEnv_gym.py          # [Stub] OpenAI Gym environment skeleton — not implemented
├── Blokus_RL_gym.py          # [Stub] Gym test script
├── blokus_pieces_lim_5.pkl   # Pickled piece definitions (21 standard Blokus pieces)
├── heuristic_buffer.cpkl     # Saved replay buffer from heuristic training
├── current_buffer.cpkl       # Saved replay buffer snapshot
├── readme_ims/               # Images for README
└── Piece Images.JPG          # Reference image of all Blokus pieces
```

## Build / Run / Test

### Dependencies
```bash
pip install -r requirements.txt
```
Python 3.9+ recommended.

### Running a game (new engine)
```bash
python game_state.py          # Plays random games in both modes, prints scores and board
```

### Running a game (legacy engine)
```bash
python Game.py                # Runs a 4-player random game on 20x20 board (uses pygame)
```

### Running tests
```bash
python -m pytest tests/ -v    # 61 tests covering pieces, moves, state transitions, both modes
```

### Generating visualizations
```bash
python generate_video.py      # Generates game_standard.gif and game_dual.gif
```

### Visualization options
- `GameState.render_image()` — returns RGB numpy array of the board
- `GameState.render_text()` — ASCII board representation
- `Board.display_pygame()` — opens a pygame window (legacy engine)
- `Board.display2()` — seaborn heatmap (legacy engine)

## Key Architectural Notes

### New Game Engine (`game_state.py`)
The new engine provides a clean, immutable `GameState` class designed for AlphaZero-style RL:

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

### Legacy Engine
The original `Game.py`/`Board.py`/`Player.py` code still works independently. The new `game_state.py` reuses `Piece.py` for loading piece definitions from the pickle file but is otherwise self-contained.

### State Representation for NN
5-channel (20×20) float32 tensor:
- Channel 0: current player's pieces (binary)
- Channels 1-3: other colors in turn order (binary)
- Channel 4: empty squares (binary)

## Development Conventions
- Use clear, descriptive commit messages
- Add thorough comments explaining non-obvious logic
- Keep game engine modifications backward-compatible
- New RL/AlphaZero code should go in separate modules, not modify the existing game engine files
- Use type hints in new code
- Run `pytest tests/ -v` before committing
