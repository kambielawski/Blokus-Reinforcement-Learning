# Blokus AlphaZero

An AlphaZero-inspired self-play reinforcement learning agent for [Blokus](https://en.wikipedia.org/wiki/Blokus).

Forked from [DerekGloudemans/Blokus-Reinforcement-Learning](https://github.com/DerekGloudemans/Blokus-Reinforcement-Learning), which provides a working game engine with board logic, piece management, and basic heuristic agents.

![](readme_ims/34.png)

## Current State

### Working
- **New game engine** (`game_state.py`): Clean, immutable `GameState` class for RL
  - Supports two game modes: **standard 4-player** and **1v1 dual-color**
  - Canonical action encoding (67,200 action space)
  - Neural network state representation (5×20×20 float tensor)
  - Fast legal move generation via numpy masks
  - Full test suite (61 tests)
- **Legacy game engine**: Full Blokus rules implementation (2 or 4 players, configurable board size)
- **Piece system**: All 21 standard pieces with rotation/flip/translation transforms
- **Heuristic agents**: Space-filling heuristic (flood-fill territory estimation) and random play
- **Visualization**: Animated GIF generation, pygame board rendering, seaborn heatmaps

### In Progress
- AlphaZero self-play training pipeline (MCTS + neural network policy/value head)

### Not Yet Implemented
- Interactive human play mode
- MCTS (Monte Carlo Tree Search)
- Neural network for joint policy + value estimation
- Self-play data generation pipeline
- Training loop with experience replay

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Play a random game using the new engine
python game_state.py

# Run the test suite
python -m pytest tests/ -v

# Generate visualization GIFs (saved to output/)
python scripts/generate_video.py

# Play a random game using the legacy engine (opens pygame window)
python Game.py
```

## Game Modes

### Standard (1v1v1v1)
4 agents, each controlling one color. Standard Blokus rules on a 20×20 board.

### Dual-Color (1v1)
2 agents, each controlling two colors. Turn order cycles through all 4 colors (0→1→2→3). Agent 0 plays colors 0 and 2; Agent 1 plays colors 1 and 3. Rewards are combined per-agent.

```python
from game_state import GameState, play_random_game

# Standard game
state = GameState.new_game('standard')

# Dual-color game
state = GameState.new_game('dual')

# Game loop
while not state.is_terminal():
    legal = state.get_legal_actions()
    if not legal:
        state = state.pass_turn()
    else:
        action = choose_action(state, legal)  # your agent here
        state = state.apply_action(action)

print(state.get_scores())
print(state.get_rewards())
```

![](readme_ims/game_heuristic_vs_random.gif)
*Heuristic agents (blue/yellow) vs random agents (red/grey)*

## Project Goals
1. Implement AlphaZero-style MCTS with neural network guidance
2. Train via self-play to produce a strong Blokus agent
3. Build an interactive mode so humans can play against the agent

## Architecture

### New Engine (`game_state.py`)
- `GameState` — immutable game state with `apply_action()` / `pass_turn()` / `get_legal_actions()`
- `encode_action()` / `decode_action()` — flat integer action encoding
- `load_pieces()` — pre-computes piece orientations from pickle
- `play_random_game()` — utility to run a full random game

### Legacy Engine
- `Piece.py` — piece representation and geometric transforms
- `Board.py` — board state and move validation
- `Player.py` — player state, valid move tracking, and move selection
- `Game.py` — game orchestration (turns, scoring, piece loading)
- `heuristics.py` — hand-crafted evaluation functions

See [CLAUDE.md](CLAUDE.md) for detailed architectural notes and development conventions.
