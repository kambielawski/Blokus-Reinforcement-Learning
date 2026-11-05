"""One human-vs-AI dual-mode game: rules enforcement, AI turns, and the record.

The record is rewritten to disk after every ply, so an abandoned or crashed game
still leaves a complete history for later analysis.
"""

import hashlib
import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from blokus.engine.game_state import (
    ACTION_ENCODING_VERSION, BOARD_SIZE, COLOR_CORNERS, COLOR_NAMES, GameState,
    NUM_COLORS, NUM_PIECES, decode_action, encode_action, load_pieces,
)

PIECE_NAMES = ('I1', 'I2', 'I3', 'I4', 'I5', 'V3', 'L4', 'L5', 'V5', 'W', 'Z5',
               'F', 'X', 'N', 'P', 'U', 'Y', 'T5', 'T4', 'O', 'S4')

TOP_MOVES_RECORDED = 5


class IllegalMove(Exception):
    """A human move the rules reject. str(exc) explains why."""


def explain_illegal(state: GameState, piece: int, orientation: int,
                    row: int, col: int) -> Optional[str]:
    """Why this placement is illegal for the current color, or None if legal."""
    color = state.current_color
    name = COLOR_NAMES[color]
    if not 0 <= piece < NUM_PIECES:
        return 'That piece does not exist.'
    if piece not in state.pieces_remaining[color]:
        return f'{name} has already played {PIECE_NAMES[piece]}.'
    orientations = state.pieces[piece].orientations
    if not 0 <= orientation < len(orientations):
        return 'That orientation does not exist for this piece.'

    cells = [(r + row, c + col) for r, c in orientations[orientation].occupied]
    if any(not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE) for r, c in cells):
        return 'The piece has to fit entirely on the board.'
    board = state.board
    own = color + 1
    if any(board[r, c] != 0 for r, c in cells):
        return 'The piece overlaps a piece already on the board.'
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == own:
                return f'The piece shares an edge with another {name} piece.'

    if not state._has_played[color]:
        corner = COLOR_CORNERS[color]
        if corner not in cells:
            return f"{name}'s first piece has to cover its corner of the board."
        return None
    for r, c in cells:
        for nr, nc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == own:
                return None
    return f'The piece has to touch another {name} piece corner to corner.'


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _agent_scores(state: GameState) -> List[int]:
    scores = state.get_scores()
    return [scores[0] + scores[2], scores[1] + scores[3]]


class GameSession:
    """State of one game. Safe to read from one thread while the AI runs in another.

    Args:
        bot: Object with select_action(state). May expose last_stats (see MCTS).
        human_agent: 0 plays Yellow and Red and moves first; 1 plays Blue and Green.
        games_dir: Where the JSON move record is written.
        ai_info: Description of the AI for the record. If it has a 'checkpoint'
            path, the file's hash is recorded too.
    """

    def __init__(self, bot: Any, human_agent: int, games_dir: str,
                 ai_info: Optional[Dict[str, Any]] = None):
        assert human_agent in (0, 1)
        self.bot = bot
        self.human_agent = human_agent
        self.state = GameState.new_game('dual')
        self.moves: List[Dict[str, Any]] = []
        self.thinking = False
        self.think_started = 0.0
        self.error: Optional[str] = None
        self.abandoned = False
        self.game_id = uuid.uuid4().hex[:8]
        self._lock = threading.RLock()
        self._turn_started = time.monotonic()

        self.started_at = datetime.now(timezone.utc)
        os.makedirs(games_dir, exist_ok=True)
        stamp = self.started_at.astimezone().strftime('%Y%m%d-%H%M%S')
        self.record_path = os.path.abspath(
            os.path.join(games_dir, f'{stamp}-{self.game_id}.json'))
        self.ai_info = dict(ai_info or {})
        if os.path.exists(self.ai_info.get('checkpoint', '')):
            self.ai_info['checkpoint_sha256'] = _sha256(self.ai_info['checkpoint'])
        self._write_record()

    # ---- Game flow ----

    def start(self) -> None:
        """Play any AI moves or passes that come before the human's first turn."""
        self._launch_advance()

    def human_move(self, piece: int, orientation: int, row: int, col: int) -> None:
        """Apply the human's move, or raise IllegalMove with the reason."""
        with self._lock:
            if self.state.is_terminal():
                raise IllegalMove('The game is over.')
            if self.thinking or self.state.get_current_agent() != self.human_agent:
                raise IllegalMove('It is not your turn.')
            reason = explain_illegal(self.state, piece, orientation, row, col)
            if reason is not None:
                raise IllegalMove(reason)
            action = encode_action(piece, orientation, row, col)
            if action not in self.state.get_legal_actions():
                # The two rule implementations should never disagree.
                raise IllegalMove('The engine rejected this move.')
            self._apply(action, 'human', {'seconds': time.monotonic() - self._turn_started})
        self._launch_advance()

    def abandon(self) -> None:
        """Mark an unfinished game as abandoned; an in-flight AI move is discarded."""
        with self._lock:
            self.abandoned = True
            self._write_record()

    def _launch_advance(self) -> None:
        with self._lock:
            self.thinking = self._ai_to_act()
            self.think_started = time.monotonic()
        threading.Thread(target=self._advance, daemon=True).start()

    def _ai_to_act(self) -> bool:
        """True if the next thing to happen is not waiting on the human."""
        state = self.state
        return not state.is_terminal() and not (
            state.get_current_agent() == self.human_agent and state.get_legal_actions())

    def _advance(self) -> None:
        """Pass and play AI moves until the human can move or the game ends."""
        try:
            while True:
                with self._lock:
                    if self.abandoned or self.state.is_terminal():
                        break
                    state = self.state
                    if not state.get_legal_actions():
                        self._apply(None, 'human' if state.get_current_agent() == self.human_agent
                                    else 'ai', {})
                        continue
                    if state.get_current_agent() == self.human_agent:
                        break
                    self.thinking = True
                    self.think_started = time.monotonic()

                # The lock is released while thinking so the UI can keep polling.
                action = self.bot.select_action(state)
                info: Dict[str, Any] = {'seconds': time.monotonic() - self.think_started}
                stats = getattr(self.bot, 'last_stats', None)
                if stats:
                    top = sorted(stats['children'].items(), key=lambda kv: -kv[1][0])
                    info.update({
                        'sims': stats['sims'],
                        'root_value': stats['value'],
                        'top_moves': [
                            {'action': a, 'visits': visits, 'q': q, 'prior': prior}
                            for a, (visits, q, prior) in top[:TOP_MOVES_RECORDED]
                        ],
                    })
                with self._lock:
                    if self.abandoned:
                        break
                    self._apply(action, 'ai', info)
        except Exception:
            with self._lock:
                self.error = traceback.format_exc()
                self._write_record()
            print(self.error, flush=True)
        finally:
            with self._lock:
                self.thinking = False
                self._turn_started = time.monotonic()

    def _apply(self, action: Optional[int], actor: str, info: Dict[str, Any]) -> None:
        """Apply a move (None = pass) to the state and append it to the record."""
        state = self.state
        entry: Dict[str, Any] = {
            'ply': len(self.moves),
            'color': state.current_color,
            'color_name': COLOR_NAMES[state.current_color],
            'agent': state.get_current_agent(),
            'actor': actor,
            'type': 'pass' if action is None else 'place',
        }
        if action is None:
            self.state = state.pass_turn()
        else:
            piece, orientation, row, col = decode_action(action)
            occupied = state.pieces[piece].orientations[orientation].occupied
            entry.update({
                'action': action, 'piece': piece, 'piece_name': PIECE_NAMES[piece],
                'orientation': orientation, 'row': row, 'col': col,
                'cells': [[r + row, c + col] for r, c in occupied],
            })
            self.state = state.apply_action(action)
        entry.update(info)
        entry['agent_scores_after'] = _agent_scores(self.state)
        self.moves.append(entry)
        self._write_record()

    # ---- Result and serialization ----

    def result(self) -> Optional[Dict[str, Any]]:
        """Final result, or None while the game is in progress."""
        if not self.state.is_terminal():
            return None
        totals = _agent_scores(self.state)
        human, ai = totals[self.human_agent], totals[1 - self.human_agent]
        return {
            'winner': 'human' if human > ai else ('ai' if ai > human else 'draw'),
            'human_score': human,
            'ai_score': ai,
            'margin': human - ai,
            'color_scores': [self.state.get_scores()[c] for c in range(NUM_COLORS)],
        }

    def _status(self) -> str:
        if self.error:
            return 'error'
        if self.state.is_terminal():
            return 'finished'
        return 'abandoned' if self.abandoned else 'playing'

    def _write_record(self) -> None:
        record = {
            'format': 1,
            'game_id': self.game_id,
            'started_at': self.started_at.isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'status': self._status(),
            'mode': 'dual',
            'action_encoding': ACTION_ENCODING_VERSION,
            'human_agent': self.human_agent,
            'human_colors': [COLOR_NAMES[c] for c in range(NUM_COLORS)
                             if c % 2 == self.human_agent],
            'ai': self.ai_info,
            'moves': self.moves,
            'result': self.result(),
            'error': self.error,
        }
        temp_path = f'{self.record_path}.tmp'
        with open(temp_path, 'w') as f:
            json.dump(record, f, indent=1)
        os.replace(temp_path, self.record_path)

    def view(self) -> Dict[str, Any]:
        """Everything the browser needs to draw the game."""
        with self._lock:
            state = self.state
            terminal = state.is_terminal()
            your_turn = (not terminal and not self.thinking
                         and state.get_current_agent() == self.human_agent)
            legal = [list(decode_action(a)) for a in state.get_legal_actions()] if your_turn else []
            return {
                'game_id': self.game_id,
                'status': self._status(),
                'human_agent': self.human_agent,
                'board': state.board.tolist(),
                'current_color': state.current_color,
                'current_agent': state.get_current_agent(),
                'your_turn': your_turn,
                'thinking': self.thinking,
                'think_seconds': time.monotonic() - self.think_started if self.thinking else 0.0,
                'legal': legal,
                'pieces_remaining': [sorted(state.pieces_remaining[c]) for c in range(NUM_COLORS)],
                'color_scores': [state.get_scores()[c] for c in range(NUM_COLORS)],
                'agent_scores': _agent_scores(state),
                'moves': self.moves,
                'result': self.result(),
                'error': self.error,
                'record_path': self.record_path,
                'ai': {k: v for k, v in self.ai_info.items() if k != 'checkpoint_sha256'},
            }


def piece_catalog() -> List[Dict[str, Any]]:
    """Piece shapes and how orientations map to each other under rotate and flip."""
    catalog = []
    for pid, piece in enumerate(load_pieces()):
        shapes = [tuple(sorted(ori.occupied)) for ori in piece.orientations]
        index = {shape: i for i, shape in enumerate(shapes)}

        def transform(shape: tuple, flip: bool) -> int:
            height = max(r for r, c in shape) + 1
            width = max(c for r, c in shape) + 1
            if flip:
                moved = [(r, width - 1 - c) for r, c in shape]
            else:  # quarter turn clockwise
                moved = [(c, height - 1 - r) for r, c in shape]
            return index[tuple(sorted(moved))]

        catalog.append({
            'id': pid,
            'name': PIECE_NAMES[pid],
            'size': piece.num_cells,
            'orientations': [[list(cell) for cell in shape] for shape in shapes],
            'rotate': [transform(shape, flip=False) for shape in shapes],
            'flip': [transform(shape, flip=True) for shape in shapes],
        })
    return catalog
