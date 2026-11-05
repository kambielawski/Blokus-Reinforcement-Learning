"""Local web server for playing the AI. Binds to loopback only; stdlib only."""

import glob
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

import torch

from blokus.agents.bots import MctsBot
from blokus.engine.game_state import BOARD_SIZE, COLOR_CORNERS, COLOR_NAMES
from blokus.nn.checkpoint import load_network
from blokus.ui.session import GameSession, IllegalMove, piece_catalog

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


class PlayServer:
    """Holds the current game and a cache of loaded networks."""

    def __init__(self, repo_root: str, default_checkpoint: str, games_dir: str,
                 device: str, defaults: Dict[str, Any]):
        self.repo_root = repo_root
        self.default_checkpoint = default_checkpoint
        self.games_dir = games_dir
        self.device = torch.device(device)
        self.defaults = defaults
        self.session: Optional[GameSession] = None
        self._networks: Dict[str, Tuple[torch.nn.Module, Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def checkpoints(self) -> list:
        """Checkpoints found under data/, as paths relative to the repo."""
        pattern = os.path.join(self.repo_root, 'data', '**', 'checkpoint*.pt')
        paths = {os.path.relpath(p, self.repo_root) for p in glob.glob(pattern, recursive=True)}
        paths.add(self.default_checkpoint)
        return sorted(paths)

    def meta(self) -> Dict[str, Any]:
        return {
            'board_size': BOARD_SIZE,
            'color_names': [COLOR_NAMES[c] for c in range(4)],
            'corners': [list(COLOR_CORNERS[c]) for c in range(4)],
            'pieces': piece_catalog(),
            'checkpoints': self.checkpoints(),
            'default_checkpoint': self.default_checkpoint,
            'defaults': self.defaults,
            'device': str(self.device),
        }

    def _network(self, path: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        if path not in self._networks:
            self._networks[path] = load_network(os.path.join(self.repo_root, path), self.device)
        return self._networks[path]

    def new_game(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Abandon any game in progress and start a new one."""
        checkpoint = options.get('checkpoint') or self.default_checkpoint
        if checkpoint not in self.checkpoints():
            raise ValueError(f'Unknown checkpoint: {checkpoint}')
        side = int(options.get('side', 0))
        if side not in (0, 1):
            raise ValueError('side must be 0 or 1')
        merged = {**self.defaults, **{k: options[k] for k in self.defaults if k in options}}
        sims = int(merged['sims'])
        time_limit = float(merged['think_seconds'])
        if sims < 1 or time_limit < 0:
            raise ValueError('sims must be at least 1 and think time cannot be negative')

        with self._lock:
            if self.session is not None and self.session.result() is None:
                self.session.abandon()
            network, net_info = self._network(checkpoint)
            bot = MctsBot(
                network, self.device, sims=sims, c_puct=float(merged['c_puct']),
                batch_size=int(merged['batch_size']), top_k=int(merged['top_k']),
                value_scale=float(merged['value_scale']),
                terminal_score_diff=bool(merged['terminal_score_diff']),
                time_limit=time_limit or None,
            )
            ai_info = {
                'checkpoint': os.path.join(self.repo_root, checkpoint),
                'checkpoint_name': checkpoint,
                'iteration': net_info['iteration'],
                'device': str(self.device),
                **merged,
            }
            self.session = GameSession(bot, side, self.games_dir, ai_info)
            self.session.start()
            return self.session.view()

    def state(self) -> Dict[str, Any]:
        return self.session.view() if self.session else {'status': 'none'}

    def move(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if self.session is None:
            raise IllegalMove('No game is in progress.')
        self.session.human_move(int(body['piece']), int(body['orientation']),
                                int(body['row']), int(body['col']))
        return self.session.view()


def make_handler(app: PlayServer) -> type:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Any, content_type: str = 'application/json') -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == '/':
                with open(os.path.join(_STATIC_DIR, 'index.html'), 'rb') as f:
                    self._send(200, f.read(), 'text/html; charset=utf-8')
            elif self.path == '/api/meta':
                self._send(200, app.meta())
            elif self.path == '/api/state':
                self._send(200, app.state())
            else:
                self._send(404, {'error': 'Not found'})

        def do_POST(self) -> None:
            length = int(self.headers.get('Content-Length', 0))
            try:
                body = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/api/new':
                    self._send(200, app.new_game(body))
                elif self.path == '/api/move':
                    self._send(200, app.move(body))
                else:
                    self._send(404, {'error': 'Not found'})
            except IllegalMove as exc:
                self._send(400, {'error': str(exc), 'illegal': True})
            except (KeyError, ValueError, TypeError) as exc:
                self._send(400, {'error': f'Bad request: {exc}'})

        def log_message(self, format: str, *args: Any) -> None:
            pass  # Polling would flood the terminal

    return Handler


def serve(app: PlayServer, port: int) -> ThreadingHTTPServer:
    """Create the HTTP server on 127.0.0.1. Call serve_forever() to run it."""
    return ThreadingHTTPServer(('127.0.0.1', port), make_handler(app))
