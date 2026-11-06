#!/usr/bin/env python3
"""Play the Blokus AI in your browser (dual-color 1v1).

Usage:
    python scripts/play.py
    python scripts/play.py --checkpoint data/tufts/checkpoint_best_eval.pt --sims 800

With no --checkpoint, the opponent defaults to the mobility heuristic; pick any trained
model in the New game dialog.
"""

import argparse
import os
import sys
import threading
import webbrowser

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from blokus.ui.server import HEURISTIC_OPPONENT, PlayServer, serve


def main() -> None:
    parser = argparse.ArgumentParser(description='Play the Blokus AI in your browser')
    parser.add_argument('--checkpoint', default=HEURISTIC_OPPONENT,
                        help='Default opponent: a checkpoint path relative to the repo root')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', default='cpu', help='cpu or mps')
    parser.add_argument('--games-dir', default=os.path.join(REPO_ROOT, 'data', 'human_games'))
    parser.add_argument('--sims', type=int, default=800, help='Default simulations per move')
    parser.add_argument('--think-seconds', type=float, default=0,
                        help='Default limit on AI thinking per move (0 = no limit)')
    parser.add_argument('--value-scale', type=float, default=1.0,
                        help='Multiplier on network values during search')
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--c-puct', type=float, default=1.5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    if args.checkpoint != HEURISTIC_OPPONENT and not os.path.exists(
            os.path.join(REPO_ROOT, args.checkpoint)):
        sys.exit(f'Checkpoint not found: {args.checkpoint}')

    app = PlayServer(
        repo_root=REPO_ROOT,
        default_checkpoint=args.checkpoint,
        games_dir=args.games_dir,
        device=args.device,
        defaults={
            'sims': args.sims,
            'think_seconds': args.think_seconds,
            'value_scale': args.value_scale,
            'top_k': args.top_k,
            'c_puct': args.c_puct,
            'batch_size': args.batch_size,
        },
    )
    server = serve(app, args.port)
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Blokus is running at {url}  (Ctrl-C to stop)')
    print(f'Game records are saved to {args.games_dir}')
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
