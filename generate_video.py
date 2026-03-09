"""Generate animated GIF visualizations of Blokus games.

Produces one GIF per game mode showing a full random game with annotations.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from game_state import (
    GameState, play_random_game, decode_action,
    BOARD_SIZE, NUM_COLORS, COLOR_NAMES, COLOR_RGB,
)


def render_frame(state: GameState, cell_size: int = 28,
                 title: str = '', move_num: int = 0) -> Image.Image:
    """Render a game state as a PIL Image with title and score bar."""
    bs = BOARD_SIZE
    board_px = bs * cell_size + 1
    margin_top = 40
    margin_bottom = 30
    img_w = board_px
    img_h = board_px + margin_top + margin_bottom

    img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    draw.text((5, 5), title, fill=(0, 0, 0), font=font)
    draw.text((5, 22), f"Move {move_num}", fill=(80, 80, 80), font=font_small)

    # Board
    y_off = margin_top
    for r in range(bs):
        for c in range(bs):
            val = int(state.board[r, c])
            rgb = COLOR_RGB[val]
            x0 = c * cell_size + 1
            y0 = r * cell_size + 1 + y_off
            draw.rectangle([x0, y0, x0 + cell_size - 2, y0 + cell_size - 2],
                           fill=rgb, outline=(160, 160, 160))

    # Grid border
    draw.rectangle([0, y_off, board_px - 1, y_off + board_px - 1],
                   outline=(60, 60, 60), width=2)

    # Score bar at bottom
    scores = state.get_scores()
    score_parts = []
    for ci in range(NUM_COLORS):
        name = COLOR_NAMES[ci]
        rgb = COLOR_RGB[ci + 1]
        score_parts.append((f"{name[0]}:{scores[ci]}", rgb))

    x_pos = 5
    y_score = board_px + margin_top + 6
    for text, rgb in score_parts:
        draw.text((x_pos, y_score), text, fill=rgb, font=font_small)
        x_pos += img_w // 4

    return img


def generate_game_gif(game_mode: str, seed: int, output_path: str,
                      cell_size: int = 28, frame_duration: int = 300):
    """Play a random game and save as animated GIF."""
    print(f"Playing {game_mode} game (seed={seed})...")
    final, history = play_random_game(game_mode, seed=seed, verbose=False)

    mode_label = "Standard 4-Player" if game_mode == 'standard' else "1v1 Dual-Color"

    frames = []
    move_num = 0
    for i, state in enumerate(history):
        # Only record frames where a piece was actually placed (board changed)
        # Plus the initial empty board
        if i == 0 or (i > 0 and not np.array_equal(state.board, history[i-1].board)):
            if i > 0:
                move_num += 1
            title = f"Blokus - {mode_label}"
            frame = render_frame(state, cell_size=cell_size,
                                 title=title, move_num=move_num)
            frames.append(frame)

    if not frames:
        print("No frames generated!")
        return

    # Add a longer pause on the last frame
    durations = [frame_duration] * len(frames)
    durations[-1] = 3000  # 3 second pause on final board

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )

    scores = final.get_scores()
    print(f"  Saved {len(frames)} frames to {output_path}")
    print(f"  Final scores: {scores}")
    if game_mode == 'dual':
        rewards = final.get_rewards()
        print(f"  Agent rewards: {rewards}")


if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))

    generate_game_gif(
        'standard', seed=42,
        output_path=os.path.join(out_dir, 'game_standard.gif'),
        cell_size=26, frame_duration=350,
    )

    generate_game_gif(
        'dual', seed=123,
        output_path=os.path.join(out_dir, 'game_dual.gif'),
        cell_size=26, frame_duration=350,
    )

    print("\nDone! View the GIFs to verify game play looks correct.")
