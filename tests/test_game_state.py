"""Comprehensive tests for the Blokus game engine (game_state.py)."""

import sys
import os
import numpy as np
import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game_state import (
    GameState, load_pieces, clear_piece_cache,
    encode_action, decode_action,
    BOARD_SIZE, NUM_PIECES, MAX_ORIENTATIONS, NUM_COLORS,
    ACTION_SPACE_SIZE, COLOR_CORNERS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure piece cache is fresh for each test module run."""
    # We don't clear per-test (slow), just ensure it's loadable
    load_pieces()


@pytest.fixture
def new_standard_game():
    return GameState.new_game('standard')


@pytest.fixture
def new_dual_game():
    return GameState.new_game('dual')


# =============================================================================
# Piece loading
# =============================================================================

class TestPieceLoading:

    def test_loads_21_pieces(self):
        pieces = load_pieces()
        assert len(pieces) == NUM_PIECES

    def test_each_piece_has_orientations(self):
        pieces = load_pieces()
        for p in pieces:
            assert 1 <= p.num_orientations <= MAX_ORIENTATIONS

    def test_orientation_cells_in_range(self):
        """All cells should start from (0,0) after shift_min."""
        pieces = load_pieces()
        for p in pieces:
            for ori in p.orientations:
                for r, c in ori.occupied:
                    assert r >= 0 and c >= 0

    def test_piece_sizes_are_1_to_5(self):
        pieces = load_pieces()
        sizes = set()
        for p in pieces:
            sizes.add(p.num_cells)
        # Standard Blokus has pieces of sizes 1-5
        assert sizes == {1, 2, 3, 4, 5}

    def test_total_orientations_reasonable(self):
        """21 pieces with 1-8 orientations each."""
        pieces = load_pieces()
        total = sum(p.num_orientations for p in pieces)
        # Should be between 21 (all symmetric) and 168 (all 8 orientations)
        assert 21 <= total <= 168

    def test_monomino_has_one_orientation(self):
        """The 1x1 piece should have exactly 1 orientation."""
        pieces = load_pieces()
        for p in pieces:
            if p.num_cells == 1:
                assert p.num_orientations == 1
                assert p.orientations[0].occupied == ((0, 0),)

    def test_orientations_are_unique(self):
        """No duplicate orientations for any piece."""
        pieces = load_pieces()
        for p in pieces:
            seen = set()
            for ori in p.orientations:
                key = ori.occupied
                assert key not in seen, f"Duplicate orientation: {key}"
                seen.add(key)


# =============================================================================
# Action encoding / decoding
# =============================================================================

class TestActionEncoding:

    def test_roundtrip(self):
        for pid in [0, 10, 20]:
            for oid in [0, 3, 7]:
                for row in [0, 10, 19]:
                    for col in [0, 10, 19]:
                        action = encode_action(pid, oid, row, col)
                        assert decode_action(action) == (pid, oid, row, col)

    def test_action_range(self):
        """All valid actions should be in [0, ACTION_SPACE_SIZE)."""
        for pid in range(NUM_PIECES):
            for oid in range(MAX_ORIENTATIONS):
                for row in range(BOARD_SIZE):
                    for col in range(BOARD_SIZE):
                        a = encode_action(pid, oid, row, col)
                        assert 0 <= a < ACTION_SPACE_SIZE

    def test_unique_encoding(self):
        """Different (pid, oid, r, c) -> different action index."""
        seen = set()
        for pid in range(NUM_PIECES):
            for oid in range(MAX_ORIENTATIONS):
                for row in range(BOARD_SIZE):
                    for col in range(BOARD_SIZE):
                        a = encode_action(pid, oid, row, col)
                        assert a not in seen
                        seen.add(a)

    def test_min_max_action(self):
        assert encode_action(0, 0, 0, 0) == 0
        assert encode_action(NUM_PIECES - 1, MAX_ORIENTATIONS - 1,
                             BOARD_SIZE - 1, BOARD_SIZE - 1) == ACTION_SPACE_SIZE - 1


# =============================================================================
# New game state
# =============================================================================

class TestNewGame:

    def test_empty_board(self, new_standard_game):
        assert np.all(new_standard_game.board == 0)

    def test_all_pieces_remaining(self, new_standard_game):
        for ci in range(NUM_COLORS):
            assert new_standard_game.pieces_remaining[ci] == frozenset(range(NUM_PIECES))

    def test_initial_color(self, new_standard_game):
        assert new_standard_game.current_color == 0

    def test_zero_passes(self, new_standard_game):
        assert new_standard_game.consecutive_passes == 0

    def test_not_terminal(self, new_standard_game):
        assert not new_standard_game.is_terminal()

    def test_has_legal_moves(self, new_standard_game):
        legal = new_standard_game.get_legal_actions()
        assert len(legal) > 0

    def test_scores_all_zero(self, new_standard_game):
        scores = new_standard_game.get_scores()
        assert all(s == 0 for s in scores.values())

    def test_game_mode_standard(self, new_standard_game):
        assert new_standard_game.game_mode == 'standard'

    def test_game_mode_dual(self, new_dual_game):
        assert new_dual_game.game_mode == 'dual'


# =============================================================================
# First move rules
# =============================================================================

class TestFirstMove:

    def test_first_move_covers_corner(self, new_standard_game):
        """Every legal first move must place a cell on the assigned corner."""
        state = new_standard_game
        corner = COLOR_CORNERS[0]
        legal = state.get_legal_actions()
        assert len(legal) > 0

        for action in legal:
            pid, oid, tx, ty = decode_action(action)
            ori = state.pieces[pid].orientations[oid]
            cells = {(cr + tx, cc + ty) for cr, cc in ori.occupied}
            assert corner in cells, (
                f"Action {action} (pid={pid}, oid={oid}, tx={tx}, ty={ty}) "
                f"does not cover corner {corner}. Cells: {cells}")

    def test_first_move_all_in_bounds(self, new_standard_game):
        state = new_standard_game
        for action in state.get_legal_actions():
            pid, oid, tx, ty = decode_action(action)
            ori = state.pieces[pid].orientations[oid]
            for cr, cc in ori.occupied:
                r, c = cr + tx, cc + ty
                assert 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def test_each_color_gets_different_corner(self):
        """Cycle through all 4 first moves, each on a different corner."""
        state = GameState.new_game('standard')
        for color_idx in range(NUM_COLORS):
            assert state.current_color == color_idx
            corner = COLOR_CORNERS[color_idx]
            legal = state.get_legal_actions()
            assert len(legal) > 0

            # Pick a legal first move
            action = legal[0]
            new_state = state.apply_action(action)

            # Verify corner is covered
            assert new_state.board[corner[0], corner[1]] == color_idx + 1

            state = new_state


# =============================================================================
# Move validation
# =============================================================================

class TestMoveValidation:

    def test_no_overlap(self, new_standard_game):
        """After applying a move, all new cells should have been empty before."""
        state = new_standard_game
        for action in state.get_legal_actions()[:5]:
            old_board = state.board.copy()
            new_state = state.apply_action(action)

            pid, oid, tx, ty = decode_action(action)
            ori = state.pieces[pid].orientations[oid]
            for cr, cc in ori.occupied:
                assert old_board[cr + tx, cc + ty] == 0

    def test_piece_removed_from_remaining(self, new_standard_game):
        state = new_standard_game
        action = state.get_legal_actions()[0]
        pid, _, _, _ = decode_action(action)

        new_state = state.apply_action(action)
        assert pid not in new_state.pieces_remaining[0]
        assert pid in state.pieces_remaining[0]  # original unchanged

    def test_turn_advances(self, new_standard_game):
        state = new_standard_game
        action = state.get_legal_actions()[0]
        new_state = state.apply_action(action)
        assert new_state.current_color == 1

    def test_diagonal_adjacency_required(self):
        """After the first move, subsequent pieces must be diag-adjacent."""
        state = GameState.new_game('standard')
        # Make first moves for all 4 colors
        for _ in range(4):
            action = state.get_legal_actions()[0]
            state = state.apply_action(action)

        # Now it's color 0's second turn
        assert state.current_color == 0
        legal = state.get_legal_actions()

        for action in legal[:3]:
            pid, oid, tx, ty = decode_action(action)
            ori = state.pieces[pid].orientations[oid]
            cells = [(cr + tx, cc + ty) for cr, cc in ori.occupied]

            # At least one cell must be diagonally adjacent to an existing same-color cell
            has_diag = False
            for r, c in cells:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if state.board[nr, nc] == 1:  # color 0 = board value 1
                            has_diag = True
                            break
                if has_diag:
                    break
            assert has_diag, f"Move has no diagonal adjacency to same color"

    def test_no_orthogonal_adjacency_to_same_color(self):
        """No placed cell should be orthogonally adjacent to same color."""
        state = GameState.new_game('standard')
        # Play 4 first moves, then check second moves
        for _ in range(4):
            state = state.apply_action(state.get_legal_actions()[0])

        legal = state.get_legal_actions()
        for action in legal[:5]:
            new_state = state.apply_action(action)
            pid, oid, tx, ty = decode_action(action)
            ori = state.pieces[pid].orientations[oid]

            for cr, cc in ori.occupied:
                r, c = cr + tx, cc + ty
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        # The cell might be part of the same piece being placed
                        if (nr - tx, nc - ty) in set(ori.occupied):
                            continue
                        assert new_state.board[nr, nc] != 1, (
                            f"Cell ({r},{c}) is ortho-adjacent to same color at ({nr},{nc})")


# =============================================================================
# State transitions
# =============================================================================

class TestStateTransitions:

    def test_apply_action_does_not_mutate_original(self, new_standard_game):
        state = new_standard_game
        original_board = state.board.copy()
        original_remaining = state.pieces_remaining

        _ = state.apply_action(state.get_legal_actions()[0])

        np.testing.assert_array_equal(state.board, original_board)
        assert state.pieces_remaining == original_remaining
        assert state.current_color == 0

    def test_pass_does_not_mutate_original(self, new_standard_game):
        state = new_standard_game
        original_color = state.current_color

        new_state = state.pass_turn()

        assert state.current_color == original_color
        assert new_state.current_color == 1
        assert new_state.consecutive_passes == 1

    def test_consecutive_passes_reset_on_move(self):
        state = GameState.new_game('standard')
        state = state.pass_turn()  # 1 pass
        state = state.pass_turn()  # 2 passes

        # Now it's color 2's turn. Let's make a first move.
        legal = state.get_legal_actions()
        assert len(legal) > 0
        new_state = state.apply_action(legal[0])
        assert new_state.consecutive_passes == 0

    def test_score_increases_after_move(self, new_standard_game):
        state = new_standard_game
        action = state.get_legal_actions()[0]
        pid, _, _, _ = decode_action(action)
        piece_size = state.pieces[pid].num_cells

        new_state = state.apply_action(action)
        assert new_state.get_scores()[0] == piece_size

    def test_copy_is_independent(self, new_standard_game):
        state = new_standard_game
        copy = state.copy()
        copy.board[0, 0] = 99
        assert state.board[0, 0] == 0


# =============================================================================
# Terminal state detection
# =============================================================================

class TestTerminal:

    def test_not_terminal_at_start(self, new_standard_game):
        assert not new_standard_game.is_terminal()

    def test_terminal_after_4_consecutive_passes(self, new_standard_game):
        state = new_standard_game
        for _ in range(NUM_COLORS):
            state = state.pass_turn()
        assert state.is_terminal()

    def test_not_terminal_after_3_passes(self, new_standard_game):
        state = new_standard_game
        for _ in range(NUM_COLORS - 1):
            state = state.pass_turn()
        assert not state.is_terminal()

    def test_terminal_when_all_pieces_played(self, new_standard_game):
        """Simulate all pieces played by one color."""
        state = new_standard_game
        # Manually create a state where color 0 has no pieces left
        pr = list(state.pieces_remaining)
        pr[0] = frozenset()
        terminal_state = GameState(
            state.board, tuple(pr), state.current_color, 0,
            state.game_mode, state.pieces, state._has_played)
        assert terminal_state.is_terminal()

    def test_full_game_terminates(self):
        """A random game should always terminate."""
        from game_state import play_random_game
        final, history = play_random_game('standard', seed=42)
        assert final.is_terminal()
        assert len(history) > 10


# =============================================================================
# Both game modes
# =============================================================================

class TestGameModes:

    def test_standard_has_4_agents(self, new_standard_game):
        assert new_standard_game.get_num_agents() == 4

    def test_dual_has_2_agents(self, new_dual_game):
        assert new_dual_game.get_num_agents() == 2

    def test_standard_agent_equals_color(self, new_standard_game):
        state = new_standard_game
        for i in range(4):
            assert state.get_current_agent() == i
            state = state.pass_turn()

    def test_dual_agent_mapping(self, new_dual_game):
        """In dual mode, colors 0,2 -> agent 0; colors 1,3 -> agent 1."""
        state = new_dual_game
        expected_agents = [0, 1, 0, 1]
        for expected in expected_agents:
            assert state.get_current_agent() == expected
            state = state.pass_turn()

    def test_dual_game_completes(self):
        from game_state import play_random_game
        final, history = play_random_game('dual', seed=99)
        assert final.is_terminal()

    def test_dual_rewards_are_two_agents(self):
        from game_state import play_random_game
        final, _ = play_random_game('dual', seed=99)
        rewards = final.get_rewards()
        assert set(rewards.keys()) == {0, 1}
        # Rewards should be opposite
        assert abs(rewards[0] + rewards[1]) < 1e-6

    def test_standard_rewards_are_four_agents(self):
        from game_state import play_random_game
        final, _ = play_random_game('standard', seed=42)
        rewards = final.get_rewards()
        assert set(rewards.keys()) == {0, 1, 2, 3}

    def test_both_modes_use_same_board_size(self, new_standard_game, new_dual_game):
        assert new_standard_game.board.shape == (BOARD_SIZE, BOARD_SIZE)
        assert new_dual_game.board.shape == (BOARD_SIZE, BOARD_SIZE)

    def test_both_modes_use_4_colors(self, new_standard_game, new_dual_game):
        """Both modes have 4 sets of pieces."""
        for state in [new_standard_game, new_dual_game]:
            assert len(state.pieces_remaining) == 4


# =============================================================================
# Neural network state
# =============================================================================

class TestNNState:

    def test_shape(self, new_standard_game):
        nn = new_standard_game.get_nn_state()
        assert nn.shape == (NUM_COLORS + 1, BOARD_SIZE, BOARD_SIZE)

    def test_dtype(self, new_standard_game):
        nn = new_standard_game.get_nn_state()
        assert nn.dtype == np.float32

    def test_empty_board_is_all_empty_channel(self, new_standard_game):
        nn = new_standard_game.get_nn_state()
        # Channel 4 (empty) should be all 1s
        np.testing.assert_array_equal(nn[4], np.ones((BOARD_SIZE, BOARD_SIZE)))
        # Color channels should be all 0s
        for i in range(4):
            np.testing.assert_array_equal(nn[i], np.zeros((BOARD_SIZE, BOARD_SIZE)))

    def test_current_player_is_channel_0(self):
        """After a move, the current player's pieces should be in channel 0."""
        state = GameState.new_game('standard')
        action = state.get_legal_actions()[0]
        state = state.apply_action(action)
        # Now it's color 1's turn
        nn = state.get_nn_state()
        # Channel 0 should be color 1's pieces (currently empty)
        # Channel 1 should be color 2's pieces (empty)
        # Channel 3 should be color 0's pieces (has the first move)
        assert nn[0].sum() == 0  # color 1 hasn't played
        assert nn[3].sum() > 0   # color 0 played (it's 3 positions away in turn order)


# =============================================================================
# Rendering
# =============================================================================

class TestRendering:

    def test_text_render(self, new_standard_game):
        text = new_standard_game.render_text()
        lines = text.strip().split('\n')
        assert len(lines) == BOARD_SIZE
        # All dots on empty board
        assert all(c in '.  ' for c in text.replace('\n', ''))

    def test_image_render_shape(self, new_standard_game):
        img = new_standard_game.render_image(cell_size=24)
        expected_size = BOARD_SIZE * 24 + 1
        assert img.shape == (expected_size, expected_size, 3)
        assert img.dtype == np.uint8


# =============================================================================
# Legal actions mask
# =============================================================================

class TestLegalActionsMask:

    def test_mask_shape(self, new_standard_game):
        mask = new_standard_game.get_legal_actions_mask()
        assert mask.shape == (ACTION_SPACE_SIZE,)

    def test_mask_matches_legal_actions(self, new_standard_game):
        state = new_standard_game
        legal = state.get_legal_actions()
        mask = state.get_legal_actions_mask()

        assert mask.sum() == len(legal)
        for a in legal:
            assert mask[a] == 1.0


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:

    def test_pass_when_no_moves(self):
        """A color with no remaining pieces has no legal moves."""
        state = GameState.new_game('standard')
        pr = list(state.pieces_remaining)
        pr[0] = frozenset()
        state2 = GameState(
            state.board, tuple(pr), 0, 0,
            state.game_mode, state.pieces, state._has_played)
        assert state2.get_legal_actions() == []

    def test_game_ends_no_crash(self):
        """Full game runs without errors."""
        from game_state import play_random_game
        for seed in [1, 42, 100, 999]:
            final, hist = play_random_game('standard', seed=seed)
            assert final.is_terminal()
            scores = final.get_scores()
            assert all(s >= 0 for s in scores.values())

    def test_dual_game_ends_no_crash(self):
        from game_state import play_random_game
        for seed in [1, 42, 100, 999]:
            final, hist = play_random_game('dual', seed=seed)
            assert final.is_terminal()

    def test_pass_turn_board_shared_not_copied(self):
        """pass_turn shares board array (no copy needed since board is unchanged)."""
        state = GameState.new_game('standard')
        passed = state.pass_turn()
        # Board should be the same object (optimization: no copy for pass)
        assert passed.board is state.board

    def test_legal_actions_deterministic(self, new_standard_game):
        """Same state should always produce same legal actions."""
        a1 = new_standard_game.get_legal_actions()
        a2 = new_standard_game.get_legal_actions()
        assert a1 == a2

    def test_repr(self, new_standard_game):
        r = repr(new_standard_game)
        assert 'standard' in r
        assert 'Yellow' in r
