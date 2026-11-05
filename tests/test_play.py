"""Tests for piece normalization, the human-play session, and the bots."""

import json
import time

import numpy as np
import pytest
import torch

from blokus.agents.bots import GreedySizeBot, MobilityBot, RandomBot, SpaceBot
from blokus.engine.game_state import (
    COLOR_CORNERS, GameState, decode_action, encode_action, load_pieces,
)
from blokus.nn.network import BlokusNetwork, LEGACY_POLICY_SHIFTS
from blokus.ui.session import (
    GameSession, IllegalMove, explain_illegal, piece_catalog,
)


def wait_until_idle(session: GameSession, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = session.view()
        if view['your_turn'] or view['status'] != 'playing':
            return
        time.sleep(0.005)
    raise TimeoutError('AI did not finish')


def play_random_plies(n: int, seed: int) -> GameState:
    rng = np.random.default_rng(seed)
    state = GameState.new_game('dual')
    for _ in range(n):
        if state.is_terminal():
            break
        legal = state.get_legal_actions()
        state = state.apply_action(legal[rng.integers(len(legal))]) if legal else state.pass_turn()
    return state


class TestPieceNormalization:
    def test_every_orientation_starts_at_origin(self):
        for piece in load_pieces():
            for ori in piece.orientations:
                assert min(r for r, c in ori.occupied) == 0
                assert min(c for r, c in ori.occupied) == 0

    @pytest.mark.parametrize('color', [0, 1, 2, 3])
    def test_every_piece_except_x_can_open_in_every_corner(self, color):
        state = GameState.new_game('dual')
        for _ in range(color):
            state = state.pass_turn()
        playable = {decode_action(a)[0] for a in state.get_legal_actions()}
        assert playable == set(range(21)) - {12}  # The X cannot cover a corner

    def test_square_reaches_top_left_cell(self):
        state = GameState.new_game('dual')
        assert encode_action(19, 0, 0, 0) in state.get_legal_actions()

    def test_legacy_policy_shift_targets_the_same_placement(self):
        """A version 1 logit at (r, c) must land on the new action covering the same cells."""
        torch.manual_seed(0)
        network = BlokusNetwork(num_blocks=1, channels=8)
        network.eval()
        state = play_random_plies(12, seed=1)
        board = torch.from_numpy(state.get_nn_state()).unsqueeze(0)
        pieces = torch.rand(1, 84)
        mask = torch.ones(1, 67200)
        with torch.no_grad():
            plain, _ = network(board, pieces, mask)
            network.legacy_action_encoding = True
            legacy, _ = network(board, pieces, mask)
        plain = plain[0].reshape(168, 20, 20)
        legacy = legacy[0].reshape(168, 20, 20)
        # Renormalization adds one constant to all log-probabilities, so compare
        # after removing it. The I1 channel does not move, which measures it.
        constant = (legacy[0] - plain[0]).mean()
        assert torch.allclose(legacy[0] - constant, plain[0], atol=1e-4)
        for channel, dr, dc in LEGACY_POLICY_SHIFTS:
            assert torch.allclose(legacy[channel, dr:, dc:] - constant,
                                  plain[channel, :20 - dr, :20 - dc], atol=1e-4)


class TestPieceCatalog:
    def test_rotate_cycles_and_flip_is_involution(self):
        for piece in piece_catalog():
            for o in range(len(piece['orientations'])):
                r = o
                for _ in range(4):
                    r = piece['rotate'][r]
                assert r == o
                assert piece['flip'][piece['flip'][o]] == o

    def test_rotate_and_flip_reach_every_orientation(self):
        for piece in piece_catalog():
            seen, frontier = {0}, [0]
            while frontier:
                o = frontier.pop()
                for nxt in (piece['rotate'][o], piece['flip'][o]):
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)
            assert len(seen) == len(piece['orientations'])

    def test_rotate_is_clockwise(self):
        i2 = piece_catalog()[1]  # Horizontal domino turns vertical
        assert i2['orientations'][0] == [[0, 0], [0, 1]]
        assert i2['orientations'][i2['rotate'][0]] == [[0, 0], [1, 0]]


class TestExplainIllegal:
    @pytest.mark.parametrize('plies', [0, 1, 5, 12, 25, 40, 55])
    def test_agrees_with_engine_on_every_placement(self, plies):
        state = play_random_plies(plies, seed=plies)
        if state.is_terminal():
            pytest.skip('Game ended early')
        legal = set(state.get_legal_actions())
        for pid, piece in enumerate(state.pieces):
            for oid, ori in enumerate(piece.orientations):
                for row in range(20 - ori.max_row):
                    for col in range(20 - ori.max_col):
                        reason = explain_illegal(state, pid, oid, row, col)
                        is_legal = encode_action(pid, oid, row, col) in legal
                        assert (reason is None) == is_legal, (pid, oid, row, col, reason)

    def test_reasons(self):
        state = GameState.new_game('dual')
        assert 'corner' in explain_illegal(state, 1, 0, 5, 5)
        assert 'board' in explain_illegal(state, 4, 0, 0, 18)
        state = state.apply_action(encode_action(1, 0, 0, 0))  # Yellow domino at corner
        for _ in range(3):
            state = state.pass_turn()
        assert 'already played' in explain_illegal(state, 1, 0, 1, 1)
        assert 'overlaps' in explain_illegal(state, 0, 0, 0, 0)
        assert 'shares an edge' in explain_illegal(state, 0, 0, 1, 0)
        assert 'corner to corner' in explain_illegal(state, 0, 0, 5, 5)
        assert explain_illegal(state, 0, 0, 1, 2) is None


class TestGameSession:
    def test_ai_moves_first_when_human_is_second(self, tmp_path):
        session = GameSession(RandomBot(0), 1, str(tmp_path))
        session.start()
        wait_until_idle(session)
        view = session.view()
        assert view['your_turn'] and view['current_color'] == 1
        assert view['moves'][0]['actor'] == 'ai'

    def test_rejects_illegal_and_out_of_turn_moves(self, tmp_path):
        session = GameSession(RandomBot(0), 0, str(tmp_path))
        session.start()
        wait_until_idle(session)
        with pytest.raises(IllegalMove, match='corner'):
            session.human_move(0, 0, 10, 10)
        with pytest.raises(IllegalMove, match='on the board'):
            session.human_move(4, 0, 0, 19)
        assert session.view()['moves'] == []

        session.bot.select_action = lambda state: time.sleep(0.3) or RandomBot(0).select_action(state)
        session.human_move(0, 0, 0, 0)
        assert session.view()['thinking']
        with pytest.raises(IllegalMove, match='not your turn'):
            session.human_move(1, 0, 1, 1)
        wait_until_idle(session)

    @pytest.mark.parametrize('human', [0, 1])
    def test_full_game_with_passes_and_scoring(self, tmp_path, human):
        rng = np.random.default_rng(human)
        session = GameSession(GreedySizeBot(seed=0), human, str(tmp_path), {'name': 'greedy'})
        session.start()
        while True:
            wait_until_idle(session)
            view = session.view()
            if view['status'] != 'playing':
                break
            piece, orientation, row, col = view['legal'][rng.integers(len(view['legal']))]
            session.human_move(piece, orientation, row, col)

        assert view['status'] == 'finished' and view['error'] is None
        result = view['result']
        scores = session.state.get_scores()
        mine = scores[human] + scores[human + 2]
        theirs = scores[1 - human] + scores[3 - human]
        assert (result['human_score'], result['ai_score']) == (mine, theirs)
        assert result['margin'] == mine - theirs

        with open(view['record_path']) as f:
            record = json.load(f)
        assert record['status'] == 'finished'
        assert record['human_agent'] == human
        assert record['result'] == result

        # Replaying the record must reproduce the final position exactly
        replay = GameState.new_game('dual')
        for move in record['moves']:
            assert move['color'] == replay.current_color
            replay = replay.pass_turn() if move['type'] == 'pass' else replay.apply_action(move['action'])
        assert np.array_equal(replay.board, session.state.board)
        assert replay.is_terminal()

    def test_passes_are_automatic_for_both_sides(self, tmp_path):
        """A color with no moves passes without waiting on anyone."""
        session = GameSession(RandomBot(0), 0, str(tmp_path))
        state = play_random_plies(70, seed=3)
        while state.get_legal_actions() or state.get_current_agent() != 0:
            if state.is_terminal():
                pytest.skip('No stuck human color in this random game')
            legal = state.get_legal_actions()
            state = state.apply_action(legal[0]) if legal else state.pass_turn()
        session.state = state
        session.start()
        wait_until_idle(session)
        moves = session.view()['moves']
        assert moves and moves[0]['type'] == 'pass' and moves[0]['actor'] == 'human'

    def test_abandon_is_recorded_and_discards_pending_ai_move(self, tmp_path):
        session = GameSession(RandomBot(0), 1, str(tmp_path))
        session.bot.select_action = lambda state: time.sleep(0.2) or state.get_legal_actions()[0]
        session.start()
        session.abandon()
        time.sleep(0.3)
        with open(session.record_path) as f:
            record = json.load(f)
        assert record['status'] == 'abandoned'
        assert record['moves'] == []


class TestBots:
    @pytest.mark.parametrize('make_bot', [
        lambda: RandomBot(0), lambda: GreedySizeBot(seed=0),
        lambda: SpaceBot(False, 0), lambda: SpaceBot(True, 0), lambda: MobilityBot(seed=0),
    ])
    def test_bots_only_play_legal_moves(self, make_bot):
        bot = make_bot()
        state = GameState.new_game('dual')
        while not state.is_terminal():
            legal = state.get_legal_actions()
            if not legal:
                state = state.pass_turn()
                continue
            action = bot.select_action(state)
            assert action in legal
            state = state.apply_action(action)

    def test_greedy_size_opens_with_a_pentomino(self):
        state = GameState.new_game('dual')
        action = GreedySizeBot(seed=0).select_action(state)
        assert state.pieces[decode_action(action)[0]].num_cells == 5


class TestMirrorSymmetry:
    @pytest.mark.parametrize('plies', [0, 1, 2, 7, 20, 35, 50])
    def test_mirror_preserves_legal_moves_and_network_input(self, plies):
        from blokus.engine.symmetry import mirror_action_table, mirror_state
        from blokus.nn.network import make_pieces_remaining_vector
        state = play_random_plies(plies, seed=plies + 100)
        mirrored = mirror_state(state)
        table = mirror_action_table()
        assert {int(table[a]) for a in state.get_legal_actions()} == set(mirrored.get_legal_actions())
        assert np.array_equal(state.get_nn_state()[:, :, ::-1], mirrored.get_nn_state())
        assert np.array_equal(make_pieces_remaining_vector(state),
                              make_pieces_remaining_vector(mirrored))
        assert state.get_scores()[0] == mirrored.get_scores()[2]

    def test_mirror_of_action_covers_mirrored_cells(self):
        from blokus.engine.symmetry import mirror_action_table
        table = mirror_action_table()
        pieces = load_pieces()
        for action in np.nonzero(table >= 0)[0][::97]:
            p, o, r, c = decode_action(int(action))
            p2, o2, r2, c2 = decode_action(int(table[action]))
            assert p == p2
            cells = {(x + r, 19 - (y + c)) for x, y in pieces[p].orientations[o].occupied}
            assert cells == {(x + r2, y + c2) for x, y in pieces[p2].orientations[o2].occupied}
            assert table[table[action]] == action
