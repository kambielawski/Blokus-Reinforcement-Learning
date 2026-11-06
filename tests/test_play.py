"""Tests for piece normalization, the human-play session, and the bots."""

import json
import time

import numpy as np
import pytest
import torch

from blokus.agents.bots import GreedySizeBot, LookaheadBot, MobilityBot, RandomBot, SpaceBot
from blokus.engine.game_state import (
    COLOR_CORNERS, COLOR_NAMES, GameState, decode_action, encode_action, load_pieces,
)
from blokus.nn.network import BlokusNetwork
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
        # Blue (the human) has placed everything, so it can only pass; Yellow can still move
        fresh = GameState.new_game('dual')
        all_ids = frozenset(range(21))
        session.state = GameState(fresh.board, (frozenset(), all_ids, frozenset(), frozenset()),
                                  0, 0, 'dual', fresh.pieces, (True, False, True, True))
        session.start()
        wait_until_idle(session)
        moves = session.view()['moves']
        assert moves[0]['type'] == 'pass' and moves[0]['actor'] == 'human'
        assert moves[1]['type'] == 'place' and moves[1]['actor'] == 'ai'

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
        lambda: LookaheadBot(seed=0),
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


class TestRotationSymmetry:
    @pytest.mark.parametrize('plies', [0, 1, 2, 7, 20, 35, 50])
    @pytest.mark.parametrize('k', [1, 2, 3])
    def test_rotation_preserves_legal_moves_and_network_input(self, plies, k):
        from blokus.engine.symmetry import rotate_state, rotation_action_tables
        from blokus.nn.network import make_pieces_remaining_vector
        state = play_random_plies(plies, seed=plies + 100)
        turned = rotate_state(state, k)
        table = rotation_action_tables()[k]
        assert {int(table[a]) for a in state.get_legal_actions()} == set(turned.get_legal_actions())
        assert np.array_equal(np.rot90(state.get_nn_state(), k=-k, axes=(1, 2)), turned.get_nn_state())
        assert np.array_equal(make_pieces_remaining_vector(state),
                              make_pieces_remaining_vector(turned))
        scores, turned_scores = state.get_scores(), turned.get_scores()
        assert all(scores[c] == turned_scores[(c + k) % 4] for c in range(4))

    def test_torch_rotation_matches_state_rotation(self):
        """The training augmentation must turn tensors the same way as the action tables."""
        from blokus.engine.symmetry import rotate_state
        state = play_random_plies(20, seed=5)
        for k in range(1, 4):
            tensor = torch.from_numpy(state.get_nn_state())
            turned = torch.rot90(tensor, k=-k, dims=(-2, -1)).numpy()
            assert np.array_equal(turned, rotate_state(state, k).get_nn_state())

    def test_rotated_action_covers_rotated_cells(self):
        from blokus.engine.symmetry import rotation_action_tables
        tables = rotation_action_tables()
        pieces = load_pieces()
        for action in np.nonzero(tables[1] >= 0)[0][::97]:
            p, o, r, c = decode_action(int(action))
            p2, o2, r2, c2 = decode_action(int(tables[1][action]))
            assert p == p2
            cells = {(y + c, 19 - (x + r)) for x, y in pieces[p].orientations[o].occupied}
            assert cells == {(x + r2, y + c2) for x, y in pieces[p2].orientations[o2].occupied}
            assert tables[3][tables[1][action]] == action
            assert tables[0][action] == action


class TestTableRules:
    """Examples are from the Mattel rule sheets R1983-0920 and BJV44-0920."""

    def test_colors_run_clockwise_from_top_left(self):
        assert [COLOR_NAMES[c] for c in range(4)] == ['Blue', 'Yellow', 'Red', 'Green']
        assert [COLOR_CORNERS[c] for c in range(4)] == [(0, 0), (0, 19), (19, 19), (19, 0)]

    def test_each_player_holds_opposite_corners(self):
        state = GameState.new_game('dual')
        agents = [state.get_current_agent()]
        for _ in range(3):
            state = state.pass_turn()
            agents.append(state.get_current_agent())
        assert agents == [0, 1, 0, 1]  # Blue and Red, then Yellow and Green

    def _finished_color(self, last_piece: int) -> GameState:
        """Blue has placed every piece, with last_piece played last; nothing else moved."""
        state = GameState.new_game('dual')
        remaining = (frozenset(), frozenset(range(21)), frozenset(range(21)), frozenset(range(21)))
        return GameState(state.board, remaining, 0, 0, 'dual', state.pieces,
                         (True, False, False, False), (last_piece, None, None, None))

    def test_score_all_pieces_with_monomino_last(self):
        assert self._finished_color(0).get_scores()[0] == 20  # Sheet example: blue +20

    def test_score_all_pieces_other_last(self):
        assert self._finished_color(4).get_scores()[0] == 15

    def test_score_counts_remaining_squares(self):
        """Sheet example: green left 1 three-square, 3 four-square and 1 five-square: -20."""
        state = GameState.new_game('dual')
        left = frozenset({2, 3, 6, 18, 4})  # I3, I4, L4, T4, I5
        remaining = (frozenset(range(21)), frozenset(range(21)), frozenset(range(21)), left)
        scored = GameState(state.board, remaining, 0, 0, 'dual', state.pieces,
                           (False, False, False, True))
        assert scored.get_scores()[3] == -20
        assert scored.get_scores()[0] == -89

    def test_placing_all_pieces_does_not_end_the_game(self):
        state = self._finished_color(0)
        assert not state.is_terminal()
        assert state.get_legal_actions() == []  # Blue only passes now
        state = state.pass_turn()
        assert not state.is_terminal()

    def test_game_ends_only_after_all_four_pass(self):
        state = GameState.new_game('dual')
        for expected in (False, False, False, True):
            state = state.pass_turn()
            assert state.is_terminal() == expected

    def test_two_player_score_sums_both_colors(self):
        state = play_random_plies(100, seed=1)
        scores = state.get_scores()
        rewards = state.get_rewards()
        a, b = scores[0] + scores[2], scores[1] + scores[3]
        assert rewards[0] == np.sign(a - b)

    def test_last_piece_tracked_through_play(self):
        state = GameState.new_game('dual').apply_action(encode_action(0, 0, 0, 0))
        assert state._last_piece[0] == 0
        for _ in range(3):
            state = state.pass_turn()
        assert state._last_piece[0] == 0
        assert state.copy()._last_piece[0] == 0

    def test_old_rules_checkpoint_is_rejected(self, tmp_path):
        from blokus.nn.checkpoint import load_network
        path = tmp_path / 'old.pt'
        torch.save({'model_state_dict': {}, 'config': {}}, path)
        with pytest.raises(ValueError, match='old-rules-final'):
            load_network(str(path), torch.device('cpu'))


class TestSearchUnderTableRules:
    def test_forced_passes_use_no_simulations(self):
        """Once only one color can move, every simulation reaches a real decision."""
        from blokus.mcts.mcts import MCTS, MCTSNode
        calls = []

        class CountingNet(BlokusNetwork):
            def forward(self, *args, **kwargs):
                calls.append(args[0].shape[0])
                return super().forward(*args, **kwargs)

        network = CountingNet(num_blocks=1, channels=8)
        state = GameState.new_game('dual')
        # Yellow, Red and Green have no pieces, so only Blue moves. Each Blue move is
        # followed by three forced passes.
        no_pieces = frozenset()
        state = GameState(state.board, (frozenset(range(21)), no_pieces, no_pieces, no_pieces),
                          0, 0, 'dual', state.pieces, (False, True, True, True))
        mcts = MCTS(network, num_simulations=8, dirichlet_alpha=0.0, batch_size=1)
        mcts.search(state)
        assert sum(calls) == 9  # Root plus one leaf per simulation, none spent on passes

    def test_terminal_value_uses_margin_scale(self):
        from blokus.mcts.mcts import MCTS, MCTSNode
        state = GameState.new_game('dual')
        for _ in range(4):
            state = state.pass_turn()
        finished = GameState(state.board, (frozenset(), frozenset(range(21)), frozenset(range(21)),
                                           frozenset(range(21))), 0, 4, 'dual', state.pieces,
                             (True, False, False, False), (0, None, None, None))
        network = BlokusNetwork(num_blocks=1, channels=8)
        mcts = MCTS(network, terminal_margin_scale=20.0)
        node = MCTSNode(finished)
        margin = (20 - 89) - (-89 - 89)
        assert mcts._terminal_value(node) == pytest.approx(np.tanh(margin / 20))
