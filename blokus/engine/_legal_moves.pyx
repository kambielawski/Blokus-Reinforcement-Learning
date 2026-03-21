# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated legal move generation for Blokus.

Replaces the numpy-based _fast_legal_actions with tight C loops over
pre-computed placement data. Avoids intermediate numpy arrays entirely.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset

# Board and piece constants (must match game_state.py)
DEF BOARD_SIZE = 20
DEF NUM_PIECES = 21
DEF MAX_PIECE_CELLS = 5
DEF STRIDE = BOARD_SIZE + 1          # padded width for flat indexing
DEF PAD_TOTAL = STRIDE * STRIDE      # total cells in padded grid


def fast_legal_actions(
    np.ndarray[np.int8_t, ndim=2] board,
    tuple pieces_remaining,
    int current_color,
    tuple has_played,
    object pieces,                    # List[PieceInfo] — only used for cache init
    object data,                      # _FastLegalData
    dict color_corners,
):
    """Drop-in replacement for _fast_legal_actions.

    Uses typed memoryviews and C loops instead of numpy temporaries.
    """
    cdef int ci = current_color
    cdef int cv = ci + 1
    cdef frozenset remaining = pieces_remaining[ci]

    if not remaining:
        return []

    # ---- Typed views into pre-computed data ----
    cdef np.int32_t[:, :] flat_indices = data.flat_indices   # (TOTAL, 5)
    cdef np.int32_t[:] actions = data.actions                # (TOTAL,)
    cdef np.int32_t[:] csr_offsets = data.csr_offsets        # (PAD_TOTAL+1,)
    cdef np.int32_t[:] csr_data = data.csr_data              # (sum of lists,)
    cdef int total = data.total

    # ---- Build reject mask (occupied | orthogonal-adjacent) on the stack ----
    # Padded to (STRIDE, STRIDE) so sentinel coords index to zeros.
    cdef unsigned char reject[STRIDE * STRIDE]
    memset(reject, 0, STRIDE * STRIDE)

    cdef int r, c
    cdef np.int8_t[:, :] b = board

    # First pass: mark occupied cells and find same-color cells
    cdef unsigned char is_color[BOARD_SIZE * BOARD_SIZE]
    memset(is_color, 0, BOARD_SIZE * BOARD_SIZE)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if b[r, c] != 0:
                reject[r * STRIDE + c] = 1        # occupied
            if b[r, c] == cv:
                is_color[r * BOARD_SIZE + c] = 1

    # Second pass: mark forbidden (orthogonally adjacent to same-color)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if is_color[r * BOARD_SIZE + c]:
                if r > 0:
                    reject[(r - 1) * STRIDE + c] = 1
                if r < BOARD_SIZE - 1:
                    reject[(r + 1) * STRIDE + c] = 1
                if c > 0:
                    reject[r * STRIDE + (c - 1)] = 1
                if c < BOARD_SIZE - 1:
                    reject[r * STRIDE + (c + 1)] = 1

    # ---- Find anchor cells ----
    cdef int n_anchors = 0
    cdef int anchor_buf[512]   # max realistic anchors (typically 20-80)

    if not has_played[ci]:
        # First move: single corner cell
        corner = color_corners[ci]
        anchor_buf[0] = corner[0] * STRIDE + corner[1]
        n_anchors = 1
    else:
        # Diagonal adjacencies to same-color cells, excluding occupied
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if b[r, c] != 0:
                    continue  # skip occupied
                # Check if diagonally adjacent to same-color
                if (r > 0 and c > 0 and is_color[(r-1) * BOARD_SIZE + (c-1)]):
                    anchor_buf[n_anchors] = r * STRIDE + c
                    n_anchors += 1
                    continue
                if (r > 0 and c < BOARD_SIZE - 1 and is_color[(r-1) * BOARD_SIZE + (c+1)]):
                    anchor_buf[n_anchors] = r * STRIDE + c
                    n_anchors += 1
                    continue
                if (r < BOARD_SIZE - 1 and c > 0 and is_color[(r+1) * BOARD_SIZE + (c-1)]):
                    anchor_buf[n_anchors] = r * STRIDE + c
                    n_anchors += 1
                    continue
                if (r < BOARD_SIZE - 1 and c < BOARD_SIZE - 1 and is_color[(r+1) * BOARD_SIZE + (c+1)]):
                    anchor_buf[n_anchors] = r * STRIDE + c
                    n_anchors += 1
                    continue

        if n_anchors == 0:
            return []

    # ---- Gather candidates from CSR and mark them ----
    # Use a byte array as a bitset for candidate placements
    cdef unsigned char* cmask = <unsigned char*>malloc(total)
    if cmask == NULL:
        raise MemoryError("Failed to allocate cmask")
    memset(cmask, 0, total)

    cdef int ac, s, e, k
    for k in range(n_anchors):
        ac = anchor_buf[k]
        s = csr_offsets[ac]
        e = csr_offsets[ac + 1]
        while s < e:
            cmask[csr_data[s]] = 1
            s += 1

    # ---- Filter by remaining pieces ----
    # Build a lookup: is piece_id in remaining?
    cdef unsigned char pid_ok[NUM_PIECES]
    memset(pid_ok, 0, NUM_PIECES)
    for pid in remaining:
        pid_ok[<int>pid] = 1

    # Use per-piece masks from data to zero out non-remaining candidates
    cdef int i, j
    cdef list pid_masks = data.pid_masks  # list of 21 bool np arrays

    # Instead of using pid_masks (python objects), decode pid from action encoding
    # action = pid * 3200 + ... => pid = action // 3200
    for i in range(total):
        if cmask[i]:
            if not pid_ok[actions[i] // 3200]:
                cmask[i] = 0

    # ---- Validate: no rejected cells in placement ----
    cdef list result = []
    cdef int f0, f1, f2, f3, f4

    for i in range(total):
        if cmask[i] == 0:
            continue
        f0 = flat_indices[i, 0]
        f1 = flat_indices[i, 1]
        f2 = flat_indices[i, 2]
        f3 = flat_indices[i, 3]
        f4 = flat_indices[i, 4]
        if (reject[f0] | reject[f1] | reject[f2] | reject[f3] | reject[f4]) == 0:
            result.append(actions[i])

    free(cmask)
    return result
