"""Bipartite matching helpers used as branch-and-bound relaxations.

- :func:`max_cardinality` -- Hopcroft-Karp style bound via Kuhn augmenting
  paths (graphs here have at most 12 x 16 nodes).
- :func:`hungarian_min` -- O(n*m*n) Hungarian algorithm for rectangular cost
  matrices with ``n_rows <= n_cols``; every row is assigned to a distinct
  column.  ``None`` entries mean "edge unavailable".
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def max_cardinality(adj: Sequence[Sequence[int]], n_right: int) -> int:
    """Maximum matching size for left nodes with neighbour lists ``adj``."""
    match_r = [-1] * n_right

    def augment(u: int, seen: List[bool]) -> bool:
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] == -1 or augment(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    size = 0
    for u in range(len(adj)):
        if augment(u, [False] * n_right):
            size += 1
    return size


def hungarian_min(
    cost: Sequence[Sequence[Optional[int]]],
) -> Tuple[int, List[int]]:
    """Minimum-cost assignment covering every row (``n_rows <= n_cols``).

    ``cost[i][j]`` is ``None`` when row ``i`` cannot use column ``j``; such
    cells are treated as prohibitively expensive.  Returns
    ``(total_cost, col_of_row)``.  Raises ``ValueError`` if no assignment
    covering every row exists.
    """
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n > m:
        raise ValueError("hungarian_min requires n_rows <= n_cols")
    if n == 0:
        return 0, []

    finite = [c for row in cost for c in row if c is not None]
    if not finite:
        raise ValueError("no finite edge in cost matrix")
    big = sum(abs(c) for row in cost for c in row if c is not None) + 1

    # 1-indexed arrays, cp-algorithms.com formulation (supports n <= m).
    inf = 10 ** 100
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            row_i0 = cost[i0 - 1]
            ui0 = u[i0]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cij = row_i0[j - 1]
                cur = (big if cij is None else cij) - ui0 - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    total = 0
    for j in range(1, m + 1):
        i = p[j]
        if i:
            cij = cost[i - 1][j - 1]
            if cij is None:
                raise ValueError("assignment forced through unavailable edge")
            assignment[i - 1] = j - 1
            total += cij
    return total, assignment
