"""Fibre-arm assignment solver.

Lexicographically optimises, over collision-free assignments:

1. number of assigned arms              (maximise)
2. sum of connected target priorities   (maximise)
3. sum of squared extensions            (minimise)
4. stable assignment sequence           (minimise lexicographically)

The stable sequence is, in arm input order, the 1-based index of the
connected target; an unassigned arm carries a sentinel larger than every
target index.  Hence at the first arm where two otherwise-tied solutions
differ, the one connecting the earlier-listed target wins, and leaving an
arm idle is the last resort.

A depth-first branch-and-bound searches arms in input order.  Relaxations
ignoring the segment-clearance constraint give optimistic bounds:

* maximum cardinality (augmenting-path bipartite matching);
* maximum priority sum (rectangular Hungarian algorithm with dummy columns);
* minimum squared-extension sum of a maximum-cardinality continuation
  (Hungarian algorithm with a cardinality bonus on real edges).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

from .geometry import segment_distance, segments_clearance_ok
from .matching import hungarian_min, max_cardinality


def _stable_key(vec: Tuple[int, ...], n_targets: int) -> Tuple[int, ...]:
    idle = n_targets + 1
    return tuple((j + 1) if j >= 0 else idle for j in vec)


def solve(
    arms: List[Dict],
    targets: List[Dict],
    clearance: float,
) -> Dict:
    """Run the branch-and-bound optimisation.

    Returns a dictionary with the best assignment vector ``vec`` (target
    index per arm, ``-1`` when idle) and the objective metrics.
    """
    n = len(arms)
    m = len(targets)

    # Reachability graph and squared extension lengths.
    reach: List[Tuple[Tuple[int, int], ...]] = []
    for a in arms:
        edges = []
        ax, ay, r2 = a["x"], a["y"], a["reach"] * a["reach"]
        for j, t in enumerate(targets):
            dx, dy = t["x"] - ax, t["y"] - ay
            d2 = dx * dx + dy * dy
            if d2 <= r2:
                edges.append((j, d2))
        reach.append(tuple(edges))

    def segment(i: int, j: int):
        a, t = arms[i], targets[j]
        # Integer coordinates: clearance is decided with exact arithmetic so
        # that coordinates beyond the float64 exact-integer range stay sound.
        return ((a["x"], a["y"]), (t["x"], t["y"]))

    dist_cache: Dict[Tuple[int, int, int, int], Tuple[float, tuple, tuple]] = {}

    def pair_distance(i: int, j: int, i2: int, j2: int):
        key = (i, j, i2, j2)
        hit = dist_cache.get(key)
        if hit is None:
            hit = segment_distance(segment(i, j), segment(i2, j2))
            dist_cache[key] = hit
        return hit

    def compatible(i: int, j: int, placed: List[Tuple[int, int]]) -> bool:
        p1, p2 = segment(i, j)
        for i2, j2 in placed:
            q1, q2 = segment(i2, j2)
            if not segments_clearance_ok(p1, p2, q1, q2, clearance):
                return False
        return True

    # ---- Relaxed bounds (reachability + uniqueness only) ----------------
    @lru_cache(maxsize=None)
    def relaxed_bounds(
        i: int, used_mask: int
    ) -> Tuple[int, int]:
        """``(max_count, max_priority)`` for arms i..n-1.

        Considers reachability and target uniqueness but ignores clearance:
        the values are optimistic upper bounds for every partial placement.
        """
        if i == n:
            return 0, 0
        avail = tuple(j for j in range(m) if not (used_mask >> j) & 1)
        if not avail:
            return 0, 0
        pos = {j: c for c, j in enumerate(avail)}
        n_rows = n - i
        n_cols = len(avail)

        adj = [
            [pos[j] for j, _ in reach[r] if j in pos]
            for r in range(i, n)
        ]
        mu = max_cardinality(adj, n_cols)
        if mu == 0:
            return 0, 0

        # Maximum-priority matching: dummy columns (cost 0) let a row stay
        # unmatched; real edges cost -priority.  Positive priorities make
        # the optimum use the maximum possible number mu of real edges.
        p_cost: List[List[int | None]] = [
            [None] * (n_cols + n_rows) for _ in range(n_rows)
        ]
        for r in range(n_rows):
            for j, _ in reach[i + r]:
                c = pos.get(j)
                if c is not None:
                    p_cost[r][c] = -targets[j]["priority"]
            for d in range(n_rows):
                p_cost[r][n_cols + d] = 0
        _, p_assign = hungarian_min(p_cost)
        pmax = sum(
            targets[avail[c]]["priority"]
            for r, c in enumerate(p_assign)
            if c < n_cols
        )
        return mu, pmax

    def tight_cardinality(
        i: int, used_mask: int, placed: List[Tuple[int, int]]
    ) -> int:
        """Maximum continuation cardinality dropping edges that collide
        with segments already placed (collisions among future segments are
        still ignored, so this stays an optimistic bound)."""
        avail = [j for j in range(m) if not (used_mask >> j) & 1]
        if not avail:
            return 0
        pos = {j: c for c, j in enumerate(avail)}
        adj = []
        for r in range(i, n):
            neigh = []
            for j, _ in reach[r]:
                c = pos.get(j)
                if c is not None and compatible(r, j, placed):
                    neigh.append(c)
            adj.append(neigh)
        return max_cardinality(adj, len(avail))

    def tight_length_bound(
        i: int,
        used_mask: int,
        placed: List[Tuple[int, int]],
        mu: int,
    ) -> int:
        """Minimum squared-extension sum of a cardinality-``mu`` matching
        over edges compatible with the placed segments."""
        avail = [j for j in range(m) if not (used_mask >> j) & 1]
        if mu == 0 or not avail:
            return 0
        pos = {j: c for c, j in enumerate(avail)}
        n_rows = n - i
        n_cols = len(avail)
        edges: List[List[Tuple[int, int]]] = [[] for _ in range(n_rows)]
        max_edge_len = 0
        for r in range(i, n):
            for j, d2 in reach[r]:
                c = pos.get(j)
                if c is not None and compatible(r, j, placed):
                    edges[r - i].append((c, d2))
                    if d2 > max_edge_len:
                        max_edge_len = d2
        # B > total squared length of any matching forces the Hungarian
        # optimum onto exactly mu real edges before minimising their sum.
        bonus = mu * max_edge_len + 1
        cost: List[List[int | None]] = [
            [None] * (n_cols + n_rows) for _ in range(n_rows)
        ]
        for r in range(n_rows):
            for c, d2 in edges[r]:
                cost[r][c] = d2 - bonus
            for d in range(n_rows):
                cost[r][n_cols + d] = 0
        _, assign = hungarian_min(cost)
        length = 0
        for r, c in enumerate(assign):
            if c < n_cols:
                length += next(d2 for cc, d2 in edges[r] if cc == c)
        return length

    # ---- Incumbent from cheap greedy passes ------------------------------
    best_vec: Tuple[int, ...] = tuple([-1] * n)
    best_count = 0
    best_prio = 0
    best_len = 0

    def metrics(vec: Tuple[int, ...]) -> Tuple[int, int, int]:
        cnt = pr = ln = 0
        for i, j in enumerate(vec):
            if j >= 0:
                cnt += 1
                pr += targets[j]["priority"]
                ln += next(d2 for jj, d2 in reach[i] if jj == j)
        return cnt, pr, ln

    def consider(vec: Tuple[int, ...]) -> None:
        nonlocal best_vec, best_count, best_prio, best_len
        cnt, pr, ln = metrics(vec)
        cand = (-cnt, -pr, ln, _stable_key(vec, m))
        cur = (-best_count, -best_prio, best_len, _stable_key(best_vec, m))
        if cand < cur:
            best_vec, best_count, best_prio, best_len = vec, cnt, pr, ln

    def greedy(order: str) -> Tuple[int, ...]:
        vec = [-1] * n
        used = [False] * m
        placed: List[Tuple[int, int]] = []
        for i in range(n):
            cands = [(j, d2) for j, d2 in reach[i] if not used[j]]
            if order == "priority":
                cands.sort(key=lambda e: (-targets[e[0]]["priority"], e[1], e[0]))
            elif order == "length":
                cands.sort(key=lambda e: (e[1], -targets[e[0]]["priority"], e[0]))
            else:
                cands.sort(key=lambda e: e[0])
            for j, _ in cands:
                if compatible(i, j, placed):
                    vec[i] = j
                    used[j] = True
                    placed.append((i, j))
                    break
        return tuple(vec)

    for order in ("index", "priority", "length"):
        consider(greedy(order))

    # ---- Branch and bound ------------------------------------------------
    def dfs(
        i: int,
        used_mask: int,
        count: int,
        prio: int,
        length: int,
        vec: List[int],
        placed: List[Tuple[int, int]],
    ) -> None:
        nonlocal best_vec, best_count, best_prio, best_len

        if count + (n - i) < best_count:
            return

        if i == n:
            tvec = tuple(vec)
            cnt, pr, ln = count, prio, length
            cand = (-cnt, -pr, ln, _stable_key(tvec, m))
            cur = (-best_count, -best_prio, best_len, _stable_key(best_vec, m))
            if cand < cur:
                best_vec, best_count, best_prio, best_len = tvec, cnt, pr, ln
            return

        # Tight cardinality bound: ignore collisions between future
        # segments, but drop edges colliding with segments already placed.
        tight_mu = tight_cardinality(i, used_mask, placed)
        if count + tight_mu < best_count:
            return

        # To improve on the incumbent a continuation must reach best_count.
        # A priority bound only matters when that requires every possible
        # remaining pair; the loose relaxation (cached) stays an upper bound
        # because all priorities are positive.
        if count + tight_mu == best_count:
            _, pmax = relaxed_bounds(i, used_mask)
            if prio + pmax < best_prio:
                return
            if prio + pmax == best_prio:
                # Lower bound on squared extension of a maximum continuation
                # over the placed-segment-compatible graph.
                lmin = tight_length_bound(i, used_mask, placed, tight_mu)
                if length + lmin > best_len:
                    return

                # All optimistic objective bounds tie the incumbent: the
                # fixed prefix must stay able to win on the stable criterion.
                if length + lmin == best_len:
                    cur_key = _stable_key(tuple(vec), m)
                    best_key = _stable_key(best_vec, m)
                    for k in range(i):
                        if cur_key[k] != best_key[k]:
                            if cur_key[k] > best_key[k]:
                                return
                            break

        # This slot is idle unless a candidate below is taken.
        vec[i] = -1

        cands = [
            (j, d2)
            for j, d2 in reach[i]
            if not (used_mask >> j) & 1 and compatible(i, j, placed)
        ]
        cands.sort(key=lambda e: e[0])  # ascending target index: stable dive

        for j, d2 in cands:
            vec[i] = j
            placed.append((i, j))
            dfs(
                i + 1,
                used_mask | (1 << j),
                count + 1,
                prio + targets[j]["priority"],
                length + d2,
                vec,
                placed,
            )
            placed.pop()
            vec[i] = -1

        # Leave arm i idle.
        dfs(i + 1, used_mask, count, prio, length, vec, placed)

    dfs(0, 0, 0, 0, 0, [-1] * n, [])

    return {
        "vec": best_vec,
        "count": best_count,
        "priority_sum": best_prio,
        "length_sq_sum": best_len,
        "pair_distance": pair_distance,
    }
