"""Assignment solvers: a bounded exact search and a deterministic penalised bipartite matching."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from sunwai.planning.voices.types import PlanError

NODE_LIMIT, TIME_LIMIT = 20000, 2.0
BIG = 10**9  # cost of a forbidden pairing in integer matching matrices
Candidates = dict[int, list[tuple[float, str]]]  # speaker id -> [(cost, voice)] best first
Separation = Callable[[str, str], float]


class BoundedSearch:
    """Exact minimum-cost assignment of distinct, mutually separated voices, cut off by node and time limits.

    Args:
        ids: Speaker ids in assignment order.
        candidates: Per-speaker candidate voices with costs.
        separation: Distance between two voices.
        min_separation: Smallest allowed distance between any two chosen voices.
    """

    def __init__(self, ids: list[int], candidates: Candidates, separation: Separation, min_separation: float) -> None:
        """Create the search."""
        self.ids = ids
        self.candidates = candidates
        self.separation = separation
        self.min_separation = min_separation
        self.best_cost: float | None = None
        self.best: list[str] | None = None
        self.limit_hit = False
        self._nodes = 0
        self._started = time.monotonic()

    def run(self) -> list[str] | None:
        """Search; return the best voices (one per id) found, or None if none was found within the limits."""
        self._descend(0, [], 0.0)
        return self.best

    def _out_of_budget(self) -> bool:
        self._nodes += 1
        over_time = self._nodes % 512 == 0 and time.monotonic() - self._started > TIME_LIMIT
        if self._nodes > NODE_LIMIT or over_time:
            self.limit_hit = True
        return self.limit_hit

    def _descend(self, i: int, chosen: list[str], cost: float) -> None:
        if self.limit_hit or self._out_of_budget():
            return
        if self.best_cost is not None and cost >= self.best_cost - 1e-9:
            return
        if i == len(self.ids):
            self.best_cost, self.best = cost, list(chosen)
            return
        for voice_cost, voice in self.candidates[self.ids[i]]:
            if voice in chosen or any(self.separation(voice, other) < self.min_separation for other in chosen):
                continue
            chosen.append(voice)
            self._descend(i + 1, chosen, cost + voice_cost)
            chosen.pop()


def separated_matching(
    ids: list[int],
    pool: list[str],
    base: dict[tuple[int, str], int],
    separation: Separation,
    min_separation: float,
    rounds: int = 8,
) -> list[str]:
    """Min-cost bipartite matching of speakers to distinct voices, nudged toward separated voices.

    Every pair closer than ``min_separation`` penalises the later speaker's voice and the matching is re-solved
    (deterministic: integer costs, lexical column order). The best round (fewest close pairs, then lowest cost) wins.

    Args:
        ids: Speaker ids.
        pool: Candidate voices (matrix columns).
        base: Integer cost per ``(speaker id, voice)``; ``BIG`` forbids a pairing.
        separation: Distance between two voices.
        min_separation: Smallest allowed distance for a pair.
        rounds: Maximum re-solves.

    Raises:
        PlanError: No full matching exists.
    """
    from scipy.optimize import linear_sum_assignment

    penalty: dict[tuple[int, str], int] = {}
    best: tuple[tuple[int, int], list[str]] | None = None
    for _ in range(rounds):
        matrix = [[base[(sid, v)] + penalty.get((sid, v), 0) for v in pool] for sid in ids]
        rows, cols = linear_sum_assignment(matrix)
        if any(matrix[i][j] >= BIG for i, j in zip(rows, cols, strict=True)):
            raise PlanError(f"capacity: no full matching of {len(ids)} speakers to distinct gated voices")
        solution = [pool[j] for _, j in sorted(zip(rows, cols, strict=True))]
        close = [
            (i, j)
            for i in range(len(solution))
            for j in range(i + 1, len(solution))
            if separation(solution[i], solution[j]) < min_separation
        ]
        key = (len(close), sum(base[(ids[i], solution[i])] for i in range(len(solution))))
        if best is None or key < best[0]:
            best = (key, solution)
        if not close:
            break
        for _, j in close:
            penalty[(ids[j], solution[j])] = penalty.get((ids[j], solution[j]), 0) + 30000
    assert best is not None
    return best[1]


def min_cost_matching(matrix: list[list[int]]) -> list[tuple[int, int]] | None:
    """Solve a min-cost assignment; return ``(row, column)`` pairs, or None if a forbidden pairing is needed."""
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(matrix)
    pairs: list[tuple[Any, Any]] = list(zip(rows, cols, strict=True))
    if any(matrix[i][j] >= BIG for i, j in pairs):
        return None
    return [(int(i), int(j)) for i, j in pairs]
