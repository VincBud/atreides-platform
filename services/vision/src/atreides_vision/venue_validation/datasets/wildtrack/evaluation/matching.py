"""Deterministic bipartite matching for the offline WILDTRACK evaluator.

The evaluator has to decide, per camera and per venue instant, which runtime
observation stands in for which WILDTRACK annotated person before any
association figure can be computed. That decision must be:

* **maximum cardinality first** — never leave a feasible observation/person
  pair unmatched just because a greedy nearest-neighbour walk consumed a
  partner early; a missed local match silently deflates recall;
* **minimum total distance second** — among all maximum-cardinality matchings,
  the one that moves observations the least onto their assigned person;
* **canonically deterministic third** — where several minimum-distance
  maximum-cardinality matchings exist (a symmetric 2x2 swap has identical
  cost), the lower observation index takes the lower person index, so the
  result never depends on iteration order or a solver's internal tie-breaking.

This module imports nothing from :mod:`atreides_vision.association` — it is the
evaluator's own matcher and shares no code with the runtime association engine.
It contains no scipy dependency: distances are integer micrometres and the
assignment is solved with a self-contained successive-shortest-augmenting-path
min-cost-max-flow followed by a canonical tie-break pass.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "LocalMatch",
    "distance_micrometres",
    "match_by_distance",
    "match_points",
]

#: Micrometres per metre. Distances are rounded to an integer number of these
#: so the matcher is exact and platform-independent.
_MICROMETRES_PER_METRE = 1_000_000


def distance_micrometres(a: tuple[float, float], b: tuple[float, float]) -> int:
    """Euclidean distance between two ground-plane metre points, in micrometres."""
    separation_m = math.hypot(a[0] - b[0], a[1] - b[1])
    return round(separation_m * _MICROMETRES_PER_METRE)


@dataclass(frozen=True, slots=True)
class LocalMatch:
    """The outcome of matching one camera's observations to its GT persons.

    ``pairs`` is ``(row, column)`` index pairs — ``row`` into the observation
    sequence, ``column`` into the person sequence — sorted ascending by row.
    ``total_distance_micrometres`` is the summed assignment cost.
    """

    pairs: tuple[tuple[int, int], ...]
    total_distance_micrometres: int

    @property
    def matched_rows(self) -> frozenset[int]:
        return frozenset(row for row, _ in self.pairs)

    @property
    def matched_columns(self) -> frozenset[int]:
        return frozenset(column for _, column in self.pairs)

    def column_for_row(self, row: int) -> int | None:
        for candidate_row, column in self.pairs:
            if candidate_row == row:
                return column
        return None


def match_points(
    row_points: Sequence[tuple[float, float]],
    column_points: Sequence[tuple[float, float]],
    *,
    radius_metres: float,
) -> LocalMatch:
    """Match two sets of ground-plane metre points within ``radius_metres``.

    A row/column pair is feasible only when their separation is
    ``<= radius_metres`` (inclusive). See :func:`match_by_distance` for the
    ordering guarantees.
    """
    radius_um = round(radius_metres * _MICROMETRES_PER_METRE)
    distances: list[list[int]] = [
        [distance_micrometres(row_point, column_point) for column_point in column_points]
        for row_point in row_points
    ]
    return match_by_distance(distances, radius_micrometres=radius_um)


def match_by_distance(
    distances_micrometres: Sequence[Sequence[int]],
    *,
    radius_micrometres: int,
) -> LocalMatch:
    """Maximum-cardinality, then minimum-distance, then canonical assignment.

    Args:
        distances_micrometres: ``distances[row][column]`` integer micrometre
            distance between row ``row`` and column ``column``.
        radius_micrometres: inclusive feasibility bound; a pair is a candidate
            edge only when its distance is ``<= radius_micrometres``.
    """
    row_count = len(distances_micrometres)
    column_count = len(distances_micrometres[0]) if row_count else 0

    edges: dict[tuple[int, int], int] = {}
    for row in range(row_count):
        row_distances = distances_micrometres[row]
        for column in range(column_count):
            cost = row_distances[column]
            if cost <= radius_micrometres:
                edges[(row, column)] = cost

    target_cardinality, target_cost = _min_cost_max_cardinality(
        list(range(row_count)), list(range(column_count)), edges
    )

    row_to_column: list[int] = [-1] * row_count
    used_columns: set[int] = set()
    fixed_cardinality = 0
    fixed_cost = 0

    for row in range(row_count):
        remaining_rows = list(range(row + 1, row_count))
        feasible_columns = sorted(
            column
            for column in range(column_count)
            if column not in used_columns and (row, column) in edges
        )
        chosen_column = -1
        for column in feasible_columns:
            available_columns = [
                candidate
                for candidate in range(column_count)
                if candidate not in used_columns and candidate != column
            ]
            sub_edges = {
                (sub_row, sub_column): weight
                for (sub_row, sub_column), weight in edges.items()
                if sub_row > row and sub_column in available_columns
            }
            sub_cardinality, sub_cost = _min_cost_max_cardinality(
                remaining_rows, available_columns, sub_edges
            )
            if (
                fixed_cardinality + 1 + sub_cardinality == target_cardinality
                and fixed_cost + edges[(row, column)] + sub_cost == target_cost
            ):
                chosen_column = column
                break
        if chosen_column == -1:
            continue
        row_to_column[row] = chosen_column
        used_columns.add(chosen_column)
        fixed_cardinality += 1
        fixed_cost += edges[(row, chosen_column)]

    pairs = tuple((row, column) for row, column in enumerate(row_to_column) if column != -1)
    return LocalMatch(pairs=pairs, total_distance_micrometres=fixed_cost)


def _min_cost_max_cardinality(
    row_ids: Sequence[int],
    column_ids: Sequence[int],
    edges: dict[tuple[int, int], int],
) -> tuple[int, int]:
    """(cardinality, total cost) of a min-cost maximum-cardinality matching.

    ``edges`` is keyed by ``(row_id, column_id)`` using the *caller's* ids; only
    entries whose endpoints appear in ``row_ids``/``column_ids`` are used.
    """
    row_index = {row_id: index for index, row_id in enumerate(row_ids)}
    column_index = {column_id: index for index, column_id in enumerate(column_ids)}
    compact_edges: dict[tuple[int, int], int] = {}
    for (row_id, column_id), weight in edges.items():
        if row_id in row_index and column_id in column_index:
            compact_edges[(row_index[row_id], column_index[column_id])] = weight
    return _solve_min_cost_max_flow(len(row_ids), len(column_ids), compact_edges)


def _solve_min_cost_max_flow(
    row_count: int,
    column_count: int,
    edges: dict[tuple[int, int], int],
) -> tuple[int, int]:
    source = 0
    sink = row_count + column_count + 1
    node_count = sink + 1

    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    edge_to: list[int] = []
    edge_capacity: list[int] = []
    edge_cost: list[int] = []

    def add_edge(tail: int, head: int, capacity: int, cost: int) -> None:
        adjacency[tail].append(len(edge_to))
        edge_to.append(head)
        edge_capacity.append(capacity)
        edge_cost.append(cost)
        adjacency[head].append(len(edge_to))
        edge_to.append(tail)
        edge_capacity.append(0)
        edge_cost.append(-cost)

    for row in range(row_count):
        add_edge(source, 1 + row, 1, 0)
    for column in range(column_count):
        add_edge(1 + row_count + column, sink, 1, 0)
    for (row, column), weight in edges.items():
        add_edge(1 + row, 1 + row_count + column, 1, weight)

    total_cost = 0
    cardinality = 0

    while True:
        distance: list[float] = [math.inf] * node_count
        in_queue = [False] * node_count
        previous_edge: list[int] = [-1] * node_count
        distance[source] = 0.0
        queue: deque[int] = deque([source])
        in_queue[source] = True
        while queue:
            node = queue.popleft()
            in_queue[node] = False
            for edge_id in adjacency[node]:
                if edge_capacity[edge_id] <= 0:
                    continue
                head = edge_to[edge_id]
                candidate = distance[node] + edge_cost[edge_id]
                if candidate < distance[head]:
                    distance[head] = candidate
                    previous_edge[head] = edge_id
                    if not in_queue[head]:
                        in_queue[head] = True
                        queue.append(head)
        if previous_edge[sink] == -1:
            break
        node = sink
        while node != source:
            edge_id = previous_edge[node]
            edge_capacity[edge_id] -= 1
            edge_capacity[edge_id ^ 1] += 1
            node = edge_to[edge_id ^ 1]
        cardinality += 1
        total_cost += int(distance[sink])

    return cardinality, total_cost
