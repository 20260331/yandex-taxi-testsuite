"""Deterministic capability dependency graph."""

from __future__ import annotations

import dataclasses
import heapq
from collections.abc import Iterable

from .registry import CapabilitySpec


@dataclasses.dataclass(frozen=True)
class CapabilityEdge:
    """Dependency edge: ``source`` capability requires ``target``."""

    source: str
    target: str
    optional: bool


@dataclasses.dataclass(frozen=True)
class CapabilityGraph:
    #: All capability names, sorted.
    nodes: tuple[str, ...]
    #: Dependency edges, sorted by (source, target).
    edges: tuple[CapabilityEdge, ...]
    #: Deterministic setup order, dependencies come first.
    order: tuple[str, ...]
    #: Cycles, each given as a closed path (last node points to first).
    cycles: tuple[tuple[str, ...], ...]


def build_graph(specs: Iterable[CapabilitySpec]) -> CapabilityGraph:
    """Build graph from unique capability specs.

    Input is expected to contain a single spec per capability name
    (duplicate providers are handled by the diagnostics layer).
    """

    by_name: dict[str, CapabilitySpec] = {}
    for spec in specs:
        by_name.setdefault(spec.name, spec)
    nodes = tuple(sorted(by_name))

    adjacency: dict[str, set[str]] = {name: set() for name in nodes}
    # An edge is required unless every requirement for that pair is optional.
    edge_optional: dict[tuple[str, str], bool] = {}
    for spec in (by_name[name] for name in nodes):
        for requirement in spec.requires:
            if requirement.name not in by_name:
                # Missing dependency is reported by the diagnostics layer.
                continue
            key = (spec.name, requirement.name)
            edge_optional[key] = edge_optional.get(key, True) and (
                requirement.optional
            )
            adjacency[spec.name].add(requirement.name)
    edges = {
        CapabilityEdge(
            source=source,
            target=target,
            optional=optional,
        )
        for (source, target), optional in edge_optional.items()
    }

    sorted_edges = tuple(
        sorted(edges, key=lambda edge: (edge.source, edge.target)),
    )
    cycles = _find_cycles(nodes, adjacency)
    cyclic_nodes = {node for cycle in cycles for node in cycle}
    order = _topological_order(nodes, adjacency, cyclic_nodes)
    return CapabilityGraph(
        nodes=nodes,
        edges=sorted_edges,
        order=order,
        cycles=cycles,
    )


def _find_cycles(
    nodes: tuple[str, ...],
    adjacency: dict[str, set[str]],
) -> tuple[tuple[str, ...], ...]:
    """Find cycles via Tarjan's SCC algorithm."""

    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []
    counter = 0

    def strong_connect(node: str) -> None:
        nonlocal counter
        indices[node] = counter
        lowlinks[node] = counter
        counter += 1
        stack.append(node)
        on_stack.add(node)
        for neighbour in sorted(adjacency[node]):
            if neighbour not in indices:
                strong_connect(neighbour)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbour])
            elif neighbour in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbour])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            components.append(component)

    for node in nodes:
        if node not in indices:
            strong_connect(node)

    cycles: list[tuple[str, ...]] = []
    for component in components:
        members = set(component)
        if len(component) > 1:
            cycles.append(_cycle_path(members, adjacency))
        elif component[0] in adjacency[component[0]]:
            # Self dependency.
            cycles.append((component[0],))
    return tuple(sorted(cycles, key=lambda cycle: cycle))


def _cycle_path(
    members: set[str],
    adjacency: dict[str, set[str]],
) -> tuple[str, ...]:
    """Build a deterministic closed path through a cyclic component."""

    start = min(members)
    path = [start]
    visited = {start}

    def visit(node: str) -> bool:
        for neighbour in sorted(adjacency[node] & members):
            if neighbour == start:
                return True
            if neighbour in visited:
                continue
            visited.add(neighbour)
            path.append(neighbour)
            if visit(neighbour):
                return True
            path.pop()
            visited.discard(neighbour)
        return False

    visit(start)
    return tuple(path)


def _topological_order(
    nodes: tuple[str, ...],
    adjacency: dict[str, set[str]],
    cyclic_nodes: set[str],
) -> tuple[str, ...]:
    """Kahn's algorithm with a sorted ready set for deterministic order.

    Edges point from a capability to its dependency, so capabilities with no
    unresolved dependencies come first (setup order).
    """

    acyclic = {node for node in nodes if node not in cyclic_nodes}
    dependents: dict[str, list[str]] = {node: [] for node in acyclic}
    remaining_deps: dict[str, int] = {node: 0 for node in acyclic}
    for node in acyclic:
        for target in adjacency[node]:
            if target in acyclic:
                remaining_deps[node] += 1
                dependents[target].append(node)

    ready = [node for node in acyclic if remaining_deps[node] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for dependent in dependents[node]:
            remaining_deps[dependent] -= 1
            if remaining_deps[dependent] == 0:
                heapq.heappush(ready, dependent)
    return tuple(order)
