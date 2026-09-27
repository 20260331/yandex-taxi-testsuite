from testsuite._internal.capabilities import (
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilitySpec,
    build_graph,
)


def build(specs):
    registry = CapabilityRegistry()
    for spec in specs:
        registry.register(spec)
    assert registry.errors == []
    return build_graph(registry.specs)


def base_spec(name, requires=()):
    return {
        'name': name,
        'provider': f'p.{name}',
        'version': '1.0',
        'requires': requires,
    }


def test_empty_graph():
    graph = build([])
    assert graph.nodes == ()
    assert graph.edges == ()
    assert graph.order == ()
    assert graph.cycles == ()


def test_nodes_and_edges_sorted_deterministically():
    graph = build(
        [
            base_spec('beta', [('alpha',)]),
            base_spec('gamma', [('alpha',)]),
            base_spec('alpha'),
        ],
    )
    assert graph.nodes == ('alpha', 'beta', 'gamma')
    assert [(edge.source, edge.target) for edge in graph.edges] == [
        ('beta', 'alpha'),
        ('gamma', 'alpha'),
    ]


def test_setup_order_dependencies_first():
    graph = build(
        [
            base_spec('app', [('service',), ('db',)]),
            base_spec('service', [('db',)]),
            base_spec('db'),
        ],
    )
    order = graph.order
    assert order.index('db') < order.index('service')
    assert order.index('service') < order.index('app')


def test_diamond_graph_order():
    graph = build(
        [
            base_spec('top', [('left',), ('right',)]),
            base_spec('left', [('base',)]),
            base_spec('right', [('base',)]),
            base_spec('base'),
        ],
    )
    order = graph.order
    assert order[0] == 'base'
    assert order[-1] == 'top'
    assert set(order[1:3]) == {'left', 'right'}


def test_order_independent_of_registration_order():
    specs = [
        base_spec('app', [('service',)]),
        base_spec('service', [('db',)]),
        base_spec('db'),
    ]
    first = build(specs)
    second = build(list(reversed(specs)))
    assert first.order == second.order == ('db', 'service', 'app')


def test_missing_dependency_produces_no_edge():
    graph = build([base_spec('alpha', [('missing',)])])
    assert graph.nodes == ('alpha',)
    assert graph.edges == ()
    assert graph.order == ('alpha',)


def test_edges_to_same_target_deduped():
    # The registry rejects duplicate requirement names; check graph level
    # dedupe with specs built directly.
    graph = build_graph(
        [
            CapabilitySpec(
                name='alpha',
                provider='p.alpha',
                requires=(
                    CapabilityRequirement('base'),
                    CapabilityRequirement('base', optional=True),
                ),
            ),
            CapabilitySpec(name='base', provider='p.base'),
        ],
    )
    assert len(graph.edges) == 1


def test_self_dependency_is_cycle():
    graph = build([base_spec('alpha', [('alpha',)])])
    assert graph.cycles == (('alpha',),)
    assert graph.order == ()


def test_three_node_cycle_detected():
    graph = build(
        [
            base_spec('alpha', [('gamma',)]),
            base_spec('beta', [('alpha',)]),
            base_spec('gamma', [('beta',)]),
        ],
    )
    assert len(graph.cycles) == 1
    cycle = graph.cycles[0]
    assert set(cycle) == {'alpha', 'beta', 'gamma'}
    # Closed path: every consecutive pair is a real dependency edge.
    requires = {
        'alpha': {'gamma'},
        'beta': {'alpha'},
        'gamma': {'beta'},
    }
    path = [*cycle, cycle[0]]
    for source, target in zip(path, path[1:]):
        assert target in requires[source]


def test_acyclic_nodes_still_ordered_with_cycle_present():
    graph = build(
        [
            base_spec('alpha', [('beta',)]),
            base_spec('beta', [('alpha',)]),
            base_spec('standalone'),
        ],
    )
    assert len(graph.cycles) == 1
    assert graph.order == ('standalone',)


def test_graph_is_json_friendly():
    graph = build([base_spec('alpha'), base_spec('beta', [('alpha',)])])
    edge = graph.edges[0]
    assert (edge.source, edge.target, edge.optional) == (
        'beta',
        'alpha',
        False,
    )
