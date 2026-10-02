from app.matching import hungarian_min, max_cardinality


def test_max_cardinality_simple():
    adj = [[0, 1], [1, 2], [2]]
    assert max_cardinality(adj, 3) == 3


def test_max_cardinality_halting_choice():
    # Greedy would take t0 for arm0 then stall; augmenting paths solve it.
    adj = [[0], [0, 1], [0, 1, 2]]
    assert max_cardinality(adj, 3) == 3


def test_hungarian_identity():
    cost = [[1, None, None], [None, 2, None], [None, None, 4]]
    total, assign = hungarian_min(cost)
    assert total == 7
    assert assign == [0, 1, 2]


def test_hungarian_forced_columns():
    # Row 1 can only use column 1 (cost 1); row 0 then takes column 0.
    cost = [[10, 2], [None, 1]]
    total, assign = hungarian_min(cost)
    assert total == 11
    assert assign == [0, 1]


def test_hungarian_rectangular():
    cost = [[5, 1, 9], [None, 8, 2]]
    total, assign = hungarian_min(cost)
    assert total == 3
    assert assign == [1, 2]
