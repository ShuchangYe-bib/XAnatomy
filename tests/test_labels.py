from xanatomy.labels import group_ids, list_groups, resolve


def test_group_lookup_contains_expected_groups():
    groups = list_groups()
    assert "lungs" in groups
    assert "bones" in groups


def test_resolve_exact_and_group_queries():
    lungs = resolve("lungs")
    assert len(lungs) >= 2
    heart = resolve("heart")
    assert len(heart) >= 1


def test_group_ids_is_stable():
    assert tuple(resolve("lungs")) == group_ids("lungs")
