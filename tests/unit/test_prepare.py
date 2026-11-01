from contract_intelligence.data.prepare import assign_split, mini


def test_mini_is_deterministic_and_exact() -> None:
    tests = [
        {"query": str(index), "snippets": [{"file_path": f"file-{index}"}]}
        for index in range(300)
    ]
    assert mini(tests) == mini(tests)
    assert len(mini(tests)) == 194


def test_assign_split_is_stable() -> None:
    assert assign_split("example query") == assign_split(" example query ")
