"""Tests for experiment source provenance."""

from __future__ import annotations

from thesis.eval.manifest import execution_source_digest, source_tree_digest


def test_execution_digest_ignores_tests_and_prose(tmp_path) -> None:
    runtime = tmp_path / "thesis" / "module.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("VALUE = 1\n", encoding="utf-8")
    test_file = tmp_path / "tests" / "test_module.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_value(): pass\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("first\n", encoding="utf-8")

    execution_before = execution_source_digest(tmp_path)
    full_before = source_tree_digest(tmp_path)
    test_file.write_text("def test_value(): assert True\n", encoding="utf-8")
    readme.write_text("second\n", encoding="utf-8")

    assert execution_source_digest(tmp_path) == execution_before
    assert source_tree_digest(tmp_path) != full_before


def test_execution_digest_changes_with_runtime_code(tmp_path) -> None:
    runtime = tmp_path / "thesis" / "module.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("VALUE = 1\n", encoding="utf-8")
    before = execution_source_digest(tmp_path)

    runtime.write_text("VALUE = 2\n", encoding="utf-8")

    assert execution_source_digest(tmp_path) != before
