"""Sandbox isolation checks — mutating ops must not touch the original repo."""

from pathlib import Path

from app.tools.repo_tools import Sandbox, apply_diff, git_diff, run_command, write_file


def test_write_file_does_not_mutate_original(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.txt").write_text("hello\n", encoding="utf-8")
    original = (repo / "hello.txt").read_text(encoding="utf-8")

    with Sandbox.create(repo) as sb:
        write_file(sb, "hello.txt", "changed\n")
        write_file(sb, "new.txt", "brand new\n")
        diff = git_diff(sb)
        assert "changed" in diff
        assert (sb.worktree_path / "hello.txt").read_text(encoding="utf-8") == "changed\n"
        assert (repo / "hello.txt").read_text(encoding="utf-8") == original
        assert not (repo / "new.txt").exists()
        assert sb.worktree_path.resolve() != repo.resolve()


def test_run_command_cwd_is_worktree(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("a\n", encoding="utf-8")

    with Sandbox.create(repo) as sb:
        result = run_command(sb, ["pwd"])
        assert result.ok
        assert Path(result.stdout.strip()).resolve() == sb.worktree_path.resolve()


def test_apply_diff_with_a_b_prefixes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "greet.py").write_text(
        'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n',
        encoding="utf-8",
    )

    diff = """--- a/greet.py
+++ b/greet.py
@@ -1,2 +1,6 @@
 def greet(name: str) -> str:
     return f"Hello, {name}!"
+
+
+def farewell(name: str) -> str:
+    return f"Goodbye, {name}!"
"""
    with Sandbox.create(repo) as sb:
        apply_diff(sb, diff)
        updated = (sb.worktree_path / "greet.py").read_text(encoding="utf-8")
        assert "def farewell" in updated
        assert (repo / "greet.py").read_text(encoding="utf-8").count("farewell") == 0
        assert "farewell" in git_diff(sb)


def test_apply_diff_tolerates_extra_blank_line_in_hunk(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "greet.py").write_text(
        'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n',
        encoding="utf-8",
    )
    diff = """--- a/greet.py
+++ b/greet.py
@@ -1,3 +1,7 @@
 def greet(name: str) -> str:
     return f"Hello, {name}!"
 
+
+def farewell(name: str) -> str:
+    return f"Goodbye, {name}!"
+
"""
    with Sandbox.create(repo) as sb:
        apply_diff(sb, diff)
        updated = (sb.worktree_path / "greet.py").read_text(encoding="utf-8")
        assert "def farewell" in updated
        assert 'return f"Goodbye, {name}!"' in updated
