"""Sandboxed repository tools.

Mutating operations (write_file, run_command, run_tests) run ONLY inside a
git worktree created per call under a temp directory. The original repo path
is never written to. This Sandbox is the shared isolation layer every later
agent will reuse.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".next",
    "dist",
    "build",
}


class SandboxError(RuntimeError):
    """Raised when a sandbox operation is invalid or fails."""


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def output(self) -> str:
        return (self.stdout or "") + (self.stderr or "")


@dataclass
class TestResult:
    passed: int
    failed: int
    exit_code: int
    output: str
    command: list[str] = field(default_factory=list)


class Sandbox:
    """Per-run git worktree under a temp directory.

    Usage:
        with Sandbox.create("/path/to/repo") as sb:
            write_file(sb, "foo.py", "...")
            diff = git_diff(sb)
            tests = run_tests(sb)
    """

    def __init__(self, original_repo: Path, root: Path, worktree_path: Path, used_worktree: bool):
        self.original_repo = original_repo
        self.root = root
        self.worktree_path = worktree_path
        self._used_worktree = used_worktree
        self._closed = False

    @classmethod
    def create(cls, repo_path: str | Path) -> "Sandbox":
        original = Path(repo_path).expanduser().resolve()
        if not original.exists() or not original.is_dir():
            raise SandboxError(f"repo_path does not exist or is not a directory: {original}")

        root = Path(tempfile.mkdtemp(prefix="specdet-sandbox-"))
        worktree_path = root / "wt"
        if worktree_path.resolve() == original.resolve():
            raise SandboxError("refusing to sandbox onto the original repo path")

        if _is_git_repo(original):
            _run_git(
                original,
                ["worktree", "add", "--detach", str(worktree_path), "HEAD"],
                check=True,
            )
            return cls(original, root, worktree_path, used_worktree=True)

        # Non-git sources: copy into the temp dir and snapshot so git_diff works.
        # Still never mutates original_repo.
        shutil.copytree(
            original,
            worktree_path,
            ignore=shutil.ignore_patterns(*IGNORE_DIRS),
            dirs_exist_ok=False,
        )
        _run_git(worktree_path, ["init"], check=True)
        _run_git(worktree_path, ["add", "-A"], check=True)
        _run_git(
            worktree_path,
            [
                "-c",
                "user.email=sandbox@specdetective.local",
                "-c",
                "user.name=Sandbox",
                "commit",
                "--quiet",
                "-m",
                "sandbox snapshot",
            ],
            check=True,
        )
        return cls(original, root, worktree_path, used_worktree=False)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._used_worktree and self.worktree_path.exists():
            try:
                _run_git(
                    self.original_repo,
                    ["worktree", "remove", "--force", str(self.worktree_path)],
                    check=False,
                )
            except Exception:
                pass
            try:
                _run_git(self.original_repo, ["worktree", "prune"], check=False)
            except Exception:
                pass
        shutil.rmtree(self.root, ignore_errors=True)

    def resolve(self, relative_path: str | Path) -> Path:
        """Resolve a path inside the worktree; reject escapes."""
        rel = Path(relative_path)
        if rel.is_absolute():
            candidate = rel.resolve()
        else:
            candidate = (self.worktree_path / rel).resolve()
        worktree = self.worktree_path.resolve()
        if candidate != worktree and not _is_relative_to(candidate, worktree):
            raise SandboxError(f"path escapes sandbox: {relative_path}")
        return candidate


def list_files(
    sandbox: Sandbox,
    path: str = ".",
    max_depth: int | None = None,
) -> list[str]:
    """List files under `path` (relative to the sandbox worktree)."""
    start = sandbox.resolve(path)
    if not start.exists():
        return []
    if start.is_file():
        return [_rel(sandbox, start)]

    results: list[str] = []
    start_depth = len(start.parts)
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        current = Path(dirpath)
        depth = len(current.parts) - start_depth
        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []
        for name in filenames:
            if name.startswith(".") and name not in {"gitignore"}:
                continue
            results.append(_rel(sandbox, current / name))
    results.sort()
    return results


def read_file(sandbox: Sandbox, path: str) -> str:
    target = sandbox.resolve(path)
    if not target.is_file():
        raise SandboxError(f"not a file: {path}")
    return target.read_text(encoding="utf-8", errors="replace")


def write_file(sandbox: Sandbox, path: str, content: str) -> None:
    """Write a file inside the sandbox worktree only."""
    target = sandbox.resolve(path)
    if not _is_relative_to(target, sandbox.worktree_path.resolve()) and target != sandbox.worktree_path.resolve():
        raise SandboxError(f"write_file refused path outside sandbox: {path}")
    original = sandbox.original_repo.resolve()
    worktree = sandbox.worktree_path.resolve()
    if _is_relative_to(target, original) and not _is_relative_to(target, worktree):
        raise SandboxError("write_file refused to mutate the original repo")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def run_command(
    sandbox: Sandbox,
    command: str | Sequence[str],
    timeout: int = 60,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Run a command with cwd locked to the sandbox worktree."""
    argv = _normalize_command(command)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            argv,
            cwd=sandbox.worktree_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged_env,
            check=False,
        )
        return CommandResult(
            command=argv,
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(
            command=argv,
            exit_code=124,
            stdout=stdout,
            stderr=(stderr + f"\n[timeout after {timeout}s]").strip(),
        )


def run_tests(sandbox: Sandbox, timeout: int = 120) -> TestResult:
    """Detect and run the repo's existing test suite inside the sandbox."""
    command = _detect_test_command(sandbox.worktree_path)
    result = run_command(sandbox, command, timeout=timeout)
    passed, failed = _parse_test_counts(result.output)
    return TestResult(
        passed=passed,
        failed=failed,
        exit_code=result.exit_code,
        output=result.output,
        command=command,
    )


def search_code(
    sandbox: Sandbox,
    pattern: str,
    path: str = ".",
    max_results: int = 40,
) -> list[dict[str, str | int]]:
    """Ripgrep search inside the sandbox worktree."""
    result = run_command(
        sandbox,
        ["rg", "--json", "-n", "--max-count", str(max_results), pattern, path],
        timeout=60,
    )
    hits: list[dict[str, str | int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue
        data = obj.get("data") or {}
        hits.append(
            {
                "path": (data.get("path") or {}).get("text", ""),
                "line": int(data.get("line_number") or 0),
                "text": (data.get("lines") or {}).get("text", "").strip(),
            }
        )
    return hits


def git_history(sandbox: Sandbox, path: str = ".", max_commits: int = 10) -> str:
    """Recent commit history for `path` inside the sandbox."""
    result = run_command(
        sandbox,
        ["git", "log", f"-{max_commits}", "--oneline", "--", path],
        timeout=30,
    )
    return result.stdout.strip()


def inspect_dependencies(sandbox: Sandbox) -> dict[str, str]:
    """Read dependency manifests present in the repo."""
    deps: dict[str, str] = {}
    for name in ("requirements.txt", "pyproject.toml", "package.json", "Pipfile"):
        target = sandbox.worktree_path / name
        if target.is_file():
            deps[name] = target.read_text(encoding="utf-8", errors="replace")[:4000]
    return deps


def git_diff(sandbox: Sandbox) -> str:
    """Return the unified diff of sandbox changes against HEAD."""
    staged = run_command(sandbox, ["git", "add", "-A"])
    if not staged.ok:
        raise SandboxError(f"git add failed: {staged.output}")
    diff = run_command(sandbox, ["git", "diff", "--cached", "--no-color"])
    if not diff.ok:
        raise SandboxError(f"git diff failed: {diff.output}")
    return diff.stdout


def delete_file(sandbox: Sandbox, path: str) -> None:
    """Remove a file inside the sandbox worktree only."""
    target = sandbox.resolve(path)
    if not target.exists():
        return
    if not _is_relative_to(target.resolve(), sandbox.worktree_path.resolve()):
        raise SandboxError(f"delete_file refused path outside sandbox: {path}")
    if target.is_dir():
        raise SandboxError(f"delete_file refused directory: {path}")
    target.unlink()


def apply_diff(sandbox: Sandbox, diff_text: str) -> CommandResult:
    """Apply a unified diff inside the sandbox worktree."""
    cleaned = _normalize_diff(extract_unified_diff(diff_text))
    if not cleaned.strip():
        raise SandboxError("empty diff; nothing to apply")

    chunks = [_repair_hunk_headers(c) for c in _split_diff_files(cleaned) if _has_hunk(c)]
    if not chunks:
        raise SandboxError("diff contained no complete file hunks (possibly truncated)")

    errors: list[str] = []
    applied = 0
    for chunk in chunks:
        try:
            _apply_single_file_diff(sandbox, chunk)
            applied += 1
        except SandboxError as exc:
            errors.append(str(exc))

    if applied == 0:
        raise SandboxError(
            "failed to apply diff.\n" + "\n".join(errors[:3]) + f"\n\ndiff:\n{chunks[0][:2000]}"
        )

    note = ""
    skipped = len(_split_diff_files(cleaned)) - len(chunks)
    if skipped:
        note = f" (skipped {skipped} incomplete file section(s))"
    if errors:
        note += f" ({applied} applied, {len(errors)} failed)"

    return CommandResult(
        command=["apply_diff"],
        exit_code=0,
        stdout=f"applied {applied} file patch(es){note}",
        stderr="\n".join(errors) if errors else "",
    )


def _apply_single_file_diff(sandbox: Sandbox, chunk: str) -> None:
    patch_path = sandbox.root / "incoming.patch"
    patch_path.write_text(chunk, encoding="utf-8")

    for argv in (
        ["git", "apply", "--whitespace=nowarn", str(patch_path)],
        ["git", "apply", "-p1", "--whitespace=nowarn", str(patch_path)],
        ["git", "apply", "--3way", "--whitespace=nowarn", str(patch_path)],
    ):
        result = run_command(sandbox, argv)
        if result.ok:
            return

    _apply_unified_diff_python(sandbox, chunk)


def _split_diff_files(text: str) -> list[str]:
    """Split a multi-file unified diff into per-file chunks."""
    lines = text.splitlines(keepends=True)
    if not any(line.startswith("diff --git ") for line in lines):
        return [text] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("diff --git ") and current:
            chunks.append("".join(current))
            current = [line]
        elif line.startswith("diff --git "):
            current = [line]
        elif current:
            current.append(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _has_hunk(chunk: str) -> bool:
    return bool(re.search(r"^@@ ", chunk, flags=re.MULTILINE))


def _repair_hunk_headers(chunk: str) -> str:
    """Fix common LLM hunk header mistakes, e.g. `@@ -1 +0,0 @@` → `@@ -1,1 +0,0 @@`."""
    out: list[str] = []
    for line in chunk.splitlines():
        m = re.match(r"^@@ -(\d+) \+(\d+),(\d+) @@(.*)$", line)
        if m:
            out.append(f"@@ -{m.group(1)},1 +{m.group(2)},{m.group(3)} @@{m.group(4)}")
            continue
        m = re.match(r"^@@ -(\d+),(\d+) \+(\d+) @@(.*)$", line)
        if m:
            out.append(f"@@ -{m.group(1)},{m.group(2)} +{m.group(3)},0 @@{m.group(4)}")
            continue
        out.append(line)
    body = "\n".join(out)
    return body + ("\n" if chunk.endswith("\n") else "")


def _normalize_diff(text: str) -> str:
    """Ensure git-style headers so `git apply` accepts LLM unified diffs."""
    body = text.strip() + "\n"
    if "diff --git" in body:
        return body
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            old_raw = line[4:].strip().split("\t")[0]
            new_raw = old_raw
            if i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                new_raw = lines[i + 1][4:].strip().split("\t")[0]
            old_path = _strip_diff_path(old_raw)
            new_path = _strip_diff_path(new_raw) or old_path
            git_path = new_path or old_path
            out.append(f"diff --git a/{git_path} b/{git_path}")
            if old_raw in {"/dev/null", "a/dev/null"}:
                out.append("new file mode 100644")
            elif new_raw in {"/dev/null", "b/dev/null"}:
                out.append("deleted file mode 100644")
            out.append(f"--- a/{old_path}" if old_path else "--- /dev/null")
            out.append(f"+++ b/{new_path}" if new_path else "+++ /dev/null")
            i += 2 if i + 1 < len(lines) and lines[i + 1].startswith("+++ ") else 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def _strip_diff_path(raw: str) -> str:
    path = raw.strip()
    if path in {"/dev/null", "a/dev/null", "b/dev/null"}:
        return ""
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path.split("\t")[0]


def _apply_unified_diff_python(sandbox: Sandbox, diff_text: str) -> None:
    """Apply unified-diff hunks without shelling out to `patch`."""
    files: dict[str, str] = {}
    deletes: set[str] = set()
    current_path: str | None = None
    delete_current = False
    hunk_old: list[str] = []
    hunk_new: list[str] = []
    in_hunk = False

    def load(path: str) -> str:
        if path not in files:
            try:
                files[path] = read_file(sandbox, path)
            except SandboxError:
                files[path] = ""
        return files[path]

    def flush_hunk() -> None:
        nonlocal hunk_old, hunk_new, in_hunk
        if not in_hunk or not current_path:
            hunk_old, hunk_new, in_hunk = [], [], False
            return
        if delete_current:
            deletes.add(current_path)
        else:
            files[current_path] = _splice_hunk(
                load(current_path), "".join(hunk_old), "".join(hunk_new), current_path
            )
        hunk_old, hunk_new, in_hunk = [], [], False

    for line in diff_text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith("diff --git "):
            flush_hunk()
            delete_current = False
            continue
        if stripped.startswith("+++ "):
            flush_hunk()
            new_raw = stripped[4:].strip().split("\t")[0]
            path = _strip_diff_path(new_raw)
            delete_current = new_raw in {"/dev/null", "b/dev/null", "dev/null"}
            current_path = path or current_path
            continue
        if stripped.startswith("--- ") or stripped.startswith("index ") or stripped.startswith("new file"):
            continue
        if stripped.startswith("deleted file"):
            delete_current = True
            continue
        if stripped.startswith("@@"):
            flush_hunk()
            in_hunk = True
            continue
        if not in_hunk:
            continue
        nl = "\n" if line.endswith("\n") else ""
        if stripped.startswith("+"):
            hunk_new.append(stripped[1:] + nl)
        elif stripped.startswith("-"):
            hunk_old.append(stripped[1:] + nl)
        elif stripped.startswith("\\"):
            continue
        else:
            body = stripped[1:] if stripped.startswith(" ") else stripped
            hunk_old.append(body + nl)
            hunk_new.append(body + nl)

    flush_hunk()
    if not files and not deletes:
        raise SandboxError("diff contained no file paths")
    for path, content in files.items():
        if path not in deletes:
            write_file(sandbox, path, content)
    for path in deletes:
        delete_file(sandbox, path)


def _splice_hunk(text: str, old: str, new: str, path: str) -> str:
    """Replace `old` with `new` inside `text`, tolerating trailing-blank mismatches."""
    if old and old in text:
        return text.replace(old, new, 1)
    if text.rstrip("\n") == old.rstrip("\n"):
        ending = "\n" if (text.endswith("\n") or new.endswith("\n")) else ""
        return new.rstrip("\n") + ending
    # Trailing extra blank line in the hunk context (common LLM mismatch).
    if old.rstrip("\n") and old.rstrip("\n") + "\n" in (text if text.endswith("\n") else text + "\n"):
        matched = old.rstrip("\n") + "\n"
        haystack = text if text.endswith("\n") else text + "\n"
        return haystack.replace(matched, new if new.endswith("\n") else new + "\n", 1)
    if not old:
        return text + new
    raise SandboxError(f"hunk context not found in {path}")


def extract_unified_diff(text: str) -> str:
    """Pull a unified diff out of model output (markdown fences allowed)."""
    if not text:
        return ""
    fenced = re.findall(r"```(?:diff|patch|udiff)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced + [text]
    for candidate in candidates:
        body = candidate.strip()
        if "diff --git" in body or body.startswith("--- ") or body.startswith("*** Begin Patch"):
            start = body.find("diff --git")
            if start == -1:
                start = body.find("--- ")
            if start == -1:
                start = 0
            return body[start:].strip() + "\n"
    return text.strip() + "\n"


def _detect_test_command(root: Path) -> list[str]:
    package_json = root / "package.json"
    if package_json.is_file():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if "test" in (data.get("scripts") or {}):
            return ["npm", "test", "--", "--watchAll=false"]

    python = sys.executable
    has_pytest = (
        (root / "pytest.ini").exists()
        or (root / "conftest.py").exists()
        or (root / "tests").is_dir()
        or any(root.glob("test_*.py"))
        or any(root.glob("**/test_*.py"))
    )
    pyproject = root / "pyproject.toml"
    if pyproject.exists() and "pytest" in pyproject.read_text(encoding="utf-8", errors="replace"):
        has_pytest = True
    if has_pytest:
        return [python, "-m", "pytest", "-q", "--tb=short"]
    return [python, "-m", "pytest", "-q", "--tb=short"]


def _parse_test_counts(output: str) -> tuple[int, int]:
    passed = failed = 0
    if m := re.search(r"(\d+)\s+passed", output):
        passed = int(m.group(1))
    if m := re.search(r"(\d+)\s+failed", output):
        failed = int(m.group(1))
    if m := re.search(r"(\d+)\s+error", output):
        failed += int(m.group(1))
    if m := re.search(r"(\d+)\s+errors", output):
        failed += int(m.group(1))
    return passed, failed


def _normalize_command(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        import shlex

        argv = shlex.split(command)
    else:
        argv = [str(part) for part in command]
    if not argv:
        raise SandboxError("empty command")
    return argv


def _is_git_repo(path: Path) -> bool:
    """True only if `path` itself is a git root — not merely inside a parent repo."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        return Path(result.stdout.strip()).resolve() == path.resolve()
    except OSError:
        return False


def _run_git(cwd: Path, args: list[str], check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


def _rel(sandbox: Sandbox, path: Path) -> str:
    return path.resolve().relative_to(sandbox.worktree_path.resolve()).as_posix()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
