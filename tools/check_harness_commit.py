#!/usr/bin/env python3
"""Validate MiniSolver harness metadata in commit messages.

This hook is intentionally lightweight. It cannot prove that the harness was
actually followed, but it forces each commit to declare the harness size,
scope, evidence, and documentation freshness decision before the commit lands.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


HARNESS_VALUES = {
    "maintainer",
    "reduced",
    "core5",
    "core5+specialist",
    "docs-release",
}

SCOPE_VALUES = {
    "docs-only",
    "test-only",
    "repo-hygiene",
    "solver-core",
    "codegen",
    "benchmark",
    "mixed",
}

DOCS_VALUES = {
    "updated",
    "not-needed",
    "deferred",
}

DOCS_REQUIRED_SCOPES = {
    "solver-core",
    "codegen",
    "benchmark",
    "mixed",
}


def run_git(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def staged_paths() -> list[str]:
    return run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMRD"])


def read_trailers(message_path: Path) -> dict[str, str]:
    trailers: dict[str, str] = {}
    for raw_line in message_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9-]*):\s*(.+)$", line)
        if match:
            trailers[match.group(1).lower()] = match.group(2).strip()
    return trailers


def infer_scope(paths: list[str]) -> str:
    if not paths:
        return "repo-hygiene"

    def is_doc(path: str) -> bool:
        return (
            path.startswith("docs/")
            or path in {"README.md", "ROADMAP.md", "CHANGELOG.md", "LICENSE-3RD-PARTY.md"}
            or path.endswith(".md")
        )

    def is_test(path: str) -> bool:
        return (
            path.startswith("tests/")
            or path.startswith("examples/")
            or path == "CMakeLists.txt"
            or path.startswith(".github/")
        )

    if all(is_doc(path) for path in paths):
        return "docs-only"

    repo_hygiene_paths = {
        ".clang-format",
        ".gitignore",
        ".pre-commit-config.yaml",
        "tools/check_docs_freshness.py",
        "tools/check_harness_commit.py",
        "tools/install_harness_commit_hook.sh",
    }
    if all(is_doc(path) or path in repo_hygiene_paths for path in paths):
        return "repo-hygiene"

    if all(is_doc(path) or is_test(path) for path in paths):
        return "test-only"

    solver_prefixes = (
        "include/minisolver/solver/",
        "include/minisolver/algorithms/",
        "include/minisolver/core/",
        "include/minisolver/integrator/",
        "src/",
    )
    if any(path.startswith(solver_prefixes) for path in paths):
        return "solver-core"
    if any(path.startswith("python/minisolver/") for path in paths):
        return "codegen"
    if any(path.startswith("tools/") or path.startswith("bench") for path in paths):
        return "benchmark"
    return "mixed"


def is_exempt_message(message: str) -> bool:
    first = next((line.strip() for line in message.splitlines() if line.strip()), "")
    return first.startswith(("Merge ", "Revert ", "fixup!", "squash!"))


def validate(message_path: Path) -> int:
    message = message_path.read_text(encoding="utf-8")
    if is_exempt_message(message):
        return 0

    trailers = read_trailers(message_path)
    paths = staged_paths()
    inferred_scope = infer_scope(paths)
    message_lower = message.lower()

    errors: list[str] = []
    harness = trailers.get("harness", "")
    scope = trailers.get("scope", "")
    evidence = trailers.get("evidence", "")
    docs = trailers.get("docs", "")

    if harness not in HARNESS_VALUES:
        errors.append(
            "missing or invalid 'Harness:' trailer "
            f"(expected one of: {', '.join(sorted(HARNESS_VALUES))})"
        )
    if scope not in SCOPE_VALUES:
        errors.append(
            "missing or invalid 'Scope:' trailer "
            f"(expected one of: {', '.join(sorted(SCOPE_VALUES))})"
        )
    if not evidence:
        errors.append("missing 'Evidence:' trailer")

    if docs and docs not in DOCS_VALUES:
        errors.append(
            "invalid 'Docs:' trailer "
            f"(expected one of: {', '.join(sorted(DOCS_VALUES))})"
        )

    docs_required = scope in DOCS_REQUIRED_SCOPES or inferred_scope in DOCS_REQUIRED_SCOPES
    if docs_required and not docs:
        errors.append(
            "missing 'Docs:' trailer for behavior/codegen/benchmark/mixed scope "
            "(use updated, not-needed, or deferred)"
        )

    if docs == "not-needed" and "docs rationale:" not in message_lower:
        errors.append("'Docs: not-needed' requires a 'Docs rationale:' line")
    if docs == "deferred" and "deferred path:" not in message_lower:
        errors.append("'Docs: deferred' requires a 'Deferred path:' line")

    if scope and scope != inferred_scope and inferred_scope in {"solver-core", "codegen"}:
        errors.append(
            f"declared Scope '{scope}' does not match staged high-risk scope '{inferred_scope}'"
        )

    if inferred_scope == "docs-only" and harness not in {"maintainer", "reduced", "core5", "core5+specialist", "docs-release"}:
        errors.append("docs-only commits may use Harness: maintainer, reduced, core5, core5+specialist, or docs-release")
    if inferred_scope == "test-only" and harness not in {"reduced", "core5", "core5+specialist"}:
        errors.append("test-only commits require Harness: reduced, core5, or core5+specialist")
    if inferred_scope in {"solver-core", "codegen"} and harness not in {"core5", "core5+specialist"}:
        errors.append(f"{inferred_scope} commits require Harness: core5 or core5+specialist")
    if inferred_scope == "benchmark" and harness not in {"reduced", "core5", "core5+specialist"}:
        errors.append("benchmark/tool commits require Harness: reduced, core5, or core5+specialist")

    if errors:
        print("MiniSolver harness commit check failed:\n", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print("\nAdd trailers like:", file=sys.stderr)
        print("  Harness: reduced", file=sys.stderr)
        print(f"  Scope: {inferred_scope}", file=sys.stderr)
        print("  Evidence: ctest --test-dir build --output-on-failure", file=sys.stderr)
        if inferred_scope in DOCS_REQUIRED_SCOPES:
            print("  Docs: updated", file=sys.stderr)
        print("\nStaged files:", file=sys.stderr)
        for path in paths[:20]:
            print(f"  {path}", file=sys.stderr)
        if len(paths) > 20:
            print(f"  ... {len(paths) - 20} more", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_harness_commit.py COMMIT_MSG_FILE", file=sys.stderr)
        return 2
    return validate(Path(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
