#!/usr/bin/env python3
"""Cheap documentation freshness checks for MiniSolver.

This script intentionally checks only mechanical invariants and obvious stale
phrases. Semantic documentation review remains a harness/skill responsibility.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


CONTRACT_ID_RE = re.compile(r"`([A-Z]+(?:/[A-Z]+)?-\d{3})`")

STALE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("unassigned contract wording", re.compile(r"\bnot assigned yet\b", re.IGNORECASE)),
    ("unfilled coverage matrix wording", re.compile(r"\bscaffolded but not filled\b", re.IGNORECASE)),
    ("phase-scaffold wording", re.compile(r"\bAdded in later phases\b", re.IGNORECASE)),
    ("old static soft-weight claim", re.compile(r"MiniModel currently stores one static", re.IGNORECASE)),
    ("old combined fraction-to-boundary claim", re.compile(r"currently uses one scalar fraction", re.IGNORECASE)),
    ("overstated embedded dependency claim", re.compile(r"No external libraries required", re.IGNORECASE)),
    ("old embedded section title", re.compile(r"^## .*Embedded Deployment\b", re.MULTILINE)),
    ("old embedded feature title", re.compile(r"^### .*Embedded Safety\b", re.MULTILINE)),
    ("current coverage wording in dated snapshot", re.compile(r"Current Coverage Overview", re.IGNORECASE)),
    ("old integrator enum RK2_EXPLICIT", re.compile(r"\bRK2_EXPLICIT\b")),
    ("old integrator enum RK2_IMPLICIT", re.compile(r"\bRK2_IMPLICIT\b")),
    ("old integrator enum RK4_EXPLICIT", re.compile(r"\bRK4_EXPLICIT\b")),
    ("old integrator enum RK4_IMPLICIT", re.compile(r"\bRK4_IMPLICIT\b")),
    ("old integrator enum MIDPOINT_EXPLICIT", re.compile(r"\bMIDPOINT_EXPLICIT\b")),
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def contract_id_errors(root: Path) -> list[str]:
    contract_dir = root / "docs" / "contracts"
    matrix_path = root / "docs" / "testing" / "contract-coverage-matrix.md"
    if not contract_dir.exists() or not matrix_path.exists():
        return []

    contract_ids: list[str] = []
    for path in contract_dir.glob("*.md"):
        if path.name == "_template.md":
            continue
        contract_ids.extend(CONTRACT_ID_RE.findall(path.read_text(encoding="utf-8")))

    matrix_text = matrix_path.read_text(encoding="utf-8")
    matrix_ids = CONTRACT_ID_RE.findall(matrix_text)
    rows = [
        line for line in matrix_text.splitlines()
        if line.startswith("| `") or line.startswith("| [`")
    ]

    missing = sorted(set(contract_ids) - set(matrix_ids))
    extra = sorted(set(matrix_ids) - set(contract_ids))
    p0_partial = sum(1 for line in rows if "`P0`" in line and "`partial`" in line)

    errors: list[str] = []
    if missing:
        errors.append(f"contract IDs missing from matrix: {missing}")
    if extra:
        errors.append(f"matrix IDs without contract definition: {extra}")
    if p0_partial:
        errors.append(f"P0 partial rows remain in contract matrix: {p0_partial}")
    return errors


def markdown_files(root: Path) -> list[Path]:
    files = [root / "README.md"]
    for base in ("docs",):
        base_path = root / base
        if not base_path.exists():
            continue
        for path in base_path.rglob("*.md"):
            relative = path.relative_to(root)
            if relative.parts[:2] in {("docs", "archive"), ("docs", "reviews")}:
                continue
            files.append(path)
    return sorted(set(files))


def link_errors(root: Path, files: list[Path]) -> list[str]:
    errors: list[str] = []
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in link_re.finditer(text):
            target = match.group(1).split("#", 1)[0].strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("/"):
                continue
            if not (path.parent / target).resolve().exists():
                errors.append(f"{path.relative_to(root)} -> {match.group(1)}")
    return errors


def stale_phrase_errors(root: Path, files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in STALE_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                errors.append(f"{path.relative_to(root)}:{line_no}: {label}: {match.group(0)!r}")
    return errors


def main() -> int:
    root = repository_root()
    files = markdown_files(root)
    errors: list[str] = []
    errors.extend(contract_id_errors(root))
    errors.extend(f"broken markdown link: {error}" for error in link_errors(root, files))
    errors.extend(f"stale documentation phrase: {error}" for error in stale_phrase_errors(root, files))

    if errors:
        print("MiniSolver docs freshness check failed:\n", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print(
            "\nIf a phrase is intentionally historical, move it under docs/reviews "
            "or docs/archive, or narrow the freshness rule.",
            file=sys.stderr,
        )
        return 1

    print("MiniSolver docs freshness check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
