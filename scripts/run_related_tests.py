"""Run pytest scoped to the test package(s) touched by a set of changed files.

Used as a pre-commit local hook: `tests/` mirrors `src/` at the package
(directory) level, not file-for-file, so a changed `src/<pkg>/...` file maps
to the whole `tests/<pkg>/` directory rather than a single test file.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dir_target(parts: tuple[str, ...]) -> str:
    while parts:
        candidate = REPO_ROOT / "tests" / Path(*parts)
        if candidate.is_dir():
            return str(candidate.relative_to(REPO_ROOT))
        parts = parts[:-1]
    return "tests"


def _test_target(changed: str) -> str | None:
    path = Path(changed)
    if path.parts[:1] == ("tests",):
        if path.name.startswith("test_") and (REPO_ROOT / path).is_file():
            return changed
        return _dir_target(path.parent.parts[1:])
    if path.parts[:1] != ("src",):
        return None
    return _dir_target(path.parent.parts[1:])


def main(argv: list[str]) -> int:
    targets = {target for changed in argv if (target := _test_target(changed))}

    if not targets:
        print("run_related_tests: no Python source/test changes, skipping pytest")
        return 0

    print(f"run_related_tests: running pytest for {sorted(targets)}")
    return subprocess.call(["uv", "run", "pytest", *sorted(targets)], cwd=REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
