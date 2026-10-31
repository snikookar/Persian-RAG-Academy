"""Check that the built sdist and wheel contain only what users need.

    python -m build
    python scripts/check_dist.py            # checks every file in dist/
    python scripts/check_dist.py dist/parsrag-0.1.0.tar.gz

Allowed in the sdist: ``src/parsrag/**``, ``README.md``, ``LICENSE``,
``pyproject.toml``, ``CHANGELOG.md``, plus the two files the build backend always
adds (the generated ``PKG-INFO`` and ``.gitignore``, which hatchling force-includes
in every sdist with no option to turn it off).
Allowed in the wheel: ``parsrag/**`` and its ``.dist-info``. Lessons,
fixtures, tests, docs, examples, the grader and other repository files must
never ship; neither must caches or compiled files. The script also checks
that a few files users do need are present and that the PyPI readme has no
relative links left (they would be broken on pypi.org).
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SDIST_FILES = {"PKG-INFO", ".gitignore", "README.md", "LICENSE", "pyproject.toml", "CHANGELOG.md"}
SDIST_DIRS = ("src/parsrag/",)
REQUIRED_SDIST = {"PKG-INFO", "README.md", "LICENSE", "pyproject.toml", "CHANGELOG.md"}
REQUIRED_WHEEL = {"parsrag/__init__.py", "parsrag/py.typed", "parsrag/api/static/index.html"}
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#|mailto:)[^)\s]+\)")


def bad_member(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return any(p in FORBIDDEN_PARTS for p in parts) or path.endswith((".pyc", ".pyo"))


def check_sdist(path: Path) -> list[str]:
    errors: list[str] = []
    with tarfile.open(path) as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
        pkg_info = next((m for m in tar.getmembers() if m.name.endswith("/PKG-INFO")), None)
        metadata = tar.extractfile(pkg_info).read().decode() if pkg_info else ""  # type: ignore[union-attr]
    prefix = path.name.removesuffix(".tar.gz") + "/"
    relative: set[str] = set()
    for name in names:
        if not name.startswith(prefix):
            errors.append(f"{path.name}: {name} is outside {prefix}")
            continue
        rel = name[len(prefix) :]
        relative.add(rel)
        if bad_member(rel):
            errors.append(f"{path.name}: cache or compiled file {rel}")
        elif rel not in SDIST_FILES and not rel.startswith(SDIST_DIRS):
            errors.append(f"{path.name}: unexpected file {rel}")
    errors += [f"{path.name}: missing {f}" for f in sorted(REQUIRED_SDIST - relative)]
    errors += check_readme(path.name, metadata)
    return errors


def check_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as whl:
        names = [n for n in whl.namelist() if not n.endswith("/")]
        meta_name = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
        metadata = whl.read(meta_name).decode() if meta_name else ""
    for name in names:
        top = name.split("/", 1)[0]
        if bad_member(name):
            errors.append(f"{path.name}: cache or compiled file {name}")
        elif top != "parsrag" and not top.endswith(".dist-info"):
            errors.append(f"{path.name}: unexpected file {name}")
    errors += [f"{path.name}: missing {f}" for f in sorted(REQUIRED_WHEEL - set(names))]
    errors += check_readme(path.name, metadata)
    return errors


def check_readme(label: str, metadata: str) -> list[str]:
    if not metadata:
        return [f"{label}: no metadata found"]
    message = Parser().parsestr(metadata)
    if message.get("Description-Content-Type") != "text/markdown":
        return [f"{label}: the readme is not declared as text/markdown"]
    body = message.get_payload()
    links = RELATIVE_LINK.findall(body if isinstance(body, str) else "")
    return [f"{label}: relative link left in the PyPI readme: {link}" for link in links[:5]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path, help="default: everything in dist/")
    args = parser.parse_args()
    files = args.files or sorted((ROOT / "dist").glob("*"))
    if not any(f.name.endswith((".tar.gz", ".whl")) for f in files):
        print("no sdist or wheel found — run `python -m build` first", file=sys.stderr)
        return 2
    errors: list[str] = []
    for f in files:
        if f.name.endswith(".tar.gz"):
            errors += check_sdist(f)
        elif f.suffix == ".whl":
            errors += check_wheel(f)
        else:
            continue
        print(f"checked {f.name}")
    for e in errors:
        print(f"✘ {e}", file=sys.stderr)
    if errors:
        return 1
    print("✔ the distributions contain only the library, its metadata and the allowed files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
