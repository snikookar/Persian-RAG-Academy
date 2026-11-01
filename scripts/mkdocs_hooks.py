"""MkDocs hooks: publish ``lessons/*/README.fa.md`` as site pages without copying them.

Each lesson README becomes a generated page ``lessons/<slug>.md``. Its relative
links are rewritten for the site:

* another lesson's README          → that lesson's page
* a file that is part of the site  → a relative link to it (images, results, ...)
* any other file in the repository → its page on GitHub (code, starters, data)

The "edit this page" button of a lesson points at the README on GitHub.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Any

from mkdocs.structure.files import File, Files

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "lessons"
REPO = "https://github.com/snikookar/Persian-RAG-Academy"
LINK = re.compile(r"(!?\[[^\]]*\])\(([^)\s]+)\)")
FENCE = re.compile(r"^(```|~~~)")


def lesson_dirs() -> list[Path]:
    return sorted(p for p in LESSONS.iterdir() if (p / "README.fa.md").is_file())


def rewrite_links(markdown: str, readme: Path, docs_dir: Path, site_files: set[str]) -> str:
    """Rewrite the relative links of one lesson README (outside fenced code blocks)."""
    page_dir = "lessons"
    out: list[str] = []
    in_fence = False
    for line in markdown.splitlines(keepends=True):
        if FENCE.match(line.lstrip()):
            in_fence = not in_fence
        if in_fence:
            out.append(line)
            continue
        out.append(LINK.sub(lambda m: _rewrite(m, readme, docs_dir, site_files, page_dir), line))
    return "".join(out)


def _rewrite(
    match: re.Match[str], readme: Path, docs_dir: Path, site_files: set[str], page_dir: str
) -> str:
    label, target = match.groups()
    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
        return match.group(0)
    path, _, anchor = target.partition("#")
    anchor = f"#{anchor}" if anchor else ""
    resolved = (readme.parent / path).resolve()
    try:
        repo_path = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return match.group(0)
    if resolved.parent.parent == LESSONS and resolved.name == "README.fa.md":
        return f"{label}({resolved.parent.name}.md{anchor})"
    try:
        site_path = resolved.relative_to(docs_dir).as_posix()
    except ValueError:
        site_path = None
    if site_path in site_files:
        return f"{label}({posixpath.relpath(site_path, page_dir)}{anchor})"
    kind = "tree" if resolved.is_dir() else "blob"
    return f"{label}({REPO}/{kind}/main/{repo_path}{anchor})"


def on_files(files: Files, config: Any) -> Files:
    docs_dir = Path(config["docs_dir"]).resolve()
    site_files = {f.src_uri for f in files}
    for directory in lesson_dirs():
        readme = directory / "README.fa.md"
        text = rewrite_links(readme.read_text(encoding="utf-8"), readme, docs_dir, site_files)
        files.append(File.generated(config, f"lessons/{directory.name}.md", content=text))
    return files


def on_page_context(context: dict[str, Any], page: Any, config: Any, nav: Any) -> dict[str, Any]:
    src = page.file.src_uri
    if src.startswith("lessons/"):
        slug = src.removeprefix("lessons/").removesuffix(".md")
        page.edit_url = f"{REPO}/edit/main/lessons/{slug}/README.fa.md"
    return context


def on_page_content(html: str, page: Any, config: Any, files: Any) -> str:
    """The English summaries (``docs/en/``) read left to right inside the Persian site."""
    if page.file.src_uri.startswith("en/"):
        return f'<div class="ltr-page" dir="ltr" lang="en">\n{html}\n</div>'
    return html
