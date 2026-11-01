"""Write shields.io endpoint badges for the docs site (stdlib only).

    pytest --cov=parsrag --cov-report=xml
    python scripts/make_badges.py coverage.xml site/badges

writes ``site/badges/coverage.json``; the README badge reads it from GitHub Pages
(``https://img.shields.io/endpoint?url=.../badges/coverage.json``), so the number
is always the one measured on ``main`` — never typed by hand.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def coverage_percent(report: Path) -> float:
    """Line coverage from a Cobertura XML report (``pytest --cov-report=xml``)."""
    rate = ET.parse(report).getroot().get("line-rate")
    if rate is None:
        raise SystemExit(f"{report}: no line-rate attribute (not a coverage XML report?)")
    return float(rate) * 100


def color(percent: float) -> str:
    if percent >= 90:
        return "brightgreen"
    if percent >= 80:
        return "green"
    if percent >= 70:
        return "yellow"
    return "red"


def badge(label: str, message: str, badge_color: str) -> dict[str, object]:
    return {"schemaVersion": 1, "label": label, "message": message, "color": badge_color}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("coverage_xml", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    percent = coverage_percent(args.coverage_xml)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = badge("coverage", f"{percent:.0f}%", color(percent))
    (args.out_dir / "coverage.json").write_text(json.dumps(data) + "\n", encoding="utf-8")
    print(f"coverage {percent:.1f}% → {args.out_dir / 'coverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
