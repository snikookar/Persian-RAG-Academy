"""scripts/make_badges.py turns a coverage XML report into a shields.io endpoint badge."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.make_badges import color, main


def test_coverage_badge(tmp_path: Path) -> None:
    report = tmp_path / "coverage.xml"
    report.write_text('<?xml version="1.0" ?><coverage line-rate="0.9661"></coverage>', "utf-8")
    assert main([str(report), str(tmp_path / "badges")]) == 0
    data = json.loads((tmp_path / "badges" / "coverage.json").read_text("utf-8"))
    assert data == {
        "schemaVersion": 1,
        "label": "coverage",
        "message": "97%",
        "color": "brightgreen",
    }
    assert [color(p) for p in (95, 85, 75, 50)] == ["brightgreen", "green", "yellow", "red"]
