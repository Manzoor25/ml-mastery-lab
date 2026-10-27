"""Checks that the curriculum plan and repo contract stay intact."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_COUNTS = {
    "00-foundations": 4,
    "01-data-and-eda": 4,
    "02-feature-engineering": 10,
    "03-classical-supervised": 5,
    "04-unsupervised": 2,
    "05-model-evaluation": 5,
    "06-ensembles-and-boosting": 4,
    "07-time-series": 4,
    "08-deep-learning-basics": 3,
    "09-computer-vision": 5,
    "10-nlp": 4,
    "11-llms-and-genai": 7,
    "12-recommender-systems": 1,
    "13-reinforcement-learning": 4,
    "14-probabilistic-and-causal": 4,
    "15-anomaly-detection": 2,
    "16-mlops-and-production": 7,
    "17-responsible-ai": 4,
    "18-real-world-case-studies": 8,
}

EXPECTED_GUIDE_COUNT = 20
ALLOWED = ("todo", "in progress", "done")
ROW_RE = re.compile(r"^\| `([^`]+)` \| ([^|]+?) \| ([^|]+?) \|$")
NEXT_RE = re.compile(r"^Next: `([^`]+)`\s*$", re.MULTILINE)


def _block(text: str, start: str, end: str) -> str:
    start_at = text.index(start)
    end_at = text.index(end, start_at)
    return text[start_at:end_at]


def _rows(block: str) -> list[tuple[str, str, str]]:
    found = []
    for line in block.splitlines():
        match = ROW_RE.match(line.strip())
        if match:
            found.append((match.group(1), match.group(2).strip(), match.group(3).strip()))
    return found


def _status_ok(status: str) -> bool:
    return status in ALLOWED or status.startswith("blocked:")


def test_progress_plan_is_complete():
    text = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    projects = _rows(_block(text, "<!-- projects-start -->", "<!-- projects-end -->"))
    assert len(projects) == sum(EXPECTED_COUNTS.values())
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for path, status, teaches in projects:
        section = path.split("/", 1)[0]
        assert section in EXPECTED_COUNTS, path
        assert path not in seen, path
        seen.add(path)
        assert teaches
        assert _status_ok(status), status
        counts[section] = counts.get(section, 0) + 1
        if status == "done":
            project = ROOT / path
            assert (project / "main.py").is_file()
            assert (project / "results" / "metrics.json").is_file()
            assert (project / "README.md").is_file()
            assert (project / "LINKEDIN.md").is_file()
            assert (project / "tests" / "test_smoke.py").is_file()
    assert counts == EXPECTED_COUNTS

    match = NEXT_RE.search(text)
    assert match is not None
    unfinished = [path for path, status, _ in projects if status != "done"]
    assert match.group(1) == unfinished[0]

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for path, status, _ in projects:
        assert f"`{path}`" in readme
        if status != "done":
            assert path in readme
    not_done = sum(1 for _, status, _ in projects if status != "done")
    assert readme.count("| not run |") == not_done
    assert "Complete beginner" in readme
    assert "Interview in 4 weeks" in readme
    assert "Already working, fill the gaps" in readme


def test_guides_are_planned():
    text = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    guides = _rows(_block(text, "<!-- guides-start -->", "<!-- guides-end -->"))
    assert len(guides) == EXPECTED_GUIDE_COUNT
    for path, status, summary in guides:
        assert summary
        assert _status_ok(status), status
        if status == "done":
            assert (ROOT / path).is_file()


def test_required_files_exist():
    required = [
        "README.md",
        "PROGRESS.md",
        "LICENSE",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "Makefile",
        "CONTRIBUTING.md",
        ".github/workflows/ci.yml",
        "templates/README_TEMPLATE.md",
        "templates/LINKEDIN_TEMPLATE.md",
    ]
    for rel in required:
        assert (ROOT / rel).is_file(), rel


def test_data_directory_is_gitignored():
    ignored = {
        line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert "data/" in ignored


def test_requirements_are_pinned():
    for name in ("requirements.txt", "requirements-dev.txt"):
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-r"):
                continue
            assert "==" in stripped, stripped


def test_ci_lints_and_tests():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "ruff check ." in workflow
    assert "pytest" in workflow
