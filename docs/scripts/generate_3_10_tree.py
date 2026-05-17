"""Export project tree for diploma section 3.10."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "3_10"
SKIP = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".cursor",
    "node_modules",
    "artifacts",
    "data",
    "docs/backtest_journal/runs",
}
SKIP_EXT = {".pyc", ".parquet", ".pkl", ".jsonl"}


def should_skip(p: Path) -> bool:
    parts = set(p.parts)
    if any(s in parts for s in SKIP):
        return True
    if p.suffix in SKIP_EXT:
        return True
    if "backtest_journal" in str(p) and p.suffix == ".json":
        return True
    return False


def tree(path: Path, prefix: str = "", max_depth: int = 4, depth: int = 0, lines: list | None = None) -> list:
    lines = lines or []
    if depth > max_depth:
        return lines
    try:
        entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except OSError:
        return lines
    entries = [e for e in entries if not should_skip(e) and not e.name.startswith(".")]
    for i, e in enumerate(entries):
        last = i == len(entries) - 1
        branch = "+-- " if last else "|-- "
        name = e.name + ("/" if e.is_dir() else "")
        lines.append(prefix + branch + name)
        if e.is_dir():
            ext = "    " if last else "|   "
            tree(e, prefix + ext, max_depth, depth + 1, lines)
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["IST/", *tree(ROOT, max_depth=3)]
    text = "\n".join(lines)
    (OUT / "project_tree.txt").write_text(text, encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {OUT / 'project_tree.txt'}")


if __name__ == "__main__":
    main()
