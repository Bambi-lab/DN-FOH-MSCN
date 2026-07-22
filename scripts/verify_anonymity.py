from __future__ import annotations

import argparse
import re
from pathlib import Path


TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh", ".ps1"}
PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "windows_absolute_path": re.compile(r"[A-Za-z]:[\\/](?![ntr])[A-Za-z0-9_.\-]"),
    "home_path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "ssh_material": re.compile(r"(?:HostName|IdentityFile|PreferredAuthentications|\.ssh)", re.I),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    findings = []
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if path.resolve() == Path(__file__).resolve():
            # The scanner necessarily contains its own detection expressions.
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append((label, path.relative_to(args.root).as_posix()))
    if findings:
        for label, path in findings:
            print(f"{label}: {path}")
        raise SystemExit(1)
    print("ANONYMITY_SCAN_PASS")


if __name__ == "__main__":
    main()
