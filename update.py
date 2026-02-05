#!/usr/bin/env python3
"""
Incremental updater for patents-rag (XML-only).

Pipeline:
  1) Determine last processed XML zip from checkpoints/build-progress.txt
  2) Download newer XML zips via download.py
  3) Run build.py to add them to FAISS + titles
  4) Redeploy MCP server via deploy.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys


CHECKPOINT_PATH = os.path.join("checkpoints", "build-progress.txt")
LOCK_PATH = ".update.lock"


def _parse_ipg_date(filename: str) -> dt.date | None:
    """
    Parse dates from filenames like:
      ipg260203.zip -> 2026-02-03
      pg041207.zip  -> 2004-12-07
    """
    m = re.search(r"\b(?:ipg|pg)(\d{6})(?:_r\d+)?\.zip\b", filename)
    if not m:
        return None
    yymmdd = m.group(1)
    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    return dt.date(2000 + yy, mm, dd)


def _latest_checkpoint_date(path: str) -> dt.date | None:
    if not os.path.exists(path):
        return None
    latest: dt.date | None = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = _parse_ipg_date(line)
            if d is None:
                continue
            if latest is None or d > latest:
                latest = d
    return latest


def _acquire_lock() -> int:
    # atomic create; fail if exists
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(LOCK_PATH, flags, 0o644)
    except FileExistsError:
        return -1
    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.close(fd)
    return 0


def _release_lock() -> None:
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


def _run(cmd: list[str], dry_run: bool) -> int:
    print("+", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental update: download newer XML, build, deploy.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them.")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    if _acquire_lock() != 0:
        print(f"{LOCK_PATH} exists; another update may be running. Remove it if stale.", file=sys.stderr)
        return 2

    try:
        latest = _latest_checkpoint_date(CHECKPOINT_PATH)
        if latest is None:
            start = dt.date(2002, 1, 1)
        else:
            start = latest + dt.timedelta(days=1)
        end = dt.date.today()

        print(f"Latest checkpoint date: {latest.isoformat() if latest else 'none'}")
        print(f"Updating range: {start.isoformat()} -> {end.isoformat()}")

        if start > end:
            print("No newer dates to download. Running build+deploy anyway.")

        py = sys.executable or "python3"

        rc = _run([py, "build.py", "--xml-only", "--xml-start", start.isoformat(), "--xml-end", end.isoformat()], args.dry_run)
        if rc != 0:
            return rc

        rc = _run([py, "deploy.py"], args.dry_run)
        return rc
    finally:
        _release_lock()


if __name__ == "__main__":
    raise SystemExit(main())

