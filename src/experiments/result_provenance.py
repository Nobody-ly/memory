"""Reproducibility metadata for experiment result bundles."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_run_manifest(config: Dict[str, Any], model: str | None) -> Dict[str, Any]:
    tracked_changes = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_tracked_changes": bool(tracked_changes),
        "source_sha256": _source_fingerprint(),
        "model": model,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config": config,
    }


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    targets = [REPO_ROOT / "src" / "experiments", REPO_ROOT / "src" / "llm_client.py"]
    files: List[Path] = []
    for target in targets:
        if target.is_dir():
            files.extend(target.rglob("*.py"))
        elif target.is_file():
            files.append(target)
    for path in sorted(set(files)):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
