import atexit
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
METADATA_DIR = REPO_ROOT / "run_metadata"


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _serialize_args(args):
    if args is None:
        return None
    d = vars(args) if hasattr(args, "__dict__") else dict(args)
    out = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = repr(v)
    return out


def _ts_for_filename(iso: str) -> str:
    return iso.replace(":", "").replace("-", "").replace("+00:00", "Z").replace(".", "")


def track_experiment(output_path, args=None, extra=None):
    started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started_perf = time.perf_counter()
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(str(output_path)).stem if output_path else "run"
    if not stem or stem == ".":
        stem = "run"
    meta_path = METADATA_DIR / f"{stem}__{_ts_for_filename(started_iso)}.json"

    data = {
        "cmd": " ".join(sys.argv),
        "script": Path(sys.argv[0]).name,
        "args": _serialize_args(args),
        "output_path": str(output_path) if output_path else None,
        "started_at": started_iso,
        "git_commit": _git_commit(),
        "status": "in_progress",
    }
    if extra:
        data.update(extra)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    def _finalize():
        finished_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        duration = round(time.perf_counter() - started_perf, 2)
        exc_type = sys.exc_info()[0]
        data["finished_at"] = finished_iso
        data["duration_seconds"] = duration
        data["status"] = "failed" if exc_type is not None else "completed"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    atexit.register(_finalize)
    return meta_path