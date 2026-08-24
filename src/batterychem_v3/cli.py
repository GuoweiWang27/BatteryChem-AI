from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .report import build_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="batterychem-v3")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", default="RULE", choices=("RULE", "LOGISTIC_AUGMENTED"))
    parser.add_argument("--boundary-student-signed", action="store_true")
    parser.add_argument("--boundary-second-signed", action="store_true")
    return parser


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    os.replace(temporary, path)


def main() -> None:
    args = _parser().parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_report(
        payload["candidates"],
        args.config_dir,
        mode=args.mode,
        student_signed=args.boundary_student_signed,
        second_signed=args.boundary_second_signed,
        external_dataset_audit=payload.get("external_dataset_audit"),
        synthetic_window=payload.get("synthetic_reactivity_window"),
    )
    rendered = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_write(args.output, rendered)


if __name__ == "__main__":
    main()
