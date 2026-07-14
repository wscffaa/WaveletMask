#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


CLIPS = ["001", "005", "008", "009", "010", "012", "014", "018", "020", "022", "024", "029"]
METHODS = ["lq", "rtn", "waveletmask", "gt"]
FRAME_ID_RE = re.compile(r"^(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frame-id manifest for WaveletMask subjective-study clips."
    )
    parser.add_argument(
        "--stimuli-root",
        type=Path,
        default=Path(
            "/Users/caifeifan/VibeScholar/Antigravity_OFR/projects/007-waveletmask/Exp/paper_data/subjective_study/stimuli"
        ),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    return parser.parse_args()


def extract_frame_id(path: Path) -> int:
    match = FRAME_ID_RE.match(path.stem)
    if not match:
        raise ValueError(f"bad frame filename: {path}")
    return int(match.group(1))


def list_frame_ids(method_dir: Path) -> list[int]:
    return sorted(extract_frame_id(path) for path in method_dir.glob("*.png"))


def build_clip_record(stimuli_root: Path, clip_id: str) -> dict:
    clip_root = stimuli_root / clip_id
    if not clip_root.is_dir():
        raise FileNotFoundError(f"missing clip dir: {clip_root}")

    per_method = {}
    for method in METHODS:
        method_dir = clip_root / method
        if not method_dir.is_dir():
            raise FileNotFoundError(f"missing method dir: {method_dir}")
        frame_ids = list_frame_ids(method_dir)
        if not frame_ids:
            raise ValueError(f"no frames in: {method_dir}")
        per_method[method] = frame_ids

    baseline = per_method["rtn"]
    mismatched = {
        method: frame_ids
        for method, frame_ids in per_method.items()
        if frame_ids != baseline
    }
    if mismatched:
        raise ValueError(
            f"clip {clip_id} frame-id mismatch: "
            + ", ".join(f"{method}={len(ids)}" for method, ids in mismatched.items())
        )

    return {
        "clip_id": clip_id,
        "source_protocol": "reds_synthetic",
        "current_methods_present": METHODS,
        "frame_count": len(baseline),
        "selected_frame_ids": baseline,
        "per_method_counts": {method: len(ids) for method, ids in per_method.items()},
        "clip_root": str(clip_root),
    }


def write_json(path: Path, records: list[dict]) -> None:
    payload = {
        "source_protocol": "reds_synthetic",
        "clip_count": len(records),
        "methods": METHODS,
        "clips": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_id",
                "source_protocol",
                "frame_count",
                "current_methods_present",
                "selected_frame_ids",
                "lq_count",
                "rtn_count",
                "waveletmask_count",
                "gt_count",
                "clip_root",
            ],
        )
        writer.writeheader()
        for record in records:
            counts = record["per_method_counts"]
            writer.writerow(
                {
                    "clip_id": record["clip_id"],
                    "source_protocol": record["source_protocol"],
                    "frame_count": record["frame_count"],
                    "current_methods_present": ",".join(record["current_methods_present"]),
                    "selected_frame_ids": ",".join(
                        f"{frame_id:08d}" for frame_id in record["selected_frame_ids"]
                    ),
                    "lq_count": counts["lq"],
                    "rtn_count": counts["rtn"],
                    "waveletmask_count": counts["waveletmask"],
                    "gt_count": counts["gt"],
                    "clip_root": record["clip_root"],
                }
            )


def main() -> None:
    args = parse_args()
    records = [build_clip_record(args.stimuli_root, clip_id) for clip_id in CLIPS]
    write_json(args.json_out, records)
    write_csv(args.csv_out, records)
    print(f"wrote {len(records)} clips -> {args.json_out} and {args.csv_out}")


if __name__ == "__main__":
    main()
