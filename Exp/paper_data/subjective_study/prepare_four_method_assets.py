#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(
    "/Users/caifeifan/VibeScholar/Antigravity_OFR/projects/007-waveletmask/Exp/paper_data/subjective_study"
)
DEFAULT_MANIFEST = Path(
    "/Users/caifeifan/VibeScholar/Antigravity_OFR/docs/workflow/10-active/"
    "20260607-222617-007-waveletmask-appsci-submission/09-evidence/raw/"
    "20260608-waveletmask_clip_frame_manifest.json"
)
DEFAULT_FRAMES_SCRIPT = Path(
    "/Users/caifeifan/VibeScholar/Antigravity_OFR/papers/shuphd-ofr-latex/_workspace/"
    "06-user-study/scripts/frames_to_video.sh"
)
FRAME_ID_RE = re.compile(r"^(\d+)")
EXISTING_VIDEO_METHODS = ("lq", "rtn", "waveletmask", "gt")
DEFAULT_SYNC_METHODS = ("rrtn", "mambaofr")


@dataclass(frozen=True)
class RemoteSource:
    host: str
    clip_root: Path


REMOTE_SOURCES = {
    "rtn": RemoteSource(
        host="L40_aliyun",
        clip_root=Path(
            "/mnt/data/OFR_Results/baselines/REDS_synthetic/rtn/rtn_synthetic/visualization/REDS_synthetic"
        ),
    ),
    "rrtn": RemoteSource(
        host="L40_aliyun",
        clip_root=Path(
            "/mnt/data/OFR_Results/baselines/REDS_synthetic/rrtn/rrtn_synthetic/visualization/REDS_synthetic"
        ),
    ),
    "mambaofr": RemoteSource(
        host="L40_aliyun",
        clip_root=Path(
            "/mnt/data/OFR_Results/baselines/REDS_synthetic/mambaofr/mambaofr_synthetic/visualization/REDS_synthetic"
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare four-method WaveletMask user-study assets from local and remote sources."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--current-videos-root", type=Path, default=BASE_DIR / "stimuli_videos")
    parser.add_argument(
        "--frame-staging-root", type=Path, default=BASE_DIR / "stimuli_four_method_frames"
    )
    parser.add_argument("--videos-root", type=Path, default=BASE_DIR / "stimuli_videos_v2")
    parser.add_argument("--frames-script", type=Path, default=DEFAULT_FRAMES_SCRIPT)
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument(
        "--sync-methods",
        nargs="*",
        default=list(DEFAULT_SYNC_METHODS),
        choices=sorted(REMOTE_SOURCES),
    )
    parser.add_argument(
        "--encode-methods",
        nargs="*",
        default=list(DEFAULT_SYNC_METHODS),
        choices=sorted(REMOTE_SOURCES),
    )
    parser.add_argument("--skip-copy-existing-videos", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def extract_frame_id(path: Path) -> int:
    match = FRAME_ID_RE.match(path.stem)
    if not match:
        raise ValueError(f"bad frame filename: {path}")
    return int(match.group(1))


def list_frame_ids(frame_dir: Path) -> list[int]:
    return sorted(extract_frame_id(path) for path in frame_dir.glob("*.png"))


def load_manifest(path: Path, selected_clips: list[str] | None = None) -> dict[str, list[int]]:
    payload = json.loads(path.read_text())
    wanted = None if not selected_clips else {clip.strip() for clip in selected_clips}
    records: dict[str, list[int]] = {}
    for record in payload["clips"]:
        clip_id = record["clip_id"]
        if wanted is not None and clip_id not in wanted:
            continue
        records[clip_id] = list(record["selected_frame_ids"])
    if wanted is not None:
        missing = sorted(wanted - set(records))
        if missing:
            raise ValueError(f"clips missing from manifest: {', '.join(missing)}")
    return dict(sorted(records.items()))


def run_command(cmd: list[str], capture: bool = False) -> None:
    if not capture:
        subprocess.run(cmd, check=True)
        return
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode == 0:
        return
    raise subprocess.CalledProcessError(
        completed.returncode,
        cmd,
        output=completed.stdout,
        stderr=completed.stderr,
    )


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_existing_videos(
    clip_frames: dict[str, list[int]],
    current_videos_root: Path,
    videos_root: Path,
    force: bool,
) -> int:
    copied = 0
    for clip_id in clip_frames:
        src_dir = current_videos_root / f"clip_{clip_id}"
        dst_dir = videos_root / f"clip_{clip_id}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for method in EXISTING_VIDEO_METHODS:
            src = src_dir / f"{method}.mp4"
            if not src.is_file():
                raise FileNotFoundError(f"missing source video: {src}")
            dst = dst_dir / f"{method}.mp4"
            if dst.exists() and not force:
                continue
            shutil.copy2(src, dst)
            copied += 1
    return copied


def validate_frame_dir(frame_dir: Path, expected_ids: list[int]) -> None:
    actual_ids = list_frame_ids(frame_dir)
    if actual_ids != expected_ids:
        raise ValueError(
            f"frame-id mismatch in {frame_dir}: expected {len(expected_ids)} frames, got {len(actual_ids)}"
        )


def sync_remote_method(
    remote_source: RemoteSource,
    clip_id: str,
    expected_ids: list[int],
    dst_dir: Path,
    force: bool,
) -> bool:
    if dst_dir.exists() and not force:
        validate_frame_dir(dst_dir, expected_ids)
        return False

    clean_dir(dst_dir)
    remote_dir = f"{remote_source.host}:{(remote_source.clip_root / clip_id / 'outputs').as_posix()}/"
    run_command(["rsync", "-a", "--delete", remote_dir, f"{dst_dir.as_posix()}/"])
    validate_frame_dir(dst_dir, expected_ids)
    return True


def encode_method_video(
    frames_script: Path,
    frame_dir: Path,
    out_path: Path,
    force: bool,
) -> bool:
    if out_path.exists() and not force:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "bash",
            str(frames_script),
            str(frame_dir),
            str(out_path),
            "24",
            "2",
            "720",
        ],
        capture=True,
    )
    return True


def main() -> None:
    args = parse_args()
    clip_frames = load_manifest(args.manifest, args.clips)
    sync_methods = list(dict.fromkeys(args.sync_methods))
    encode_methods = list(dict.fromkeys(args.encode_methods))

    if not args.skip_copy_existing_videos:
        copied = copy_existing_videos(
            clip_frames, args.current_videos_root, args.videos_root, args.force
        )
        print(f"copied_existing_videos={copied}")

    synced = 0
    if not args.skip_sync:
        for method in sync_methods:
            remote_source = REMOTE_SOURCES[method]
            for clip_id, expected_ids in clip_frames.items():
                dst_dir = args.frame_staging_root / clip_id / method
                if sync_remote_method(remote_source, clip_id, expected_ids, dst_dir, args.force):
                    synced += 1
        print(f"synced_frame_dirs={synced}")

    encoded = 0
    if not args.skip_encode:
        for method in encode_methods:
            for clip_id in clip_frames:
                frame_dir = args.frame_staging_root / clip_id / method
                if not frame_dir.is_dir():
                    raise FileNotFoundError(f"missing frame dir for encode: {frame_dir}")
                out_path = args.videos_root / f"clip_{clip_id}" / f"{method}.mp4"
                if encode_method_video(args.frames_script, frame_dir, out_path, args.force):
                    encoded += 1
        print(f"encoded_videos={encoded}")

    print(f"clips={len(clip_frames)} methods_synced={','.join(sync_methods)}")


if __name__ == "__main__":
    main()
