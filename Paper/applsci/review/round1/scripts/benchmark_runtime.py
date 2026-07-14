#!/usr/bin/env python3
"""Runtime benchmark for WaveletMask round-1 revision.

Measures architecture-level inference latency and peak CUDA memory on synthetic
inputs. Checkpoints are intentionally not loaded because latency and memory are
determined by the deployed graph, not by trained weight values.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "framework").is_dir() and (parent / "projects").is_dir():
            return parent
    raise RuntimeError("Cannot locate repo root containing framework/ and projects/.")


REPO_ROOT = find_repo_root()
FRAMEWORK_ROOT = REPO_ROOT / "framework"
sys.path.insert(0, str(FRAMEWORK_ROOT))
os.environ.setdefault("OFR_PROJECTS_DIR", str(REPO_ROOT / "projects"))

import torch  # noqa: E402


ARCHES = {
    "RTN baseline": "WaveletMaskBaseNet",
    "WaveletMask": "WaveletMaskABNet",
}


def parse_resolution(text: str) -> tuple[str, int, int]:
    if "=" in text:
        label, value = text.split("=", 1)
    else:
        value = text
        label = text
    width_text, height_text = value.lower().split("x", 1)
    return label, int(height_text), int(width_text)


def build_model(arch_name: str):
    import basicofr.archs  # noqa: F401
    from basicsr.utils.registry import ARCH_REGISTRY

    cls = ARCH_REGISTRY.get(arch_name)
    if cls is None:
        raise RuntimeError(f"{arch_name} is not registered.")
    return cls(num_feat=16, num_block=6, flow_type="raft")


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def count_param_scopes(model: torch.nn.Module) -> dict[str, int]:
    registered_trainable, registered_total = count_params(model)
    flow_module = getattr(getattr(model, "flow_estimator", None), "model", None)
    if isinstance(flow_module, torch.nn.Module):
        flow_trainable, flow_total = count_params(flow_module)
    else:
        flow_trainable, flow_total = 0, 0
    return {
        "registered_trainable_params": registered_trainable,
        "registered_total_params": registered_total,
        "flow_estimator_trainable_params": flow_trainable,
        "flow_estimator_total_params": flow_total,
        "full_pipeline_trainable_params": registered_trainable + flow_trainable,
        "full_pipeline_total_params": registered_total + flow_total,
    }


def cuda_info(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "total_memory_mb": round(props.total_memory / 1024 / 1024, 1),
        "cuda_runtime": torch.version.cuda,
    }


def summarize_times(times_ms: list[float]) -> dict:
    return {
        "mean_ms": round(statistics.fmean(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "std_ms": round(statistics.pstdev(times_ms), 3) if len(times_ms) > 1 else 0.0,
        "min_ms": round(min(times_ms), 3),
        "max_ms": round(max(times_ms), 3),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_one(
    *,
    model_name: str,
    arch_name: str,
    batch: int,
    frames: int,
    height: int,
    width: int,
    warmup: int,
    runs: int,
    device: torch.device,
) -> dict:
    result = {
        "model": model_name,
        "arch": arch_name,
        "batch": batch,
        "frames": frames,
        "height": height,
        "width": width,
        "input_shape": [batch, frames, 3, height, width],
        "warmup": warmup,
        "runs": runs,
        "status": "pending",
    }
    model = None
    dummy = None
    try:
        model = build_model(arch_name).to(device).eval()
        scopes = count_param_scopes(model)
        result.update(scopes)
        # Backward-compatible aliases kept for older readers of the JSON artifact.
        result["trainable_params"] = scopes["registered_trainable_params"]
        result["total_params"] = scopes["registered_total_params"]
        result["trainable_params_m"] = round(scopes["registered_trainable_params"] / 1_000_000, 4)
        result["total_params_m"] = round(scopes["registered_total_params"] / 1_000_000, 4)
        result["registered_trainable_params_m"] = round(scopes["registered_trainable_params"] / 1_000_000, 4)
        result["registered_total_params_m"] = round(scopes["registered_total_params"] / 1_000_000, 4)
        result["flow_estimator_trainable_params_m"] = round(scopes["flow_estimator_trainable_params"] / 1_000_000, 4)
        result["flow_estimator_total_params_m"] = round(scopes["flow_estimator_total_params"] / 1_000_000, 4)
        result["full_pipeline_trainable_params_m"] = round(scopes["full_pipeline_trainable_params"] / 1_000_000, 4)
        result["full_pipeline_total_params_m"] = round(scopes["full_pipeline_total_params"] / 1_000_000, 4)

        dummy = torch.randn(batch, frames, 3, height, width, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        with torch.inference_mode():
            for _ in range(warmup):
                model(dummy)
                synchronize(device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            times_ms: list[float] = []
            for _ in range(runs):
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(dummy)
                    end.record()
                    synchronize(device)
                    times_ms.append(float(start.elapsed_time(end)))
                else:
                    t0 = time.perf_counter()
                    model(dummy)
                    times_ms.append((time.perf_counter() - t0) * 1000.0)

        timing = summarize_times(times_ms)
        latency_ms = timing["mean_ms"]
        result.update(timing)
        result["latency_ms_per_clip"] = round(latency_ms / batch, 3)
        result["latency_ms_per_frame"] = round(latency_ms / (batch * frames), 3)
        result["throughput_clips_per_s"] = round(batch * 1000.0 / latency_ms, 3)
        result["throughput_frames_per_s"] = round(batch * frames * 1000.0 / latency_ms, 3)
        if device.type == "cuda":
            result["peak_memory_allocated_mb"] = round(
                torch.cuda.max_memory_allocated(device) / 1024 / 1024,
                1,
            )
            result["peak_memory_reserved_mb"] = round(
                torch.cuda.max_memory_reserved(device) / 1024 / 1024,
                1,
            )
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom"
        result["error"] = str(exc).splitlines()[0]
    except RuntimeError as exc:
        text = str(exc)
        result["status"] = "oom" if "out of memory" in text.lower() else "error"
        result["error"] = text.splitlines()[0]
    except Exception as exc:  # fail visible in JSON
        result["status"] = "error"
        result["error"] = repr(exc)
    finally:
        del dummy
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def write_markdown(path: Path, payload: dict) -> None:
    rows = payload["results"]
    lines = [
        "# T9 Runtime Benchmark",
        "",
        "Architecture-level inference benchmark on synthetic inputs. No checkpoint weights were loaded.",
        "Parameter scope note: `Params(reg)` counts only the registered restoration module. `Params(full)` adds the RAFT flow estimator used by the executable forward pass.",
        "",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Device: `{payload['device']}`",
        f"- PyTorch: `{payload['torch_version']}`",
        f"- Protocol: FP32, eval mode, inference_mode, warmup={payload['warmup']}, runs={payload['runs']}",
        "",
        "| Model | Batch | Resolution | Status | Latency ms/clip | ms/frame | Clips/s | Frames/s | Peak allocated MB | Params(reg) M | Params(full) M |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        resolution = f"{row['width']}x{row['height']}"
        if row["status"] == "ok":
            lines.append(
                "| {model} | {batch} | {resolution} | {status} | {clip:.3f} | {frame:.3f} | "
                "{clips:.3f} | {frames:.3f} | {mem:.1f} | {params_reg:.4f} | {params_full:.4f} |".format(
                    model=row["model"],
                    batch=row["batch"],
                    resolution=resolution,
                    status=row["status"],
                    clip=row["latency_ms_per_clip"],
                    frame=row["latency_ms_per_frame"],
                    clips=row["throughput_clips_per_s"],
                    frames=row["throughput_frames_per_s"],
                    mem=row.get("peak_memory_allocated_mb", -1),
                    params_reg=row.get("registered_total_params_m", row.get("total_params_m", -1)),
                    params_full=row.get("full_pipeline_total_params_m", -1),
                )
            )
        else:
            lines.append(
                f"| {row['model']} | {row['batch']} | {resolution} | {row['status']}: "
                f"{row.get('error', '')} |  |  |  |  |  | {row.get('registered_total_params_m', row.get('total_params_m', -1))} | {row.get('full_pipeline_total_params_m', -1)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--batches", default="1,4")
    parser.add_argument("--resolutions", default="256=256x256,720p=1280x720")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    batches = [int(item) for item in args.batches.split(",") if item]
    resolutions = [parse_resolution(item) for item in args.resolutions.split(",") if item]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hardware": cuda_info(device),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "warmup": args.warmup,
        "runs": args.runs,
        "note": "Architecture-level inference benchmark; checkpoint weights are not loaded.",
        "results": [],
    }

    for model_name, arch_name in ARCHES.items():
        for _, height, width in resolutions:
            for batch in batches:
                row = run_one(
                    model_name=model_name,
                    arch_name=arch_name,
                    batch=batch,
                    frames=args.frames,
                    height=height,
                    width=width,
                    warmup=args.warmup,
                    runs=args.runs,
                    device=device,
                )
                payload["results"].append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    json_path = args.output_dir / "runtime_benchmark.json"
    md_path = args.output_dir / "runtime_benchmark.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
