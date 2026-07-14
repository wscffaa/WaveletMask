"""Model complexity analysis utilities.

Compute Params / MACs / FLOPs / Runtime for OFR architectures.
Uses calflops (recommended) → fvcore → PyTorch FlopCounterMode → manual.

Usage:
    python -m basicofr.utils.complexity --arch MambaOFRNet
    python -m basicofr.utils.complexity --arch MambaOFRNet --input-size 1,7,3,256,256 --json
"""
import argparse
import json
import sys
import time

import torch
import torch.nn as nn


def _count_module_params(module: nn.Module, trainable_only: bool = False) -> int:
    """Return parameter count for one nn.Module tree."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _mark_module_tree(module: nn.Module, seen_modules: set) -> None:
    """Mark one module tree as visited to avoid double counting."""
    for submodule in module.modules():
        seen_modules.add(id(submodule))


def _count_non_module_params(obj, seen_objs: set, seen_modules: set) -> int:
    """Recursively count nn.Module params stored in non-Module attributes."""
    obj_id = id(obj)
    if obj_id in seen_objs:
        return 0
    seen_objs.add(obj_id)

    if isinstance(obj, dict):
        values = obj.values()
    elif isinstance(obj, (list, tuple, set)):
        values = obj
    elif hasattr(obj, '__dict__'):
        values = vars(obj).values()
    else:
        return 0

    total = 0
    for value in values:
        if isinstance(value, nn.Module):
            if id(value) not in seen_modules:
                total += _count_module_params(value)
                _mark_module_tree(value, seen_modules)
            continue

        nested_model = getattr(value, 'model', None)
        if isinstance(nested_model, nn.Module) and id(nested_model) not in seen_modules:
            total += _count_module_params(nested_model)
            _mark_module_tree(nested_model, seen_modules)

        total += _count_non_module_params(value, seen_objs, seen_modules)

    return total


def count_params(model: nn.Module, include_non_module: bool = True) -> float:
    """Return total parameter count in millions (M)."""
    total = _count_module_params(model)
    if include_non_module:
        seen_objs = set()
        seen_modules = {id(module) for module in model.modules()}
        total += _count_non_module_params(model, seen_objs, seen_modules)
    return total / 1e6


def count_flops(model: nn.Module,
                input_size: tuple = (1, 7, 3, 256, 256),
                device: str = 'cuda') -> dict:
    """Estimate MACs and FLOPs.

    Returns dict with macs_g and flops_g.
    Fallback: torch_builtin → calflops → fvcore → manual params only.
    """
    result = {'macs_g': -1.0, 'flops_g': -1.0}

    # Auto-detect input channels from model (e.g. RRTN input_channel=1)
    in_ch = getattr(model, 'input_channel', None)
    if in_ch is not None and len(input_size) == 5 and input_size[2] != in_ch:
        input_size = (*input_size[:2], in_ch, *input_size[3:])

    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)

    # 1) PyTorch built-in FlopCounterMode (>= 2.1) — most robust
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fcm:
            with torch.no_grad():
                model(dummy)
        total = fcm.get_total_flops()
        result['flops_g'] = round(total / 1e9, 3)
        result['macs_g'] = round(total / 2e9, 3)
        result['backend'] = 'torch_builtin'
        result['input_size'] = list(input_size)
        return result
    except Exception:
        pass

    # 2) calflops — good for functional ops
    try:
        from calflops import calculate_flops
        flops, macs, params = calculate_flops(
            model=model,
            input_shape=input_size,
            output_as_string=False,
            print_results=False,
        )
        result['flops_g'] = round(flops / 1e9, 3)
        result['macs_g'] = round(macs / 1e9, 3)
        result['backend'] = 'calflops'
        result['input_size'] = list(input_size)
        return result
    except Exception:
        pass

    # 3) fvcore — module-level breakdown
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, (dummy,))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = fca.total()
        result['flops_g'] = round(total / 1e9, 3)
        result['macs_g'] = round(total / 2e9, 3)
        result['backend'] = 'fvcore'
        result['input_size'] = list(input_size)
        return result
    except Exception:
        pass

    result['backend'] = 'none'
    return result


def measure_runtime(model: nn.Module,
                    input_size: tuple = (1, 7, 3, 256, 256),
                    n_warmup: int = 3,
                    n_runs: int = 10,
                    device: str = 'cuda') -> float:
    """Average inference time in ms (CUDA event timing)."""
    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)
    use_cuda = device.startswith('cuda') and torch.cuda.is_available()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
            if use_cuda:
                torch.cuda.synchronize()

    if use_cuda:
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                se.record()
                model(dummy)
                ee.record()
                torch.cuda.synchronize()
                times.append(se.elapsed_time(ee))
        return sum(times) / len(times)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def measure_peak_memory(model: nn.Module,
                        input_size: tuple = (1, 7, 3, 256, 256),
                        n_warmup: int = 2,
                        device: str = 'cuda') -> float:
    """Peak CUDA memory (GB) for a single inference forward pass.

    Returns -1.0 when not on an available CUDA device: GPU peak memory cannot be
    measured on CPU and must not be faked as 0. Mirrors the round-2 WaveletMask
    benchmark (``torch.cuda.max_memory_allocated`` after warmup + reset).
    """
    if not (device.startswith('cuda') and torch.cuda.is_available()):
        return -1.0

    in_ch = getattr(model, 'input_channel', None)
    if in_ch is not None and len(input_size) == 5 and input_size[2] != in_ch:
        input_size = (*input_size[:2], in_ch, *input_size[3:])

    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)
    try:
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            model(dummy)
            torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated(device)
    finally:
        del dummy
        torch.cuda.empty_cache()
    return round(peak_bytes / (1024 ** 3), 3)


def analyze_model(model: nn.Module,
                  model_name: str = 'unknown',
                  input_size: tuple = (1, 7, 3, 256, 256),
                  device: str = 'cuda',
                  measure_speed: bool = True) -> dict:
    """Full complexity analysis → dict."""
    # Auto-detect input channels (e.g. RRTN input_channel=1 for grayscale)
    in_ch = getattr(model, 'input_channel', None)
    if in_ch is not None and len(input_size) == 5 and input_size[2] != in_ch:
        input_size = (*input_size[:2], in_ch, *input_size[3:])

    params = _count_module_params(model, trainable_only=True) / 1e6
    params_all = count_params(model)
    flop_info = count_flops(model, input_size, device)
    runtime = -1.0
    if measure_speed:
        try:
            runtime = measure_runtime(model, input_size, device=device)
        except Exception as e:
            print(f"[WARN] Runtime failed: {e}", file=sys.stderr)

    n_frames = input_size[1] if len(input_size) == 5 else 1
    per_frame_runtime = round(runtime / n_frames, 2) if runtime >= 0 and n_frames > 0 else -1
    flops_paper = flop_info['macs_g']
    per_frame_flops_paper = round(flops_paper / n_frames, 3) if flops_paper >= 0 and n_frames > 0 else -1
    # FPS = frames per second from the raw runtime (un-rounded), so it stays
    # accurate even when per-frame runtime rounds to few decimals.
    fps = round(n_frames * 1000.0 / runtime, 2) if runtime > 0 and n_frames > 0 else -1

    peak_mem_gb = -1.0
    if measure_speed:
        try:
            peak_mem_gb = measure_peak_memory(model, input_size, device=device)
        except Exception as e:
            print(f"[WARN] Peak memory failed: {e}", file=sys.stderr)

    return {
        'name': model_name,
        'params_m': round(params, 3),
        'params_all_m': round(params_all, 3),
        'macs_g': flop_info['macs_g'],
        'flops_g': flop_info['flops_g'],
        'flops_paper_g': flops_paper,
        'per_frame_flops_paper_g': per_frame_flops_paper,
        'flops_backend': flop_info.get('backend', 'none'),
        'runtime_ms': round(runtime, 2) if runtime >= 0 else -1,
        'ms_per_frame': per_frame_runtime,
        'per_frame_runtime_ms': per_frame_runtime,
        'fps': fps,
        'peak_mem_gb': peak_mem_gb,
        'input_size': list(input_size),
    }


def print_table(results: list, paper_format: bool = False):
    """Pretty-print results table."""
    if paper_format:
        hdr = (f"{'Method':<20} {'Params(M)':>10} {'FLOPs(G)*':>10} "
               f"{'Runtime(ms/frame)':>18} {'FPS':>8} {'PeakMem(GB)':>12}")
        print(hdr)
        print('-' * len(hdr))
        for r in results:
            params = r.get('params_all_m', r.get('params_m', -1))
            flops = r.get('per_frame_flops_paper_g', -1)
            runtime = r.get('per_frame_runtime_ms', r.get('ms_per_frame', -1))
            fps = r.get('fps', -1)
            peak = r.get('peak_mem_gb', -1)
            p = f"{params:.3f}" if params >= 0 else 'N/A'
            f = f"{flops:.3f}" if flops >= 0 else 'N/A'
            rt = f"{runtime:.2f}" if runtime >= 0 else 'N/A'
            fp = f"{fps:.2f}" if fps >= 0 else 'N/A'
            pk = f"{peak:.3f}" if peak >= 0 else 'N/A'
            print(f"{r['name']:<20} {p:>10} {f:>10} {rt:>18} {fp:>8} {pk:>12}")
        print("* FLOPs measured as MACs (1 MAC = 1 FLOP convention, consistent with fvcore/thop)")
        return

    hdr = (f"{'Model':<20} {'Params(M)':>10} {'MACs(G)':>10} {'FLOPs(G)':>10} "
           f"{'Runtime(ms)':>12} {'FPS':>8} {'PeakMem(GB)':>12} {'Backend':>12}")
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        m = f"{r['macs_g']:.2f}" if r['macs_g'] >= 0 else 'N/A'
        f = f"{r['flops_g']:.2f}" if r['flops_g'] >= 0 else 'N/A'
        rt = f"{r['runtime_ms']:.1f}" if r['runtime_ms'] >= 0 else 'N/A'
        fp = f"{r.get('fps', -1):.2f}" if r.get('fps', -1) >= 0 else 'N/A'
        pk = f"{r.get('peak_mem_gb', -1):.3f}" if r.get('peak_mem_gb', -1) >= 0 else 'N/A'
        be = r.get('flops_backend', '?')
        print(f"{r['name']:<20} {r['params_m']:>10.3f} {m:>10} {f:>10} {rt:>12} {fp:>8} {pk:>12} {be:>12}")


def _build_model(arch_name: str) -> nn.Module:
    """Build from BasicSR ARCH_REGISTRY."""
    try:
        import basicofr.archs  # trigger auto-registration
        from basicsr.utils.registry import ARCH_REGISTRY
        cls = ARCH_REGISTRY.get(arch_name)
        if cls is None:
            avail = list(ARCH_REGISTRY._obj_map.keys())
            print(f"[ERROR] '{arch_name}' not in registry. Available: {avail}", file=sys.stderr)
            sys.exit(1)
        return cls()
    except Exception as e:
        print(f"[ERROR] Cannot build '{arch_name}': {e}", file=sys.stderr)
        sys.exit(1)


def build_model_from_config(config_path: str):
    """Build a model from a training/test YAML's ``network_g`` block.

    Training-config-faithful: instantiates the arch with the exact
    hyper-parameters it was trained with (e.g. ``num_feat``), instead of the
    arch's bare ``cls()`` defaults. Reported Params/FLOPs therefore match the
    weights behind the paper's accuracy numbers.

    Returns (net_type, model).
    """
    try:
        import yaml

        import basicofr.archs  # trigger auto-registration
        from basicsr.utils.registry import ARCH_REGISTRY
    except Exception as e:
        print(f"[ERROR] import failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Cannot read config '{config_path}': {e}", file=sys.stderr)
        sys.exit(1)

    net_cfg = dict((cfg or {}).get('network_g') or {})
    net_type = net_cfg.pop('type', None)
    if net_type is None:
        print(f"[ERROR] '{config_path}': missing network_g.type", file=sys.stderr)
        sys.exit(1)

    cls = ARCH_REGISTRY.get(net_type)
    if cls is None:
        avail = list(ARCH_REGISTRY._obj_map.keys())
        print(f"[ERROR] '{net_type}' not in registry. Available: {avail}", file=sys.stderr)
        sys.exit(1)

    try:
        return net_type, cls(**net_cfg)
    except Exception as e:
        print(f"[ERROR] Cannot build '{net_type}' from '{config_path}': {e}", file=sys.stderr)
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description='Model complexity analyzer')
    p.add_argument('--arch', default=None, help='Registry arch name (default-config build)')
    p.add_argument('--config', default=None,
                   help='YAML with network_g block (training-config-faithful build)')
    p.add_argument('--input-size', default='1,7,3,256,256')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-speed', action='store_true')
    p.add_argument('--json', action='store_true')
    p.add_argument('--paper', action='store_true', help='Print paper-format table')
    args = p.parse_args()

    sz = tuple(int(x) for x in args.input_size.split(','))
    if args.config:
        name, model = build_model_from_config(args.config)
    elif args.arch:
        name, model = args.arch, _build_model(args.arch)
    else:
        p.error('one of --arch / --config is required')
    r = analyze_model(model, name, sz, args.device, not args.no_speed)

    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print_table([r], paper_format=args.paper)


if __name__ == '__main__':
    main()
