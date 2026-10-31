import warnings
# 屏蔽 mcPyTorch NHWC 布局警告
warnings.filterwarnings("ignore", message=".*view size is not compatible.*")

import os
import os.path as osp
import sys
import types
from pathlib import Path
from typing import List, Optional, Set

# 可选：加载实验扩展（将扩展根目录加入 sys.path，供 basicofr 命名空间扩展使用）
def _infer_ofr_ext_paths_from_argv(root_path: str) -> List[str]:
    """从命令行 -opt/--opt 推断扩展根目录（适配 ofr_projects/<exp>/options/...）。"""
    opt_path: Optional[str] = None
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg in ('-opt', '--opt') and i + 1 < len(argv):
            opt_path = argv[i + 1]
            break
        if arg.startswith('-opt='):
            opt_path = arg.split('=', 1)[1]
            break
        if arg.startswith('--opt='):
            opt_path = arg.split('=', 1)[1]
            break
    if not opt_path:
        return []
    opt_path_p = Path(opt_path)
    if not opt_path_p.is_absolute():
        opt_path_p = (Path(root_path) / opt_path_p).resolve()
    parts = opt_path_p.parts
    if 'ofr_projects' not in parts:
        return []
    idx = parts.index('ofr_projects')
    if idx + 1 >= len(parts):
        return []
    ext_root = Path(*parts[:idx + 2])
    return [str(ext_root)] if ext_root.is_dir() else []


def _read_ofr_deps(ext_root: str, repo_root: str) -> List[str]:
    deps_file = Path(ext_root) / 'ofr_deps.txt'
    if not deps_file.is_file():
        return []
    deps: List[str] = []
    for raw in deps_file.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if '#' in line:
            line = line.split('#', 1)[0].strip()
        if not line:
            continue
        p = Path(line)
        if not p.is_absolute():
            cand = (Path(ext_root) / p).resolve()
            if cand.is_dir():
                deps.append(str(cand))
                continue
            cand = (Path(repo_root) / p).resolve()
            if cand.is_dir():
                deps.append(str(cand))
                continue
            continue
        if p.is_dir():
            deps.append(str(p))
    return deps


def _expand_ext_paths_with_deps(ext_paths: List[str], repo_root: str) -> List[str]:
    expanded: List[str] = []
    seen: Set[str] = set()

    def visit(path_str: str) -> None:
        if path_str in seen:
            return
        seen.add(path_str)
        expanded.append(path_str)
        for dep in _read_ofr_deps(path_str, repo_root):
            visit(dep)

    for p in ext_paths:
        visit(p)
    return expanded


def _prepend_ofr_ext_paths(root_path: str) -> None:
    ext_paths_env = os.environ.get('OFR_EXT_PATH', '').strip()
    ext_paths = (
        ext_paths_env.split(os.pathsep)
        if ext_paths_env
        else _infer_ofr_ext_paths_from_argv(root_path)
    )
    if not ext_paths:
        return
    norm_paths: List[str] = []
    for raw in ext_paths:
        raw = raw.strip()
        if not raw:
            continue
        ext_path = raw
        if not osp.isabs(ext_path):
            ext_path = osp.abspath(osp.join(root_path, ext_path))
        if osp.isdir(ext_path):
            norm_paths.append(ext_path)

    norm_paths = _expand_ext_paths_with_deps(norm_paths, root_path)

    for ext_path in reversed(norm_paths):
        sys.path.insert(0, ext_path)

# 确保本地 basicsr 和 basicofr 优先
ROOT_PATH = osp.abspath(osp.dirname(__file__))
_prepend_ofr_ext_paths(ROOT_PATH)
sys.path.insert(0, ROOT_PATH)

# 以独立论文仓根目录为 CWD 基准。
# YAML 中的相对路径（datasets/、Exp/ 等）以 OFR_ROOT 为基准。
OFR_ROOT = os.environ.get('OFR_ROOT')
if OFR_ROOT is None:
    OFR_ROOT = ROOT_PATH

# CWD 切到项目根目录，使 YAML 中的相对路径正确解析
os.chdir(OFR_ROOT)

# 向后兼容：WORKSPACE_PATH 仍指向 OFR_ROOT
WORKSPACE_PATH = OFR_ROOT

# Provide backward compatibility for deprecated torchvision tensor transforms.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*torchvision.transforms.functional_tensor.*')
    try:  # pragma: no cover - only hits on old torchvision
        import torchvision.transforms.functional_tensor  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover
        from torchvision.transforms import functional as _tv_functional

        functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
        functional_tensor.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
        sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor

import basicofr.archs  # noqa: F401,E402
import basicofr.data  # noqa: F401,E402
import basicofr.losses  # noqa: F401,E402
import basicofr.models  # noqa: F401,E402

from basicsr.test import test_pipeline  # noqa: E402


if __name__ == '__main__':
    test_pipeline(WORKSPACE_PATH)
