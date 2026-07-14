"""架构模块

包含：
- RTN 主架构
- RRTN / MambaOFR 基线架构
- projects/ 独立项目目录下的实验架构
- 空间修复模块（Swin / Mamba）
- 光流估计（RAFT/SpyNet）
- 通用组件（门控、判别器等）
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from os import path as osp
from pkgutil import extend_path

from basicsr.utils import scandir

# 允许通过在 sys.path 中追加"扩展根目录"来为 basicofr.archs 注入实验架构
__path__ = extend_path(__path__, __name__)

# ===================== 项目目录路径解析（双模式）=====================
# 1) 仓根一级方法包：<repo>/<method>/archs/*_arch.py（论文自包含仓默认）
# 2) OFR_PROJECTS_DIR 或 <repo>/projects/<NNN>/archs（FreqDelta/旧部署兼容）
_FRAMEWORK_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))  # parent of basicofr/
_REPO_ROOT = (
    osp.dirname(_FRAMEWORK_ROOT)
    if osp.basename(_FRAMEWORK_ROOT) == 'framework'
    else _FRAMEWORK_ROOT
)
_DEFAULT_PROJECTS_DIR = osp.join(_REPO_ROOT, 'projects')
_PROJECTS_DIR = osp.abspath(os.environ.get('OFR_PROJECTS_DIR', _DEFAULT_PROJECTS_DIR))
_PROJECTS_PACKAGE = f'{__name__}._projects'
_SKIP_SIBLING_NAMES = frozenset({
    'basicofr', 'basicsr', 'options', 'scripts', 'docs', 'tests', 'test',
    'Idea', 'Exp', 'Paper', 'pretrained', 'datasets', 'template', 'experiments',
    'framework', 'projects', 'tb_logger', 'results',
})
# ====================================================================


def _ensure_namespace_package(module_name: str, package_path: str | None = None) -> types.ModuleType:
    """确保动态命名空间包已注册。"""
    module = sys.modules.get(module_name)
    if module is None:
        if package_path is None:
            module = types.ModuleType(module_name)
            module.__file__ = __file__
            module.__package__ = module_name
            spec = importlib.machinery.ModuleSpec(module_name, loader=None, is_package=True)
            spec.origin = __file__
            spec.submodule_search_locations = []
            module.__loader__ = None
            module.__spec__ = spec
            module.__path__ = spec.submodule_search_locations
        else:
            package_init = osp.join(package_path, '__init__.py')
            spec = importlib.util.spec_from_file_location(
                module_name,
                package_init,
                submodule_search_locations=[package_path],
            )
            if spec is None:
                raise ImportError(f'无法为项目包创建导入规范: {module_name}')
            module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

    if package_path is not None and package_path not in module.__path__:
        module.__path__.append(package_path)
        if module.__spec__ is not None and module.__spec__.submodule_search_locations is not None:
            module.__spec__.submodule_search_locations.append(package_path)
        if not getattr(module, '__file__', None):
            module.__file__ = osp.join(package_path, '__init__.py')
        if module.__spec__ is not None and not module.__spec__.origin:
            module.__spec__.origin = module.__file__

    parent_name, _, child_name = module_name.rpartition('.')
    parent = sys.modules.get(parent_name)
    if parent is not None and not hasattr(parent, child_name):
        setattr(parent, child_name, module)

    return module


def _project_package_name(project_name: str) -> str:
    """将项目目录名转换为稳定的模块名。"""
    module_name = project_name.replace('-', '_')
    if module_name and module_name[0].isdigit():
        module_name = f'p{module_name}'
    return f'{_PROJECTS_PACKAGE}.{module_name}'


def _register_project_package(project_name: str, archs_dir: str) -> str:
    """为方法包 archs 注册独立命名空间，支持相对导入与同名 components 隔离。"""
    _ensure_namespace_package(_PROJECTS_PACKAGE)
    package_name = _project_package_name(project_name)
    _ensure_namespace_package(package_name, archs_dir)
    return package_name


def _iter_method_arch_dirs() -> list[tuple[str, str]]:
    """返回 (project_name, archs_dir)。sibling 模式优先，再 projects/ 兼容模式。"""
    found: list[tuple[str, str]] = []
    seen_dirs: set[str] = set()

    if osp.isdir(_REPO_ROOT):
        for project_name in sorted(os.listdir(_REPO_ROOT)):
            if project_name in _SKIP_SIBLING_NAMES or project_name.startswith(('.', '_')):
                continue
            archs_dir = osp.join(_REPO_ROOT, project_name, 'archs')
            if not osp.isdir(archs_dir):
                continue
            abs_dir = osp.abspath(archs_dir)
            if abs_dir in seen_dirs:
                continue
            seen_dirs.add(abs_dir)
            found.append((project_name, archs_dir))

    if osp.isdir(_PROJECTS_DIR):
        for project_name in sorted(os.listdir(_PROJECTS_DIR)):
            if project_name.startswith(('_', '.')):
                continue
            archs_dir = osp.join(_PROJECTS_DIR, project_name, 'archs')
            if not osp.isdir(archs_dir):
                continue
            abs_dir = osp.abspath(archs_dir)
            if abs_dir in seen_dirs:
                continue
            seen_dirs.add(abs_dir)
            found.append((project_name, archs_dir))

    return found


def _collect_arch_filenames() -> list[tuple[str, str | None, str | None]]:
    """收集所有 *_arch.py。

    扫描顺序:
      1. 本目录根下的 *_arch.py（基线架构）
      2. 仓根一级 <method>/archs/（论文自包含仓）
      3. OFR_PROJECTS_DIR 或 <repo>/projects/<NNN>/archs/（旧部署兼容）

    Returns:
        list of (module_name, owner_name, source):
            owner_name 为 None 表示根目录基线架构
            source='project' 时 owner_name 为动态项目包名
            source: 'root' | 'project'
    """
    results: list[tuple[str, str | None, str | None]] = []
    seen: set[tuple[str, str | None, str]] = set()

    # 1. 扫描根目录下的 *_arch.py（基线架构）
    for arch_folder in list(__path__):
        if not osp.isdir(arch_folder):
            continue
        for v in scandir(arch_folder):
            if v.endswith('_arch.py'):
                name = osp.splitext(osp.basename(v))[0]
                key = ('root', None, name)
                if key not in seen:
                    seen.add(key)
                    results.append((name, None, 'root'))

    # 2/3. sibling method packages + projects/ compatibility
    for project_name, archs_dir in _iter_method_arch_dirs():
        package_name = _register_project_package(project_name, archs_dir)
        for v in scandir(archs_dir):
            if v.endswith('_arch.py'):
                name = osp.splitext(osp.basename(v))[0]
                key = ('project', package_name, name)
                if key not in seen:
                    seen.add(key)
                    results.append((name, package_name, 'project'))

    return results


def _import_arch_module(file_name: str, owner_name: str | None, source: str):
    """按来源导入架构模块。"""
    if source == 'root':
        module_path = f'basicofr.archs.{file_name}'
        try:
            return importlib.import_module(module_path)
        except Exception as e:
            import warnings
            warnings.warn(f'跳过加载 {module_path}: {e}')
            return None
    if source != 'project':
        return None

    module_path = f'{owner_name}.{file_name}'
    module = sys.modules.get(module_path)
    if module is not None:
        return module

    parent_pkg = sys.modules.get(owner_name)
    package_paths = getattr(parent_pkg, '__path__', None)
    if not package_paths:
        import warnings
        warnings.warn(f'跳过加载 {module_path}: 项目包未注册或缺少 __path__')
        return None

    file_path = osp.join(package_paths[0], f'{file_name}.py')

    try:
        spec = importlib.util.spec_from_file_location(module_path, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'无法为架构模块创建导入规范: {file_path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        sys.modules.pop(module_path, None)
        import warnings
        warnings.warn(f'跳过加载 {module_path}: {e}')
        return None


_arch_modules = []
for _fn, _in, _src in _collect_arch_filenames():
    _mod = _import_arch_module(_fn, _in, _src)
    if _mod is not None:
        _arch_modules.append(_mod)

# 导入 components 模块（包含判别器等）
from . import components  # noqa: F401
