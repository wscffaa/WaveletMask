"""模型模块

包含训练循环实现：
- rtn_model: RTN GAN 训练模型
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import warnings
from os import path as osp
from pkgutil import extend_path

from basicsr.utils import scandir

# 支持从扩展目录注入自定义 model（见 basicofr.__init__ 的说明）
__path__ = extend_path(__path__, __name__)

_FRAMEWORK_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))
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
                raise ImportError(f'无法为项目 model 包创建导入规范: {module_name}')
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
    module_name = project_name.replace('-', '_')
    if module_name and module_name[0].isdigit():
        module_name = f'p{module_name}'
    return f'{_PROJECTS_PACKAGE}.{module_name}'


def _register_project_package(project_name: str, models_dir: str) -> str:
    _ensure_namespace_package(_PROJECTS_PACKAGE)
    package_name = _project_package_name(project_name)
    _ensure_namespace_package(package_name, models_dir)
    return package_name


def _iter_method_model_dirs() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen_dirs: set[str] = set()

    if osp.isdir(_REPO_ROOT):
        for project_name in sorted(os.listdir(_REPO_ROOT)):
            if project_name in _SKIP_SIBLING_NAMES or project_name.startswith(('.', '_')):
                continue
            models_dir = osp.join(_REPO_ROOT, project_name, 'models')
            if not osp.isdir(models_dir):
                continue
            abs_dir = osp.abspath(models_dir)
            if abs_dir in seen_dirs:
                continue
            seen_dirs.add(abs_dir)
            found.append((project_name, models_dir))

    if osp.isdir(_PROJECTS_DIR):
        for project_name in sorted(os.listdir(_PROJECTS_DIR)):
            if project_name.startswith(('_', '.')):
                continue
            models_dir = osp.join(_PROJECTS_DIR, project_name, 'models')
            if not osp.isdir(models_dir):
                continue
            abs_dir = osp.abspath(models_dir)
            if abs_dir in seen_dirs:
                continue
            seen_dirs.add(abs_dir)
            found.append((project_name, models_dir))

    return found


def _collect_model_entries() -> list[tuple[str, str | None]]:
    entries: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()

    for model_folder in list(__path__):
        if not osp.isdir(model_folder):
            continue
        for v in scandir(model_folder):
            if v.endswith('_model.py'):
                name = osp.splitext(osp.basename(v))[0]
                key = (name, None)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(key)

    for project_name, models_dir in _iter_method_model_dirs():
        package_name = _register_project_package(project_name, models_dir)
        for v in scandir(models_dir):
            if not v.endswith('_model.py'):
                continue
            name = osp.splitext(osp.basename(v))[0]
            key = (name, package_name)
            if key in seen:
                continue
            seen.add(key)
            entries.append(key)

    return entries


def _import_model_module(file_name: str, owner_name: str | None):
    if owner_name is None:
        module_path = f'basicofr.models.{file_name}'
        try:
            return importlib.import_module(module_path)
        except Exception as exc:
            warnings.warn(f'跳过加载 {module_path}: {exc}')
            return None

    module_path = f'{owner_name}.{file_name}'
    module = sys.modules.get(module_path)
    if module is not None:
        return module

    parent_pkg = sys.modules.get(owner_name)
    package_paths = getattr(parent_pkg, '__path__', None)
    if not package_paths:
        warnings.warn(f'跳过加载 {module_path}: 项目包未注册或缺少 __path__')
        return None

    file_path = osp.join(package_paths[0], f'{file_name}.py')
    try:
        spec = importlib.util.spec_from_file_location(module_path, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'无法为模型模块创建导入规范: {file_path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        sys.modules.pop(module_path, None)
        warnings.warn(f'跳过加载 {module_path}: {exc}')
        return None


_model_modules = []
for _file_name, _owner_name in _collect_model_entries():
    _module = _import_model_module(_file_name, _owner_name)
    if _module is not None:
        _model_modules.append(_module)
