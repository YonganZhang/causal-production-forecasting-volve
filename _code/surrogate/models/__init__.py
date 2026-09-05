"""模型注册表。加模型 = 在本目录放一个 .py 并用 @register("名字") 装饰类。"""
import importlib
import pkgutil
from pathlib import Path

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def _discover():
    """自动发现本目录下所有模块 —— 旧版是硬编码 import, 新增文件会 KeyError。"""
    for m in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if m.name.startswith("_") or m.name == "base":
            continue
        try:
            importlib.import_module(f"{__name__}.{m.name}")
        except Exception as e:                      # 坏插件不许拖垮整个流水线
            print(f"  ⚠️ 插件 {m.name} 导入失败: {type(e).__name__}: {e}")


def get(name):
    _discover()
    if name not in REGISTRY:
        raise KeyError(f"未注册的模型 {name}; 已注册: {sorted(REGISTRY)}")
    return REGISTRY[name]


def available():
    _discover()
    return sorted(REGISTRY)
