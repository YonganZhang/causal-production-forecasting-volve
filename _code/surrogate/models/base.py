"""模型插件协议。加一个模型 = 加一个文件, 实现这三个方法。"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Surrogate(Protocol):
    name: str

    def fit(self, X: np.ndarray, Yf: np.ndarray, Yo: np.ndarray) -> None:
        """X (n,13) float32; Yf (n,8,2,44431) float16; Yo (n,2640) float32"""

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (场 (n,8,2,44431) float32, 井观测 (n,2640) float32)"""

    def describe(self) -> dict:
        """超参、参数量、训练耗时等, 落盘用"""
