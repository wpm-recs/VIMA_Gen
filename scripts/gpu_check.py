#!/usr/bin/env python3
"""
GPU 自检：确认 torch / kornia / pybullet 渲染是否真的跑在 GPU 上。

用法:
  python scripts/gpu_check.py                  # auto：有 CUDA 就用 GPU
  python scripts/gpu_check.py --device cpu     # 强制 CPU
  python scripts/gpu_check.py --require-cuda   # CUDA 不可用时以退出码 1 结束

退出码: 0 正常 / 1 自检失败（--require-cuda 且无 CUDA，或某项检查抛错）
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# 保证能 import 到 vima_bench（脚本位于 scripts/ 下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import vima_bench
from vima_bench.tasks.utils import misc_utils as utils
from vima_bench.tasks.utils.device import pybullet_renderer


def bench_warp_affine(device: torch.device, iters: int = 50, n: int = 32, size: int = 128) -> float:
    """单次 kornia warp_affine 的平均耗时（毫秒）。"""
    from kornia.geometry.transform import warp_affine

    x = torch.rand(n, 3, size, size, device=device)
    M = torch.eye(2, 3, device=device).unsqueeze(0).repeat(n, 1, 1)
    for _ in range(5):  # warmup
        warp_affine(x, M, dsize=(size, size))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        warp_affine(x, M, dsize=(size, size))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def check_image_rotator() -> None:
    """确认 ImageRotator 的输出落在配置的设备上。"""
    device = vima_bench.get_device()
    rotator = utils.ImageRotator(4, device=device)
    xs = [torch.rand(3, 64, 64) for _ in range(4)]  # 故意给 CPU 张量
    out = rotator(xs, pivot=(32, 32))
    print(f"  ImageRotator: in=cpu -> out device={out[0].device} shape={tuple(out[0].shape)}")
    if out[0].device != device:
        raise AssertionError(f"ImageRotator 输出在 {out[0].device}，期望 {device}")


def check_pybullet_render() -> None:
    """确认 pybullet 渲染可用，并报告所用渲染器。"""
    import numpy as np
    import pybullet as p

    renderer = pybullet_renderer()
    name = {
        p.ER_TINY_RENDERER: "ER_TINY_RENDERER (CPU 光栅化)",
        p.ER_BULLET_HARDWARE_OPENGL: "ER_BULLET_HARDWARE_OPENGL (GPU/EGL OpenGL)",
    }.get(renderer, str(renderer))
    print(f"  pybullet renderer: {name}")

    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "vima_bench", "tasks", "assets")
        )
        p.loadURDF("plane/plane.urdf", physicsClientId=cid)
        view = p.computeViewMatrix([0.5, 0, 1.0], [0.5, 0, 0], [0, 0, 1], physicsClientId=cid)
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10, physicsClientId=cid)
        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            p.getCameraImage(256, 256, view, proj, renderer=renderer, physicsClientId=cid)
        dt = (time.perf_counter() - t0) / iters * 1000.0
        print(f"  256x256 渲染: {dt:.2f} ms/帧")
    finally:
        p.disconnect(cid)


def main() -> int:
    parser = argparse.ArgumentParser(description="VIMA-Bench GPU 自检")
    parser.add_argument("--device", type=str, default=None,
                        help="auto（默认）/ cpu / cuda / cuda:0，也可用 VIMA_DEVICE")
    parser.add_argument("--require-cuda", action="store_true",
                        help="CUDA 不可用时以退出码 1 结束")
    args = parser.parse_args()

    try:
        device = vima_bench.configure_device(args.device, verbose=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"设备配置失败: {exc}")
        return 1

    if args.require_cuda and device.type != "cuda":
        print("错误: --require-cuda 要求 CUDA，但当前设备是 CPU。")
        return 1

    failed = False
    print("[1/3] ImageRotator 设备检查")
    try:
        check_image_rotator()
    except Exception as exc:
        print(f"  失败: {type(exc).__name__}: {exc}")
        failed = True

    print("[2/3] kornia warp_affine 吞吐（32x3x128x128）")
    try:
        cpu_ms = bench_warp_affine(torch.device("cpu"))
        print(f"  CPU: {cpu_ms:.3f} ms/次")
        if torch.cuda.is_available():
            gpu_ms = bench_warp_affine(torch.device("cuda:0"))
            print(f"  GPU: {gpu_ms:.3f} ms/次  (加速比 {cpu_ms / gpu_ms:.2f}x)")
            if gpu_ms > cpu_ms:
                print("  注意: 该尺寸下 GPU 因 kernel 启动开销反而更慢；批量更大时 GPU 才占优。")
    except Exception as exc:
        print(f"  失败: {type(exc).__name__}: {exc}")
        failed = True

    print("[3/3] pybullet 渲染检查")
    try:
        check_pybullet_render()
    except Exception as exc:
        print(f"  失败: {type(exc).__name__}: {exc}")
        failed = True

    print("\n结论:", "自检失败" if failed else "自检通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
