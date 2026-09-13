"""
Device / GPU configuration for VIMA-Bench.

The VIMA-Bench runtime is mostly CPU-bound (pybullet physics + numpy/cv2 image
processing), but two things can use the GPU:

1. ``torch`` / ``kornia`` tensor utilities (e.g.
   :class:`vima_bench.tasks.utils.misc_utils.ImageRotator`).
2. pybullet's hardware OpenGL camera renderer (``ER_BULLET_HARDWARE_OPENGL``),
   which is GPU/EGL-accelerated when the platform exposes a GL device.

This module centralises the choice so that every entry point resolves the
device in exactly the same way.

Resolution order (highest priority first):

1. explicit ``spec`` argument
2. the ``VIMA_DEVICE`` environment variable
3. automatic: ``cuda:0`` when CUDA is available, otherwise ``cpu``

Accepted ``spec`` values: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:1"``,
``"gpu"``, ``0`` (index), or a :class:`torch.device`.

Environment variables
---------------------
``VIMA_DEVICE``
    Default device spec used when no explicit spec is passed.
``VIMA_TORCH_DEFAULT_DEVICE``
    Set to ``1``/``true`` to additionally call ``torch.set_default_device()``.
    This hijacks *every* ``torch`` allocation, including ones later converted
    with ``.numpy()``, so it is opt-in only.
``VIMA_RENDERER``
    ``hardware`` (default, GPU/EGL OpenGL) or ``tiny`` (pybullet's built-in CPU
    rasteriser). See :func:`pybullet_renderer`.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import torch

__all__ = [
    "DEVICE_ENV_VAR",
    "RENDERER_ENV_VAR",
    "cuda_available",
    "resolve_device",
    "configure_device",
    "get_device",
    "to_device",
    "describe",
    "pybullet_renderer",
]

DEVICE_ENV_VAR = "VIMA_DEVICE"
RENDERER_ENV_VAR = "VIMA_RENDERER"
_DEFAULT_DEVICE_ENV_VAR = "VIMA_TORCH_DEFAULT_DEVICE"

_CURRENT_DEVICE: Optional[torch.device] = None

DeviceSpec = Optional[Union[str, int, torch.device]]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def cuda_available() -> bool:
    """Return whether a usable CUDA device is present (never raises)."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def _auto_device() -> torch.device:
    return torch.device("cuda:0") if cuda_available() else torch.device("cpu")


def resolve_device(spec: DeviceSpec = None) -> torch.device:
    """Resolve ``spec`` into a concrete :class:`torch.device`.

    Falls back to CPU when a CUDA device is requested but unavailable, so the
    pipeline keeps working on GPU-less machines.
    """
    if spec is None:
        spec = os.environ.get(DEVICE_ENV_VAR) or None

    if spec is None:
        return _auto_device()

    if isinstance(spec, torch.device):
        device = spec
    elif isinstance(spec, int):
        device = torch.device("cuda", spec)
    else:
        text = str(spec).strip().lower()
        if text in ("", "auto", "default"):
            return _auto_device()
        if text == "cpu":
            device = torch.device("cpu")
        elif text in ("cuda", "gpu"):
            device = torch.device("cuda:0")
        elif text.isdigit():
            device = torch.device("cuda", int(text))
        else:
            device = torch.device(text)

    if device.type == "cuda" and not cuda_available():
        return torch.device("cpu")
    return device


def configure_device(spec: DeviceSpec = None, verbose: bool = True) -> torch.device:
    """Resolve the device and apply process-wide torch settings.

    Safe to call multiple times and from worker processes. Returns the resolved
    device, which is also cached for :func:`get_device`.
    """
    device = resolve_device(spec)

    if device.type == "cuda":
        try:
            torch.cuda.set_device(device)
        except Exception:  # pragma: no cover - defensive
            pass
        # Fixed input shapes in this codebase -> let cuDNN autotune kernels.
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:  # pragma: no cover
            pass
        # TF32 / "high" precision is a good default for the small convolution
        # style workloads used by kornia transforms.
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:  # pragma: no cover
            pass

    if os.environ.get(_DEFAULT_DEVICE_ENV_VAR, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        try:
            torch.set_default_device(device)
        except Exception:  # pragma: no cover - torch < 2.0
            pass

    global _CURRENT_DEVICE
    _CURRENT_DEVICE = device

    if verbose:
        print(f"[VIMA][device] {describe(device)}")
    return device


def get_device() -> torch.device:
    """Return the process device, resolving lazily on first use."""
    global _CURRENT_DEVICE
    if _CURRENT_DEVICE is None:
        _CURRENT_DEVICE = resolve_device()
    return _CURRENT_DEVICE


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def to_device(obj: Any, device: DeviceSpec = None, non_blocking: bool = True) -> Any:
    """Recursively move tensors inside nested dict/list/tuple structures.

    Non-tensor leaves are returned unchanged, so numpy arrays and strings pass
    through untouched.
    """
    target = torch.device(device) if device is not None else get_device()
    if torch.is_tensor(obj):
        return obj.to(target, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: to_device(v, target, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, target, non_blocking) for v in obj)
    return obj


def describe(device: DeviceSpec = None) -> str:
    """Human-readable one-line summary of the runtime device."""
    device = torch.device(device) if device is not None else get_device()
    parts = [
        f"torch={torch.__version__}",
        f"cuda_build={torch.version.cuda}",
        f"cuda_available={cuda_available()}",
        f"device={device}",
    ]
    if device.type == "cuda":
        try:
            index = device.index if device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            total = getattr(props, "total_memory", 0)
            parts += [
                f"name={props.name!r}",
                f"cc={props.major}.{props.minor}",
                f"mem={round(total / 2**30, 1)}GiB",
            ]
        except Exception:  # pragma: no cover - defensive
            pass
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# pybullet rendering
# ---------------------------------------------------------------------------


def pybullet_renderer(preference: Optional[str] = None) -> int:
    """Return the pybullet renderer constant to use for camera rendering.

    ``hardware`` (default) selects ``ER_BULLET_HARDWARE_OPENGL``, which is
    GPU/EGL-accelerated when the platform provides a GL device and silently
    falls back to a software rasteriser otherwise. ``tiny`` selects pybullet's
    built-in CPU rasteriser (``ER_TINY_RENDERER``), useful for debugging or for
    environments without any GL stack.
    """
    import pybullet as p

    pref = (preference or os.environ.get(RENDERER_ENV_VAR, "hardware"))
    pref = pref.strip().lower()
    if pref in ("tiny", "cpu", "software"):
        return p.ER_TINY_RENDERER
    return p.ER_BULLET_HARDWARE_OPENGL
