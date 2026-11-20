"""Device resolution and Intel Extension for PyTorch (IPEX / XPU) integration."""

from typing import Any, Dict, Optional, Tuple, Union
import warnings
import torch
import torch.nn as nn

# Suppress noisy pkg_resources deprecation warnings emitted by IPEX on import
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="intel_extension_for_pytorch")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore

        HAS_IPEX = True
        IPEX_VERSION = getattr(ipex, "__version__", "unknown")
    except ImportError:
        ipex = None  # type: ignore
        HAS_IPEX = False
        IPEX_VERSION = None


def is_xpu_available() -> bool:
    """Return True if Intel XPU backend is present and an XPU device is available."""
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def resolve_device(requested: Union[str, torch.device] = "auto") -> torch.device:
    """
    Resolve compute device with priority:
      1. Intel XPU ('xpu') if available
      2. NVIDIA CUDA ('cuda') if available
      3. CPU ('cpu')
    """
    if isinstance(requested, torch.device):
        return requested
    req = requested.strip().lower()
    if req == "auto":
        if is_xpu_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if req.startswith("xpu") and not is_xpu_available():
        warnings.warn("Requested 'xpu' device, but torch.xpu.is_available() is False. Falling back to 'cpu'.")
        return torch.device("cpu")
    return torch.device(req)


def get_device_summary(device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
    """Return diagnostic details about the active PyTorch / IPEX / XPU setup."""
    dev = resolve_device(device if device is not None else "auto")
    xpu_avail = is_xpu_available()
    xpu_name = None
    if xpu_avail:
        try:
            idx = dev.index if (dev.type == "xpu" and dev.index is not None) else 0
            xpu_name = torch.xpu.get_device_name(idx)
        except Exception:
            xpu_name = "Intel XPU Device"

    return {
        "active_device": str(dev),
        "torch_version": torch.__version__,
        "ipex_installed": HAS_IPEX,
        "ipex_version": IPEX_VERSION,
        "xpu_available": xpu_avail,
        "xpu_device_name": xpu_name,
    }


def optimize_with_ipex(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer]]:
    """
    Apply `ipex.optimize` to `model` (and `optimizer` if provided) when IPEX is installed.
    Safely falls back to the unmodified model/optimizer if IPEX is not installed or raises.
    """
    if not HAS_IPEX or ipex is None:
        return model, optimizer

    try:
        if optimizer is not None:
            model.train()
            opt_model, opt_optimizer = ipex.optimize(
                model, optimizer=optimizer, dtype=dtype
            )
            return opt_model, opt_optimizer
        else:
            model.eval()
            opt_model = ipex.optimize(model, dtype=dtype)
            return opt_model, None
    except Exception as exc:
        warnings.warn(f"ipex.optimize skipped ({exc}); using standard eager model.")
        return model, optimizer


def synchronize_device(device: Union[str, torch.device]) -> None:
    """Synchronize asynchronous device queues (XPU or CUDA) for accurate timing."""
    dev = resolve_device(device)
    if dev.type == "xpu" and is_xpu_available():
        torch.xpu.synchronize(dev)
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)
