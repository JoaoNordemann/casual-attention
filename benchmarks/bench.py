"""Compare forward + backward in isolated processes: python -m benchmarks.bench."""
import argparse
import json
import platform
import subprocess
import sys
import time

import torch

from src.attention import CausalSelfAttention


def worker(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    model = CausalSelfAttention(
        args.d_model, args.heads, args.seq_len, use_flash=args.worker == "sdpa"
    ).to(device).train()
    x = torch.randn(args.batch, args.seq_len, args.d_model, device=device, requires_grad=True)

    def step():
        model.zero_grad(set_to_none=True)
        x.grad = None
        model(x).square().mean().backward()

    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**20
    elif sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        peak = counters.PeakWorkingSetSize / 2**20
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak /= 2**20 if sys.platform == "darwin" else 1024
    print(json.dumps({"tokens_s": args.batch * args.seq_len * args.steps / elapsed, "peak_mib": peak}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--worker", choices=["manual", "sdpa"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("batch", "seq_len", "d_model", "heads", "threads", "warmup", "steps"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.d_model % args.heads:
        parser.error("d-model must be divisible by heads")
    if args.worker:
        worker(args)
        return
    print(f"{platform.platform()}; Python {platform.python_version()}; PyTorch {torch.__version__}")
    print(f"Device: {args.device}; float32; B={args.batch}; T={args.seq_len}; "
          f"d_model={args.d_model}; heads={args.heads}; threads={args.threads}; "
          f"warmup={args.warmup}; steps={args.steps}; forward+backward")
    print("| Implementação | Tokens/s | Pico de memória (MiB) |")
    print("| --- | ---: | ---: |")
    for mode in ("manual", "sdpa"):
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.bench", *sys.argv[1:], "--worker", mode],
            check=True, capture_output=True, text=True,
        )
        data = json.loads(result.stdout)
        print(f"| {mode} | {data['tokens_s']:.0f} | {data['peak_mib']:.1f} |", flush=True)


if __name__ == "__main__":
    main()
