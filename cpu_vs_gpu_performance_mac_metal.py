### As part of learning local LLMs, this code is supposed to demonstrate how much faster running LLM/NN workloads on GPU can be compared to CPU.
## This specifically is designed for Mac running Apple Silicon (M1+). The result can be about X% faster on GPU.
## Before running the below, recommend to set up a new python environment:

## conda create -n torch_bench python=3.11 pytorch torchaudio -c pytorch -c conda-forge -y && \
## conda activate torch_bench && \
## conda install -c conda-forge transformers datasets -y



##  Load packages
import os, time, platform, random
import torch
import torch.nn as nn

## Defining functions, mostly for formatting

def fmt_sec(s: float) -> str:
    return f"{s * 1000:.2f} ms"

def sync_device(dev: torch.device | str):
    """Synchronize CUDA or MPS to get accurate timings."""
    d = torch.device(dev) if isinstance(dev, str) else dev
    if d.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif d.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()

def timeit(func, warmup: int = 3, repeat: int = 5, sync: bool = True, label: str = "") -> float:
    for _ in range(max(0, warmup)):
        func()
        if sync:  # make sure kernels finish before timing/looping
            sync_device(_infer_current_device())

    times = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        func()
        if sync:
            sync_device(_infer_current_device())
        end = time.perf_counter()
        times.append(end - start)

    times.sort()
    avg = sum(times) / len(times)
    p90_index = max(0, int(0.9 * (len(times) - 1)))
    p90 = times[p90_index]
    print(f"{label:<55} avg={fmt_sec(avg)}   p90={fmt_sec(p90)}   runs={len(times)}")
    return avg

# ------------------------------------------------------------
# Device helpers
# ------------------------------------------------------------
def device_name(dev: torch.device | str) -> str:
    d = torch.device(dev) if isinstance(dev, str) else dev
    if d.type == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA device"
    if d.type == "mps":
        # Apple Silicon
        chip = platform.machine()
        return f"Apple GPU (MPS on {chip})"
    # CPU
    return f"CPU ({platform.processor() or 'generic'})"

def pick_devices():
    """Return a list of devices to test, in order."""
    devices = ["cpu"]
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices

def _infer_current_device():
    # Best effort: if CUDA available and a tensor was created there, synchronize via CUDA;
    # for our controlled calls we sync based on availability priority (CUDA > MPS > CPU).
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ------------------------------------------------------------
# Bench kernels
# ------------------------------------------------------------
def matmul_bench(dev: torch.device | str, M=4096, K=4096, N=4096, dtype=torch.float32):
    d = torch.device(dev)
    A = torch.randn(M, K, device=d, dtype=dtype)
    B = torch.randn(K, N, device=d, dtype=dtype)
    def run():
        return A @ B
    return run

def conv2d_bench(dev: torch.device | str, batch=32, cin=64, cout=128, H=128, W=128, k=3, dtype=torch.float32):
    d = torch.device(dev)
    x = torch.randn(batch, cin, H, W, device=d, dtype=dtype)
    conv = nn.Conv2d(cin, cout, kernel_size=k, stride=1, padding=k // 2, bias=False).to(d, dtype=dtype)
    def run():
        return conv(x)
    return run

def hf_bench(dev: torch.device | str, dtype=torch.float32, batch=16, seq=128):
    """
    Returns (callable_bench, err_msg).
    callable_bench() runs a single forward pass of DistilBERT on dummy text.
    If err_msg is not None, the demo should be considered skipped and the reason shown.
    """
    d = torch.device(dev)
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        return None, f"Transformers not installed (ImportError): {e}"
    except Exception as e:
        return None, f"Transformers import failed: {type(e).__name__}: {e}"

    try:
        model_id = "distilbert-base-uncased-finetuned-sst-2-english"
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.to(d)  # keep FP32 for reliability on MPS

        # Only cast to half on CUDA for reliability.
        use_half = (d.type == "cuda" and dtype == torch.float16)
        if use_half:
            model.half()

        model.eval()

        texts = [
            "this movie was surprisingly good!" if i % 2 == 0 else "I did not like this at all."
            for i in range(batch)
        ]
        batch_inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=seq)
        batch_inputs = {k: v.to(d) for k, v in batch_inputs.items()}

        @torch.inference_mode()
        def bench():
            _ = model(**batch_inputs).logits

        return bench, None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ------------------------------------------------------------
# Suite runner
# ------------------------------------------------------------
def run_suite(dev: torch.device | str, allow_fp16_on_cuda: bool = True, allow_fp16_on_mps: bool = False):
    torch.manual_seed(0)
    random.seed(0)
    d = torch.device(dev)

    print("\n" + "=" * 90)
    print(f"Running on: {device_name(d)}")
    print("=" * 90)

    # Dtype policy
    if d.type == "cuda" and allow_fp16_on_cuda:
        dtype = torch.float16
        torch.backends.cudnn.benchmark = True
    elif d.type == "mps" and allow_fp16_on_mps:
        # You can try FP16 on MPS by setting allow_fp16_on_mps=True.
        # Defaults to FP32 for broad reliability across ops.
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 1) Big GEMM
    mm = matmul_bench(d, M=4096, K=4096, N=4096, dtype=dtype)
    timeit(mm, label=f"[MatMul 4096x4096 FP{16 if dtype==torch.float16 else 32} on {d.type}]")

    # 2) Conv2D
    conv = conv2d_bench(d, batch=32, cin=64, cout=128, H=128, W=128, k=3, dtype=dtype)
    timeit(conv, label=f"[Conv2D 32x64x128x128 FP{16 if dtype==torch.float16 else 32} on {d.type}]")

    # 3) Tiny Transformer forward (optional)
    hf, hf_err = hf_bench(d, dtype=dtype, batch=16, seq=128)
    if hf is not None:
        timeit(hf, label=f"[HF DistilBERT batch=16 seq=128 FP{16 if dtype==torch.float16 else 32} on {d.type}]")
    else:
        print(f"[HF demo skipped] {hf_err or 'unknown reason'}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    # Always run CPU baseline
    run_suite("cpu", allow_fp16_on_cuda=False, allow_fp16_on_mps=False)

    # Run MPS if present (Apple GPU)
    if torch.backends.mps.is_available():
        run_suite("mps", allow_fp16_on_cuda=False, allow_fp16_on_mps=False)
    else:
        print("\nNo MPS device detected. (If on Apple Silicon, update macOS and PyTorch.)")

    # Run CUDA if present
    if torch.cuda.is_available():
        # Warm up then run
        torch.cuda.synchronize()
        run_suite("cuda", allow_fp16_on_cuda=True, allow_fp16_on_mps=False)

    print("\nTip: On CUDA, FP16 is common and often increases speedup vs CPU FP32.")
    print("On MPS, start with FP32 for reliability; you can toggle FP16 in run_suite if curious.")

if __name__ == "__main__":
    main()
