### As part of learning local LLMs, this code is supposed to demonstrate how much faster running LLM/NN workloads on GPU can be compared to CPU.
## This specifically is designed to use CUDA, the framework for NVIDIA GPUs. The result can be about 99% faster on GPU.
## Before running the below, recommend to set up a new python environment:

# conda create -n torch_bench python=3.11 -y
# conda activate torch_bench
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# conda install -c conda-forge transformers


##### Load packages
import os, time, platform, random
import torch
import torch.nn as nn

### Create functions primarily for formatting


# set up time formatting for measurement
def fmt_sec(s: float) -> str:
    return f"{s * 1000:.2f} ms"


## sets up function to time benchmarking of function

def timeit(func, warmup: int = 3, repeat: int = 5, sync_cuda: bool = True, label: str = "") -> float:
    # Warmup
    for _ in range(max(0, warmup)):
        func()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
    # Timed runs
    times = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        func()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    times.sort()
    avg = sum(times) / len(times)
    p90_index = max(0, int(0.9 * (len(times) - 1)))
    p90 = times[p90_index]
    print(f"{label:<45} avg={fmt_sec(avg)}   p90={fmt_sec(p90)}   runs={len(times)}")
    return avg


## Determine what is the name of the hardware you are benchmarking
def device_name(dev: torch.device | str) -> str:
    if isinstance(dev, str):
        dev = torch.device(dev)
    if dev.type == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA device"
    return f"CPU ({platform.processor() or 'generic'})"

## Set up benchmarks

## Benchmark 1 -- large matrix multiplication

def matmul_bench(dev: torch.device | str, M=4096, K=4096, N=4096, dtype=torch.float32):
    dev = torch.device(dev)
    A = torch.randn(M, K, device=dev, dtype=dtype)
    B = torch.randn(K, N, device=dev, dtype=dtype)
    def run():
        return A @ B
    return run


## Benchmark 2 -- convolution 
def conv2d_bench(dev: torch.device | str, batch=32, cin=64, cout=128, H=128, W=128, k=3, dtype=torch.float32):
    dev = torch.device(dev)
    x = torch.randn(batch, cin, H, W, device=dev, dtype=dtype)
    conv = nn.Conv2d(cin, cout, kernel_size=k, stride=1, padding=k // 2, bias=False).to(dev, dtype=dtype)
    def run():
        return conv(x)
    return run

## Benchmark 3 — Demo huggingface (pre-trained LLM models). Uses DistilBERT, a small transformer model on dummy text.

def hf_bench(dev: torch.device | str, dtype=torch.float32, batch=16, seq=128):
    """
    Returns (callable_bench, err_msg).
    callable_bench() runs a single forward pass of DistilBERT on dummy text.
    If err_msg is not None, the demo should be considered skipped and the reason shown.
    """
    dev = torch.device(dev)

    # Import only fails the demo if it is a true ImportError
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        return None, f"Transformers not installed (ImportError): {e}"
    except Exception as e:
        return None, f"Transformers import failed: {type(e).__name__}: {e}"

    try:
        model_id = "distilbert-base-uncased-finetuned-sst-2-english"
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # Load model in FP32, then optionally convert weights to FP16 on CUDA
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.to(dev)

        use_half = (dev.type == "cuda" and dtype == torch.float16)
        if use_half:
            # Cast weights to half only on CUDA
            model.half()

        model.eval()

        # Prepare dummy batch (token ids must remain int64)
        texts = [
            "this movie was surprisingly good!" if i % 2 == 0 else "I did not like this at all."
            for i in range(batch)
        ]
        batch_inputs = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq,
        )
        batch_inputs = {k: v.to(dev) for k, v in batch_inputs.items()}

        @torch.inference_mode()
        def bench():
            _ = model(**batch_inputs).logits

        return bench, None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

##########
## Run the benchmarks ("suite runner")

def run_suite(dev: torch.device | str, fp16_on_cuda: bool = True):
    torch.manual_seed(0)
    random.seed(0)
    if isinstance(dev, str):
        dev = torch.device(dev)

    print("\n" + "=" * 80)
    print(f"Running on: {device_name(dev)}")
    print("=" * 80)

    # Choose dtype (only use FP16 on CUDA)
    if dev.type == "cuda" and fp16_on_cuda:
        dtype = torch.float16
        # Better convolution kernels with cuDNN autotuning
        torch.backends.cudnn.benchmark = True
    else:
        dtype = torch.float32

    # 1) Big GEMM
    mm = matmul_bench(dev, M=4096, K=4096, N=4096, dtype=dtype)
    timeit(mm, label=f"[MatMul 4096x4096 FP{16 if dtype == torch.float16 else 32}]")

    # 2) Conv2D
    conv = conv2d_bench(dev, batch=32, cin=64, cout=128, H=128, W=128, k=3, dtype=dtype)
    timeit(conv, label=f"[Conv2D 32x64x128x128 FP{16 if dtype == torch.float16 else 32}]")

    # 3) Tiny Transformer forward (optional)
    hf, hf_err = hf_bench(dev, dtype=dtype, batch=16, seq=128)
    if hf is not None:
        timeit(hf, label=f"[HF DistilBERT batch=16 seq=128 FP{16 if dtype == torch.float16 else 32}]")
    else:
        reason = hf_err or "unknown reason"
        print(f"[HF demo skipped] {reason}")


##########
# Finalize and print outputs

def main():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    # CPU first (baseline)
    run_suite("cpu", fp16_on_cuda=False)

    # GPU suite (if present)
    if torch.cuda.is_available():
        # Warm up GPU context
        torch.cuda.synchronize()
        run_suite("cuda", fp16_on_cuda=True)
    else:
        print("\nNo CUDA GPU detected. Only CPU results shown.")

    print("\nTip: FP16 on CUDA is typical in practice and will usually increase speedup over CPU FP32.")

if __name__ == "__main__":
    main()
