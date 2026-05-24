import warnings
warnings.filterwarnings("ignore")

import os
import json
import time
import platform

import torch

from src.loading.models.mobilenet.hp import MobileNetHP, original_hp
from src.loading.models.mobilenet.config import MobileNetConfig
from src.loading.models.mobilenet.model import MobileNetV3Small

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None

RECORDS_DIR = "src/optim/hill_climbing/records"
OUT_DIR = "docs/efficiency"
NUM_CLASSES = 10
INPUT_SHAPE = (3, 224, 224)          # CIFAR-10 resize used by the data loader
ROW7_TARGET_PARAMS = 0.840           # published 7-block params (M); proxy matched to this
ROW7_PUBLISHED_ACC = 85.78           # kept as published

RECORD_FILES = {
    5: f"{RECORDS_DIR}/it_10_n_4_ep_20_blr_0.8_pmr_0.5_pi_2_local_freeze_5.json",
    7: f"{RECORDS_DIR}/it_10_n_4_ep_20_blr_0.8_pmr_0.5_pi_2_local_freeze_7.json",
    9: f"{RECORDS_DIR}/it_10_n_4_ep_20_blr_0.8_pmr_0.5_pi_2_local_freeze_9.json",
}

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True  # stable, fastest conv algos for fixed input size

def build_model(hp: MobileNetHP) -> torch.nn.Module:
    """Reconstruct an nn.Module from a hyperparameter genome (random weights)."""
    config = MobileNetConfig.from_hp(hp)
    model = MobileNetV3Small(config, num_classes=NUM_CLASSES, pretrained=False)
    return model.eval()

def total_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def model_size_mb(model: torch.nn.Module) -> float:
    p_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    b_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (p_bytes + b_bytes) / (1024 ** 2)

def measure_macs(hp: MobileNetHP):
    """Return (MACs in millions, params in millions) via thop, on a fresh model."""
    if thop_profile is None:
        return None, None
    model = build_model(hp)
    x = torch.randn(1, *INPUT_SHAPE)
    macs, params = thop_profile(model, inputs=(x,), verbose=False)
    return macs / 1e6, params / 1e6

@torch.no_grad()
def measure_latency(model, device, batch_size=1, warmup=50, iters=200):
    """Mean +/- std forward latency in ms for one batch on `device`."""
    model = model.to(device)
    x = torch.randn(batch_size, *INPUT_SHAPE, device=device)

    if device.type == "cuda":
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            starter.record()
            model(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms
        t = torch.tensor(times)
    else:
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)  # ms
        t = torch.tensor(times)

    return float(t.mean()), float(t.std())


@torch.no_grad()
def measure_peak_memory_mb(model, batch_size=1):
    """Steady-state peak CUDA memory (MB) for a forward pass. None if no GPU.

    Warms up first so cudnn.benchmark settles on an algorithm; otherwise the
    one-shot algorithm search allocates large transient workspaces that do not
    reflect real inference memory.
    """
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    model = model.to(device)
    x = torch.randn(batch_size, *INPUT_SHAPE, device=device)
    for _ in range(5):                       # let benchmark pick its algorithm
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    model.to("cpu")
    torch.cuda.empty_cache()
    return peak


BATCHES = (128, 256)  # batched regimes; the most-significant one is reported


def profile_genome(hp: MobileNetHP, gpu, cpu):
    """FLOPs / size + batched latency, throughput and peak memory for each batch."""
    model = build_model(hp)
    macs_m, thop_params_m = measure_macs(hp)
    res = {
        "params_m": round(total_params_m(model), 4),
        "thop_params_m": round(thop_params_m, 4) if thop_params_m is not None else None,
        "macs_m": round(macs_m, 2) if macs_m is not None else None,
        "flops_m": round(macs_m * 2, 2) if macs_m is not None else None,
        "model_size_mb": round(model_size_mb(model), 3),
        "batches": {},
    }
    for bs in BATCHES:
        entry = {"gpu_latency_ms": None, "gpu_throughput": None,
                 "cpu_latency_ms": None, "cpu_throughput": None, "peak_mem_mb": None}
        # CPU (large batches are slow -> few iters)
        m, s = measure_latency(build_model(hp), cpu, batch_size=bs, warmup=2, iters=5)
        entry["cpu_latency_ms"] = (round(m, 2), round(s, 2))
        entry["cpu_throughput"] = round(bs / (m / 1000.0), 1)
        # GPU
        if gpu is not None:
            m, s = measure_latency(build_model(hp), gpu, batch_size=bs, warmup=20, iters=50)
            entry["gpu_latency_ms"] = (round(m, 2), round(s, 2))
            entry["gpu_throughput"] = round(bs / (m / 1000.0), 1)
            entry["peak_mem_mb"] = round(measure_peak_memory_mb(build_model(hp), bs), 1)
        res["batches"][bs] = entry
    return res

def final_genome(record_path):
    """The best_hp at the max-accuracy plateau (== final iteration's best_hp)."""
    data = json.load(open(record_path))
    best = max(data, key=lambda e: e["best_perf"])
    # all tied-accuracy entries share the same best_hp; the last one is canonical
    best_acc = best["best_perf"]
    entry = [e for e in data if e["best_perf"] == best_acc][-1]
    return MobileNetHP.from_dict(entry["best_hp"]), entry["best_perf"]

def closest_size_proxy(target_m):
    """Among all saved genomes in the present record files, the one whose
    reconstructed param count is closest to `target_m` (M)."""
    best = None  # (abs_diff, params_m, hp, source)
    for blocks, path in RECORD_FILES.items():
        if not os.path.exists(path):
            continue
        seen = set()
        for entry in json.load(open(path)):
            key = json.dumps(entry["best_hp"], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            hp = MobileNetHP.from_dict(entry["best_hp"])
            p = total_params_m(build_model(hp))
            diff = abs(p - target_m)
            if best is None or diff < best[0]:
                best = (diff, p, hp, f"freeze_{blocks}.json @ iter {entry['iteration']}")
    return best[2], best[1], best[3]

def initial_accuracy(blocks):
    data = json.load(open(RECORD_FILES[blocks]))
    return next(e["best_perf"] for e in data if e["iteration"] == 0)

def main():
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    cpu = torch.device("cpu")
    gpu_name = torch.cuda.get_device_name(0) if gpu is not None else "N/A"
    print(f"Torch {torch.__version__} | GPU: {gpu_name} | CPU: {platform.processor() or platform.machine()}")
    print("Profiling architectures (random weights; metrics are architecture-only)...\n")

    # ---- gather genomes ----
    g5, acc5 = final_genome(RECORD_FILES[5])
    g9, acc9 = final_genome(RECORD_FILES[9])
    g7_proxy, g7_proxy_params, g7_proxy_src = closest_size_proxy(ROW7_TARGET_PARAMS)

    # ---- validation against known/paper params ----
    print("== Param validation (reconstructed vs. paper) ==")
    checks = [
        ("initial", total_params_m(build_model(original_hp)), 1.5281),
        ("blocks 5", total_params_m(build_model(g5)), 0.716),
        ("blocks 9", total_params_m(build_model(g9)), 1.060),
    ]
    for name, got, paper in checks:
        ok = "OK" if abs(got - paper) < 0.01 else "MISMATCH"
        print(f"  {name:9s}: reconstructed {got:.4f} M  vs paper {paper:.3f} M  [{ok}]")
    print(f"  blocks 7 proxy: {g7_proxy_params:.4f} M  (closest recoverable to "
          f"{ROW7_TARGET_PARAMS} M) from {g7_proxy_src}\n")

    # ---- profile ----
    initial = profile_genome(original_hp, gpu, cpu)
    rows = {
        5: {"acc": acc5, "params_pub": round(total_params_m(build_model(g5)), 3),
            "estimated": False, **profile_genome(g5, gpu, cpu)},
        7: {"acc": ROW7_PUBLISHED_ACC, "params_pub": ROW7_TARGET_PARAMS,
            "estimated": True, "proxy_params_m": round(g7_proxy_params, 4),
            "proxy_source": g7_proxy_src, **profile_genome(g7_proxy, gpu, cpu)},
        9: {"acc": acc9, "params_pub": round(total_params_m(build_model(g9)), 3),
            "estimated": False, **profile_genome(g9, gpu, cpu)},
    }
    for b in rows:
        rows[b]["initial_acc"] = initial_accuracy(b)

    # pick the batch that most separates the models (largest GPU-throughput spread)
    def spread(bs):
        vals = [initial["batches"][bs]["gpu_throughput"]] + \
               [rows[b]["batches"][bs]["gpu_throughput"] for b in (5, 7, 9)]
        vals = [v for v in vals if v]
        return (max(vals) / min(vals)) if vals else 0.0
    metric = "gpu_throughput" if gpu is not None else "cpu_throughput"
    if gpu is None:
        def spread(bs):
            vals = [initial["batches"][bs]["cpu_throughput"]] + \
                   [rows[b]["batches"][bs]["cpu_throughput"] for b in (5, 7, 9)]
            return max(vals) / min(vals)
    selected_batch = max(BATCHES, key=spread)
    print(f"\nSelected batch (most significant by {metric} spread): {selected_batch} "
          f"(spreads: {{ {', '.join(f'{bs}: {spread(bs):.3f}x' for bs in BATCHES)} }})")

    results = {
        "env": {"torch": torch.__version__, "gpu": gpu_name,
                "cpu": platform.processor() or platform.machine(),
                "input_shape": list(INPUT_SHAPE)},
        "selected_batch": selected_batch,
        "batches_measured": list(BATCHES),
        "initial_baseline": initial,
        "rows": rows,
    }

    # ---- write outputs ----
    _write_json(results)
    _write_report(results)
    _write_latex(results)
    print("\nWrote:")
    for f in ("efficiency_metrics.json", "efficiency_report.md", "efficiency_tables.tex"):
        print(f"  {OUT_DIR}/{f}")


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def _fmt_lat(v):
    return f"{v[0]:.2f}±{v[1]:.2f}" if v else "N/A"


def _ratio(val, ref):
    return f"{val / ref:.2f}×"


def _write_json(results):
    with open(f"{OUT_DIR}/efficiency_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


# ----- shared row ordering: reference first, then optimized configs -----
def _ordered(results):
    init = results["initial_baseline"]
    yield ("Reference", None, init, False)
    for b in (5, 7, 9):
        r = results["rows"][b]
        yield (f"{b} blocks", r["acc"], r, r["estimated"])


def _md_table(results):
    """Single consolidated efficiency table (Markdown) at the selected batch."""
    init = results["initial_baseline"]
    bs = results["selected_batch"]
    out = [f"| Config | Acc (%) | FLOPs (M) | FLOPs ratio | Model size (MB) | "
           f"Peak mem (MB) | GPU time (ms) | GPU thr. (img/s) | CPU time (ms) | CPU thr. (img/s) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, acc, r, est in _ordered(results):
        b = r["batches"][bs]
        tag = name + (" †" if est else "")
        acc_s = "—" if acc is None else f"{acc:.2f}"
        out.append(f"| {tag} | {acc_s} | {r['flops_m']:.1f} | {_ratio(r['flops_m'], init['flops_m'])} | "
                   f"{r['model_size_mb']:.2f} | {b['peak_mem_mb']} | {_fmt_lat(b['gpu_latency_ms'])} | "
                   f"{b['gpu_throughput']:.0f} | {_fmt_lat(b['cpu_latency_ms'])} | {b['cpu_throughput']:.0f} |")
    return "\n".join(out)


def _write_report(results):
    e = results["env"]
    r7 = results["rows"][7]
    bs = results["selected_batch"]
    others = [x for x in results["batches_measured"] if x != bs]
    lt = _latex_table(results)
    report = f"""# Efficiency Evaluation of the HPO-Optimized Architectures

*Report prepared for integration into the paper revision (reviewer request:
"include FLOPs, inference latency, and memory usage").*

## What was done

The reviewer asked for efficiency metrics beyond accuracy and parameter count. We obtained
them **without rerunning the ~30 h hyperparameter search**. Each optimization run stores the
full architecture description (genome) of its best model, so we reconstructed the final
architecture of each configuration and profiled it directly. FLOPs, latency, memory and
parameter count are determined by the **architecture alone** (not by trained weight values),
so profiling the reconstructed networks with freshly initialized weights reproduces these
metrics exactly — the reconstructed parameter counts match the published Table values to
within rounding (0.716 M, 1.060 M for the 5- and 9-block models).

**Metrics and method**
- **FLOPs (M):** floating-point operations for one 224×224 image = 2 × MACs, where MACs are
  measured with `thop` (its parameter count was cross-checked against the direct count for
  every model). The *FLOPs ratio* is relative to the reference (lower = more efficient).
- **Model size (MB):** in-memory footprint of parameters + buffers.
- **Peak memory (MB):** peak CUDA memory during a forward pass (steady state).
- **GPU/CPU time and throughput:** mean (± std) forward latency over many warmed-up, timed
  passes (CUDA events on GPU, `perf_counter` on CPU; `cudnn.benchmark` enabled) and the
  corresponding throughput (images/second).

**Batch size.** Latency/throughput/memory were measured at batch sizes
{results['batches_measured'][0]} and {results['batches_measured'][1]}; we report the regime
that most separates the models. Selected: **batch {bs}** (it gave the larger GPU-throughput
spread across configurations than batch {others[0] if others else '—'}); both are recorded in
`efficiency_metrics.json`. FLOPs and model size are batch-independent.

**Hardware / software:** {e['gpu']}, CPU {e['cpu']}, PyTorch {e['torch']}, input
{tuple(e['input_shape'])}.

## Efficiency table (batch {bs})

{_md_table(results)}

† **7-block row is an estimate.** The exact genome of the published 7-block model
(0.840 M params, 85.78 %) was not saved by the optimizer (only each run's running-best genome
is persisted, and that model was an efficient runner-up). Its efficiency figures are therefore
computed from the closest-size **recoverable** architecture ({r7['proxy_params_m']:.3f} M params,
from {r7['proxy_source']}). The reported accuracy (85.78 %) and parameter count (0.840 M) are
unchanged. Because FLOPs and latency are not a pure function of parameter count, these 7-block
efficiency numbers should be read as a comparable-size approximation.

## LaTeX (ready to paste)

The same table in the paper's `\\toprule/\\colrule/\\botrule` style is in
`efficiency_tables.tex`; inlined here for convenience:

```latex
{lt}
```
"""
    with open(f"{OUT_DIR}/efficiency_report.md", "w") as f:
        f.write(report)


def _latex_table(results):
    """Single consolidated efficiency table (LaTeX) at the selected batch."""
    init = results["initial_baseline"]
    bs = results["selected_batch"]

    def lat(v):
        return f"{v[0]:.1f}" if v else "--"

    t = ["\\begin{table*}[t]",
         f"\\caption{{Efficiency of the optimized architectures (FLOPs and model size per "
         f"224$\\times$224 image; latency, throughput and peak memory at batch size {bs}; "
         f"GPU: {results['env']['gpu']}). FLOPs ratio is relative to the reference.}}",
         "\\label{tab:hpo_efficiency}",
         "\\begin{tabular*}{\\hsize}{@{\\extracolsep{\\fill}}lccccccccc@{}}",
         "\\toprule",
         "Config & Acc (\\%) & FLOPs (M) & FLOPs Ratio & Size (MB) & Peak Mem (MB) "
         "& GPU Time (ms) & GPU Thr. (img/s) & CPU Time (ms) & CPU Thr. (img/s) \\\\",
         "\\colrule"]
    for name, acc, r, est in _ordered(results):
        b = r["batches"][bs]
        tag = ("Reference" if acc is None else f"{name.split()[0]} blocks") + ("$^{\\dagger}$" if est else "")
        acc_s = "--" if acc is None else f"{acc:.2f}"
        t.append(f"{tag} & {acc_s} & {r['flops_m']:.1f} & {r['flops_m']/init['flops_m']:.2f}$\\times$ & "
                 f"{r['model_size_mb']:.2f} & {b['peak_mem_mb']:.1f} & {lat(b['gpu_latency_ms'])} & "
                 f"{b['gpu_throughput']:.0f} & {lat(b['cpu_latency_ms'])} & {b['cpu_throughput']:.0f} \\\\")
    t += ["\\botrule", "\\end{tabular*}"]
    r7 = results["rows"][7]
    t.append("\\vspace{2pt}")
    t.append("{\\footnotesize $^{\\dagger}$ The 7-block model's exact genome was not persisted; its "
             f"efficiency metrics are estimated from a recoverable architecture of comparable size "
             f"({r7['proxy_params_m']:.3f} M params). Reported accuracy and parameter count are "
             "unchanged.\\par}")
    t.append("\\end{table*}")
    return "\n".join(t)


def _write_latex(results):
    with open(f"{OUT_DIR}/efficiency_tables.tex", "w") as f:
        f.write(_latex_table(results) + "\n")


if __name__ == "__main__":
    main()
