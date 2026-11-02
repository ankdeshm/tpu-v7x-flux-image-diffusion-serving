# Cloud TPU `v7x` (`Ironwood`, `2x2x1` Single-Host `v7x-8`) Image Diffusion Serving (`FLUX.1-schnell`, `FLUX.1-dev`, `Z-Image-Turbo`)

Reproducible TorchTPU serving infrastructure, containers, Kubernetes Job manifests, and empirical multi-configuration benchmark results (`131` verified points, `0` errors) for **high-throughput & ultra-low-latency image diffusion models** on **Cloud TPU `v7x-8` (`Ironwood`)** (`1` host × `4` physical `v7x` chips = `8` TensorCores / `760 GiB` HBM3e / `1,512 GiB` host RAM, topology `2x2x1`) across BOTH TorchTPU serving engines:

1. **`sglang-torchtpu`** ([`scripts/run_v7x_flux_sglang_serving.py`](./scripts/run_v7x_flux_sglang_serving.py), OpenAI `/v1/images/generations` API, `TpuSplashAttention` Pallas FlashAttention)
2. **`vllm-torchtpu` (`vLLM-Omni` v0.32.0)** ([`scripts/run_v7x_flux_vllm_omni_serving.py`](./scripts/run_v7x_flux_vllm_omni_serving.py), `vllm serve --omni`, `USP` Ulysses Sequence Parallelism)

- **Complete Phase 1 + Phase 2 Benchmark Report:** [`V7X_FLUX_SERVING_BENCHMARK_RESULTS.md`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) · Raw Summaries: [`phase1_sglang_summary.json`](./results/phase1_sglang_summary.json) · [`phase2_vllm_omni_summary.json`](./results/phase2_vllm_omni_summary.json)

---

## 1. Headline Empirical Results (`1 × v7x-8` Host = `4` Ironwood Chips / `8` TensorCores) → Exact Script, Manifest & Proof

| Engine | Model | Topology (`DP × U`) | Resolution & Steps | Conc | Warm P50 (s) | Tput (img/s) | **Tput (IPM)** | Peak HBM/TC | Hosts (`60` / `300` IPM) | Script (`#line`) & Manifest | Result Proof |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip #1!)* | `512×512`, `4` steps | `4` | **`0.559`** | **`6.220`** | **`373.19`** | `34.30 GiB` | **`1` / `1`** | [`omni.py#L1255`](./scripts/run_v7x_flux_vllm_omni_serving.py#L1255-L1360) · [`yaml`](./k8s/v7x-spot-flux-vllm-omni-serving.yaml) | [`JSON`](./results/phase2_vllm_omni_summary.json) · [`JSONL`](./results/phase2_vllm_omni_results.jsonl) |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip #1!)* | `1024×1024`, `1` step | `4` | **`0.581`** | **`6.213`** | **`372.79`** | `35.35 GiB` | **`1` / `1`** | [`omni.py#L1255`](./scripts/run_v7x_flux_vllm_omni_serving.py#L1255-L1360) · [`yaml`](./k8s/v7x-spot-flux-vllm-omni-serving.yaml) | [`JSON`](./results/phase2_vllm_omni_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(Min Latency!)* | `512×512`, `1` step | `1` | **`0.232`** | `4.207` | **`252.40`** | `34.04 GiB` | **`1` / `2`** | [`omni.py#L1255`](./scripts/run_v7x_flux_vllm_omni_serving.py#L1255-L1360) · [`yaml`](./k8s/v7x-spot-flux-vllm-omni-serving.yaml) | [`JSON`](./results/phase2_vllm_omni_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`vllm-omni`** | **`Z-Image-Turbo`** | **`DP=4 × U=2`** *(Intra-Chip #1!)* | `1024×1024`, `8` steps | `8` | **`2.156`** *(`1.227s` @ `c=4`)* | **`3.421`** | **`205.28`** | `22.25 GiB` | **`1` / `2`** | [`omni.py#L1255`](./scripts/run_v7x_flux_vllm_omni_serving.py#L1255-L1360) · [`yaml`](./k8s/v7x-spot-flux-vllm-omni-serving.yaml) | [`JSON`](./results/phase2_vllm_omni_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(Max-DP #1!)* | `1024×1024`, `4` steps | `16` | `4.809` | **`3.220`** | **`193.18`** | `32.50 GiB` | **`1` / `2`** | [`sglang.py#L1090`](./scripts/run_v7x_flux_sglang_serving.py#L1090-L1200) · [`yaml`](./k8s/v7x-spot-flux-sglang-serving.yaml) | [`JSON`](./results/phase1_sglang_summary.json) · [`JSONL`](./results/phase1_sglang_results.jsonl) |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `1024×1024`, `4` steps | `32` | `6.045` *(`1.446s` @ `c=4`)* | `2.990` | **`179.41`** | `35.35 GiB` | **`1` / `2`** | [`omni.py#L1255`](./scripts/run_v7x_flux_vllm_omni_serving.py#L1255-L1360) · [`yaml`](./k8s/v7x-spot-flux-vllm-omni-serving.yaml) | [`JSON`](./results/phase2_vllm_omni_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`sglang-torchtpu`** | **`Z-Image-Turbo`** | **`DP=8 × U=1`** *(Max-DP)* | `1024×1024`, `8` steps | `16` | `3.463` | **`2.717`** | **`163.03`** | `22.47 GiB` | **`1` / `2`** | [`sglang.py#L1090`](./scripts/run_v7x_flux_sglang_serving.py#L1090-L1200) · [`yaml`](./k8s/v7x-spot-flux-sglang-serving.yaml) | [`JSON`](./results/phase1_sglang_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(1024² Low-Lat)* | `1024×1024`, `1` / `4` steps | `1` | **`0.498` / `1.286`** | `1.827` / `0.751` | `109.61` / `45.06` | `32.50 GiB` | `1–2` / `3–7` | [`sglang.py#L1090`](./scripts/run_v7x_flux_sglang_serving.py#L1090-L1200) · [`yaml`](./k8s/v7x-spot-flux-sglang-serving.yaml) | [`JSON`](./results/phase1_sglang_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`sglang-torchtpu`** | **`Z-Image-Turbo`** | **`DP=8 × U=1`** *(4MP Max-DP #1!)* | `2048×2048`, `8` steps | `8` | `8.763` | **`0.868`** | **`52.07`** | `22.81 GiB` | `2` / `6` | [`sglang.py#L1090`](./scripts/run_v7x_flux_sglang_serving.py#L1090-L1200) · [`yaml`](./k8s/v7x-spot-flux-sglang-serving.yaml) | [`JSON`](./results/phase1_sglang_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`vllm-omni`** | **`Z-Image-Turbo`** | **`DP=4 × U=2`** *(4MP Low-Lat)* | `2048×2048`, `8` steps | `4` | **`4.959`** | `0.763` | **`45.78`** | `26.45 GiB` | `2` / `7` | [`omni.py#L1255`](./scripts/run_v7x_flux_vllm_omni_serving.py#L1255-L1360) · [`yaml`](./k8s/v7x-spot-flux-vllm-omni-serving.yaml) | [`JSON`](./results/phase2_vllm_omni_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(4MP Max-DP)* | `2048×2048`, `4` steps | `8` | `9.966` *(`2.727s` @ `U=8`)* | **`0.774`** | **`46.42`** | `63.07 GiB` | `2` / `7` | [`sglang.py#L1090`](./scripts/run_v7x_flux_sglang_serving.py#L1090-L1200) · [`yaml`](./k8s/v7x-spot-flux-sglang-serving.yaml) | [`JSON`](./results/phase1_sglang_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(20-step #1!)* | `1024×1024`, `20` steps | `8` | `12.127` *(`4.599s` @ `U=8`)* | **`0.545`** | **`32.69`** | `34.53 GiB` | `2` / `10` | [`sglang.py#L1090`](./scripts/run_v7x_flux_sglang_serving.py#L1090-L1200) · [`yaml`](./k8s/v7x-spot-flux-sglang-serving.yaml) | [`JSON`](./results/phase1_sglang_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`vllm-omni`** | **`FLUX.1-dev`** | **`DP=4 × U=2`** *(28-step #1!)* | `1024×1024`, `28` steps | `8` | **`12.500`** *(`8.366s` @ `c=4`)* | **`0.479`** | **`28.76`** | `35.35 GiB` | **`3` / `11`** | [`omni.py#L1255`](./scripts/run_v7x_flux_vllm_omni_serving.py#L1255-L1360) · [`yaml`](./k8s/v7x-spot-flux-vllm-omni-serving.yaml) | [`JSON`](./results/phase2_vllm_omni_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=1 × U=8`** *(28-step Min-Lat)* | `1024×1024`, `28` steps | `1` | **`5.943`** | `0.166` | `9.95` | `32.53 GiB` | `7` / `31` | [`sglang.py#L1090`](./scripts/run_v7x_flux_sglang_serving.py#L1090-L1200) · [`yaml`](./k8s/v7x-spot-flux-sglang-serving.yaml) | [`JSON`](./results/phase1_sglang_summary.json) · [`Report`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) |

---

## 2. How Baseline Single-Instance Serving Was Optimized (`DP=1 × U=8` → `DP=4 × U=2` & `DP=8 × U=1`)

1. **Sub-Host VFIO Isolation (`/dev/vfio/0..7` + `LD_PRELOAD` `metadata_hook.so`):** Unlocked `DP=8 × U=1` (`1` TensorCore/replica) and **`DP=4 × U=2` (`1` physical dual-TensorCore `v7x` chip/replica, `0` inter-chip ICI!)** via [`ensure_subhost_isolation_ready()` (`#L289`)](./scripts/run_v7x_flux_vllm_omni_serving.py#L289-L350), multiplying host throughput up to **`5.75×`** (`32.33 -> 179.41 IPM` on `FLUX.1-schnell` `1024²` `4`-step, and `5.00 -> 28.76 IPM` on `FLUX.1-dev` `28`-step) while simultaneously **lowering `c=1` latency (`1.935s -> 1.407s`)**!
2. **`vLLM-Omni` TPU Device & Pallas FlashAttention Patch ([`PATCHED_SET_DEVICE_METHOD` `#L204`](./scripts/run_v7x_flux_vllm_omni_serving.py#L204-L280)):** Eliminated sub-host rank collision across multi-replica `vllm serve --omni` workers on a single `v7x-8` node.

---

## 3. Repository Structure & Quickstart

- [`V7X_FLUX_SERVING_BENCHMARK_RESULTS.md`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) — Complete Phase 1 (`sglang-torchtpu`) + Phase 2 (`vllm-omni`) empirical report (`131` points)
- [`docker/Dockerfile.sglang_tpu`](./docker/Dockerfile.sglang_tpu), [`docker/prep_sglang_tpu.py`](./docker/prep_sglang_tpu.py), [`docker/Dockerfile.vllm_omni`](./docker/Dockerfile.vllm_omni), [`docker/prep_vllm_omni.py`](./docker/prep_vllm_omni.py)
- [`scripts/run_v7x_flux_sglang_serving.py`](./scripts/run_v7x_flux_sglang_serving.py) — Phase 1 `sglang-torchtpu` `DP × Ulysses` benchmark harness
- [`scripts/run_v7x_flux_vllm_omni_serving.py`](./scripts/run_v7x_flux_vllm_omni_serving.py) — Phase 2 `vLLM-Omni` `DP × Ulysses` benchmark harness
- [`scripts/stage_all_flux_models_gcs.py`](./scripts/stage_all_flux_models_gcs.py) & [`scripts/stage_and_probe_flux.py`](./scripts/stage_and_probe_flux.py)
- [`k8s/v7x-spot-flux-sglang-serving.yaml`](./k8s/v7x-spot-flux-sglang-serving.yaml) & [`k8s/v7x-spot-flux-vllm-omni-serving.yaml`](./k8s/v7x-spot-flux-vllm-omni-serving.yaml)
- [`results/phase1_sglang_summary.json`](./results/phase1_sglang_summary.json) & [`results/phase2_vllm_omni_summary.json`](./results/phase2_vllm_omni_summary.json)

```bash
# 1. Upload serving benchmark harnesses to regional GCS
gcloud storage cp scripts/*.py gs://YOUR_GCS_BUCKET/flux-serving-scripts/

# 2. Launch Phase 1 (sglang-torchtpu) & Phase 2 (vLLM-Omni) FLUX suites on TPU v7x-8 (tpu-v7x-4-spot-pool)
kubectl apply -f k8s/v7x-spot-flux-sglang-serving.yaml
kubectl apply -f k8s/v7x-spot-flux-vllm-omni-serving.yaml
```
