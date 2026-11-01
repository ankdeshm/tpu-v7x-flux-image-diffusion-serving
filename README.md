# Cloud TPU `v7x` (`Ironwood`, `2x2x1` Single-Host `v7x-8`) Image Diffusion Serving (`FLUX.1-schnell`, `FLUX.1-dev`, `Z-Image-Turbo`)

Reproducible TorchTPU serving infrastructure, containers, Kubernetes Job manifests, and empirical multi-configuration benchmark results for **high-throughput & ultra-low-latency image diffusion models** on **Cloud TPU `v7x-8` (`Ironwood`)** (`1` host × `4` physical `v7x` chips = `8` TensorCores / `760 GiB` HBM3e / `1,512 GiB` host RAM, topology `2x2x1`) across BOTH TorchTPU serving engines:

1. **`sglang-torchtpu`** (`sglang serve`, OpenAI `/v1/images/generations` API, `TpuSplashAttention` Pallas FlashAttention & `Torch SDPA`)
2. **`vllm-torchtpu` (`vLLM-Omni`)** (`vllm serve --omni`, OpenAI `/v1/images/generations` & `/v1/chat/completions` API, `USP` Ulysses Sequence Parallelism)

---

## 1. Headline Empirical Results on `1 × v7x-8` Host (`4` Ironwood Chips = `8` TensorCores)

| Engine | Model | Topology (`DP × Ulysses`) | Precision & Attention | Resolution & Steps | Concurrency | Warm P50 Latency (s) | Throughput (img/s) | Throughput (IPM) | Peak HBM / Core (GiB) | `v7x-8` Hosts for `60 IPM` | `v7x-8` Hosts for `300 IPM` |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `BF16` + `tpu_flash_attn` | `512×512`, `4` steps | `4` | **`0.559`** | **`6.220`** | **`373.19`** | `34.30` | **`1`** | **`1`** |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `BF16` + `tpu_flash_attn` | `1024×1024`, `1` step | `4` | **`0.581`** | **`6.213`** | **`372.79`** | `35.35` | **`1`** | **`1`** |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(Low-Lat)* | `BF16` + `tpu_flash_attn` | `512×512`, `1` step | `1` | **`0.232`** | `4.207` | **`252.40`** | `34.04` | **`1`** | `2` |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `1` step | `16` | `3.062` | `3.805` | **`228.33`** | `34.49` | **`1`** | `2` |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `4` steps | `16` | `4.809` | **`3.220`** | **`193.18`** | `32.50` | **`1`** | **`2`** |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `768×1344`, `4` steps | `8` | `2.449` | `3.150` | **`189.01`** | `32.63` | **`1`** | **`2`** |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1344×768`, `4` steps | `8` | `2.458` | `3.145` | **`188.68`** | `32.52` | **`1`** | **`2`** |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `BF16` + `tpu_flash_attn` | `1024×1024`, `4` steps | `32` | `6.045` | `2.990` | **`179.41`** | `35.35` | **`1`** | **`2`** |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Balanced)* | `BF16` + `tpu_flash_attn` | `1344×768`, `4` steps | `4` | **`1.420`** | `2.705` | **`162.33`** | `35.33` | **`1`** | **`2`** |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Balanced)* | `BF16` + `tpu_flash_attn` | `1024×1024`, `4` steps | `4` | **`1.446`** | `2.659` | **`159.53`** | `35.35` | **`1`** | **`2`** |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(Low-Lat)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `1` step | `1` | **`0.498`** | `1.827` | `109.61` | `32.50` | **`1`** | `3` |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(Low-Lat)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `4` steps | `1` | **`1.286`** | `0.751` | `45.06` | `32.50` | `2` | `7` |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(4MP Max-DP)* | `BF16` + `tpu_splash_attn` | `2048×2048`, `4` steps | `8` | `9.966` | **`0.774`** | **`46.42`** | `63.07` | `2` | `7` |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(4MP Low-Lat)* | `BF16` + `tpu_splash_attn` | `2048×2048`, `4` steps | `1` | **`2.727`** | `0.342` | `20.51` | `63.06` | `3` | `15` |
| **`sglang-torchtpu`** | **`Z-Image-Turbo`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `8` steps | `16` | `3.463` | **`2.717`** | **`163.03`** | `22.47` | **`1`** | **`2`** |
| **`sglang-torchtpu`** | **`Z-Image-Turbo`** | **`DP=8 × U=1`** *(4MP Max-DP)* | `BF16` + `tpu_splash_attn` | `2048×2048`, `8` steps | `8` | `8.763` | **`0.868`** | **`52.07`** | `22.81` | `2` | `6` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `20` steps | `8` | `12.127` | **`0.545`** | **`32.69`** | `34.53` | `2` | `10` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `768×1344`, `28` steps | `8` | `15.682` | `0.507` | **`30.41`** | `32.66` | `2` | `10` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `28` steps | `32` | `42.459` | `0.464` | **`27.81`** | `34.56` | `3` | `11` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=1 × U=8`** *(Low-Lat)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `20` steps | `1` | **`4.599`** | `0.208` | `12.49` | `32.53` | `5` | `25` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=1 × U=8`** *(Low-Lat)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `28` steps | `1` | **`5.943`** | `0.166` | `9.95` | `32.53` | `7` | `31` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(50-step)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `50` steps | `8` | `28.280` | `0.282` | **`16.91`** | `32.53` | `4` | `18` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=1 × U=8`** *(4MP Low-Lat)* | `BF16` + `tpu_splash_attn` | `2048×2048`, `28` steps | `1` | **`14.056`** | `0.070` | `4.22` | `63.10` | `15` | `72` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(4MP Max-DP)* | `BF16` + `tpu_splash_attn` | `2048×2048`, `28` steps | `8` | `70.596` | **`0.112`** | **`6.74`** | `62.74` | `9` | `45` |

See [`V7X_FLUX_SERVING_BENCHMARK_RESULTS.md`](./V7X_FLUX_SERVING_BENCHMARK_RESULTS.md) and [`results/`](./results/) for full per-point JSONL telemetry and `DP × Ulysses` scaling analysis.

---

## 2. Benchmark Matrix on Single-Host `v7x-8` (`8` TensorCores = `760 GiB` HBM3e)

- **Models:**
  - [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) (`12B` rectified-flow DiT, Apache-2.0, `1`-step & `4`-step distilled)
  - [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) (`12B` guidance-distilled DiT, `20`-step, `28`-step & `50`-step high-fidelity)
  - [`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (`6B` single-stream distilled DiT comparison)
- **Multi-Replica `DP × Ulysses` Topologies Across `8` TensorCores:**
  - `1 TensorCore` (`num_gpus=1`, `Ulysses=1`) -> **`DP=8` independent replicas/host** (`193.18 IPM` on `FLUX.1-schnell` `1024²` `4`-step)
  - `2 TensorCores` (`num_gpus=2`, `Ulysses=2`, `1` physical `v7x` chip) -> **`DP=4` replicas/host** (strictly intra-chip Ulysses, `0` inter-chip ICI -> **`373.19 IPM` @ `512²` / `372.79 IPM` @ `1024²` `1`-step / `179.41 IPM` @ `1024²` `4`-step**)
  - `4 TensorCores` (`num_gpus=4`, `Ulysses=4`, `2` physical `v7x` chips) -> **`DP=2` replicas/host**
  - `8 TensorCores` (`num_gpus=8`, `Ulysses=8`, all `4` physical `v7x` chips) -> **ultra-low single-image latency** (`0.232s` @ `512²` `1`-step, `0.498s` @ `1024²` `1`-step, `1.286s` @ `1024²` `4`-step, `2.727s` @ `2048²` `4`-step)

---

## 3. Repository Structure

```text
├── README.md
├── V7X_FLUX_SERVING_BENCHMARK_RESULTS.md       # Complete Phase 1 + Phase 2 empirical report
├── docker/
│   ├── Dockerfile.sglang_tpu                   # TorchTPU 0.2.0 + sglang-torchtpu + Pallas Splash Attention
│   ├── prep_sglang_tpu.py                      # ABI constraint lock + TOML-verified sglang-torchtpu setup
│   ├── Dockerfile.vllm_omni                    # TorchTPU 0.2.0 + vllm-torchtpu v0.32.0 + vLLM-Omni
│   └── prep_vllm_omni.py                       # ABI constraint lock for vLLM-Omni on TPU
├── scripts/
│   ├── stage_all_flux_models_gcs.py            # Parallel HF Hub -> regional GCS stager for FLUX.1 & Z-Image-Turbo
│   ├── stage_and_probe_flux.py                 # Node /dev/shm/hf_cache hydrator + sub-host topology probe
│   ├── run_v7x_flux_sglang_serving.py          # Phase 1 sglang-torchtpu FLUX & Z-Image-Turbo benchmark harness
│   └── run_v7x_flux_vllm_omni_serving.py       # Phase 2 vLLM-Omni FLUX DP x Ulysses benchmark harness
├── k8s/
│   ├── v7x-spot-flux-sglang-serving.yaml       # GKE Spot Job (tpu-v7x-4-spot-pool, 2x2x1) for sglang-torchtpu
│   └── v7x-spot-flux-vllm-omni-serving.yaml    # GKE Spot Job (tpu-v7x-4-spot-pool, 2x2x1) for vLLM-Omni
└── results/                                    # Auto-synced JSONL, summary JSON & sample artifacts
```

---

## 4. Quickstart

```bash
# 1. Upload serving benchmark harnesses to regional GCS
CLOUDSDK_METRICS_ENVIRONMENT=datacloud.antigravity gcloud storage cp scripts/*.py gs://YOUR_GCS_BUCKET/flux-serving-scripts/

# 2. Launch TorchTPU FLUX benchmark suites on single-host TPU v7x-8 (tpu-v7x-4-spot-pool)
kubectl --context=gke_YOUR_GCP_PROJECT_ID_us-central1_tpu-v5p-cluster apply -f k8s/v7x-spot-flux-sglang-serving.yaml
kubectl --context=gke_YOUR_GCP_PROJECT_ID_us-central1_tpu-v5p-cluster apply -f k8s/v7x-spot-flux-vllm-omni-serving.yaml
```
