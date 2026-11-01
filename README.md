# Cloud TPU `v7x` (`Ironwood`, `2x2x1` Single-Host `v7x-8`) Image Diffusion Serving (`FLUX.1-schnell`, `FLUX.1-dev`, `Z-Image-Turbo`)

Reproducible TorchTPU serving infrastructure, containers, Kubernetes Job manifests, and multi-configuration benchmark harnesses for **high-throughput & ultra-low-latency image diffusion models** on **Cloud TPU `v7x-8` (`Ironwood`)** (`1` host × `4` physical `v7x` chips = `8` TensorCores / `760 GiB` HBM3e / `1,512 GiB` host RAM, topology `2x2x1`) across BOTH TorchTPU serving engines:

1. **`sglang-torchtpu`** (`sglang serve`, OpenAI `/v1/images/generations` API, `TpuSplashAttention` Pallas FlashAttention & `Torch SDPA`, `BF16` & Online `FP8`)
2. **`vllm-torchtpu` (`vLLM-Omni`)** (`vllm serve --omni`, OpenAI `/v1/images/generations` & `/v1/chat/completions` API, `USP` Ulysses Sequence Parallelism)

---

## 1. Benchmark Matrix on Single-Host `v7x-8` (`8` TensorCores = `760 GiB` HBM3e)

- **Models:**
  - [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) (`12B` rectified-flow DiT, Apache-2.0, `1`-step & `4`-step distilled)
  - [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) (`12B` guidance-distilled DiT, `20`-step, `28`-step & `50`-step high-fidelity)
  - [`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (`6B` single-stream distilled DiT comparison)
- **Multi-Replica `DP × Ulysses` Topologies Across `8` TensorCores:**
  - `1 TensorCore` (`num_gpus=1`, `Ulysses=1`) -> **`DP=8` independent replicas/host** (maximum IPM throughput)
  - `2 TensorCores` (`num_gpus=2`, `Ulysses=2`, `1` physical `v7x` chip) -> **`DP=4` replicas/host** (strictly intra-chip Ulysses, zero inter-chip ICI)
  - `4 TensorCores` (`num_gpus=4`, `Ulysses=4`, `2` physical `v7x` chips) -> **`DP=2` replicas/host**
  - `8 TensorCores` (`num_gpus=8`, `Ulysses=8`, all `4` physical `v7x` chips) -> **ultra-low single-image latency** (especially at `2048×2048` 4MP UHD)
- **Resolutions, Steps, Precisions & Concurrency:**
  - **Resolutions:** `512×512`, `1024×1024` (`1MP`), `1344×768` (`16:9`), `768×1344` (`9:16`), and `2048×2048` (`4MP` UHD)
  - **Steps:** `1`, `4` (`FLUX.1-schnell`); `20`, `28`, `50` (`FLUX.1-dev`); `8` (`Z-Image-Turbo`)
  - **Precision & Attention:** `BF16` vs `FP8` (`--quantization fp8`), `Pallas Splash Attention` vs `Torch SDPA`
  - **Concurrency:** `c = 1, 2, 4, 8, 16, 32`
  - **Sizing Targets:** `v7x-8` hosts required for **`60 IPM`** (`1 img/s`) and **`300 IPM`** (`5 img/s`) (latency, IPS, IPM, Peak HBM, Cold vs Warm compile/inference time — zero cost columns).

---

## 2. Repository Structure

```text
├── README.md
├── docker/
│   ├── Dockerfile.sglang_tpu                   # TorchTPU 0.2.0 + sglang-torchtpu + Pallas Splash Attention
│   ├── prep_sglang_tpu.py                      # ABI constraint lock + TOML-verified sglang-torchtpu setup
│   ├── Dockerfile.vllm_omni                    # TorchTPU 0.2.0 + vllm-torchtpu v0.32.0 + vLLM-Omni
│   └── prep_vllm_omni.py                       # ABI constraint lock for vLLM-Omni on TPU
├── scripts/
│   ├── stage_all_flux_models_gcs.py            # Parallel HF Hub -> regional GCS stager for FLUX.1 & Z-Image-Turbo
│   ├── stage_and_probe_flux.py                 # Node /dev/shm/hf_cache hydrator + sub-host topology probe
│   └── run_v7x_flux_serving.py                 # Full sglang-torchtpu & vLLM-Omni FLUX benchmark harness
├── k8s/
│   ├── v7x-spot-flux-sglang-serving.yaml       # GKE Spot Job (tpu-v7x-4-spot-pool, 2x2x1) for sglang-torchtpu
│   └── v7x-spot-flux-vllm-omni-serving.yaml    # GKE Spot Job (tpu-v7x-4-spot-pool, 2x2x1) for vLLM-Omni
└── results/                                    # Auto-synced Markdown & JSONL benchmark summaries
```

---

## 3. Quickstart

```bash
# 1. Upload serving benchmark harnesses to regional GCS
CLOUDSDK_METRICS_ENVIRONMENT=datacloud.antigravity gcloud storage cp scripts/*.py gs://YOUR_GCS_BUCKET/flux-serving-scripts/

# 2. Launch TorchTPU FLUX benchmark suite on single-host TPU v7x-8 (tpu-v7x-4-spot-pool)
CLOUDSDK_METRICS_ENVIRONMENT=datacloud.antigravity kubectl --context=gke_YOUR_GCP_PROJECT_ID_us-central1_tpu-v5p-cluster apply -f k8s/v7x-spot-flux-sglang-serving.yaml
```
