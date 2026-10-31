# Cloud TPU `v7x-8` (`Ironwood`) — FLUX & Image Diffusion TorchTPU Serving Benchmark Results

- **Project / Region / Zone:** `diesel-patrol-382622` / `us-central1` / `us-central1-c`
- **Cluster & Isolated Nodepool:** `ank-tpu-v5p-usc1` -> **`ank-v7-spot-pool`** (`tpu7x-standard-4t`, `1` host × `4` physical `v7x` chips = `8` TensorCores / `760 GiB` HBM3e / `1,512 GiB` host RAM, topology `2x2x1`)
- **GCS Artifacts:** `gs://ank-usc1-bucket-314837540096/flux-serving-results/` & `gs://ank-usc1-bucket-314837540096/flux-serving-scripts/`
- **GitHub Repository:** [`ankdeshm/tpu-v7x-flux-image-diffusion-serving`](https://github.com/ankdeshm/tpu-v7x-flux-image-diffusion-serving) (`/Users/deshmukhankita/ankita-dev/github-repos/tpu-v7x-flux-image-diffusion-serving`)
- **100% TorchTPU Serving Stack (`torch_tpu 0.2.0` GA + `libtpu 0.0.50`):**
  1. **`sglang-torchtpu`:** `us-central1-docker.pkg.dev/diesel-patrol-382622/vllm/sglang-torchtpu-diffusion:v0.2.0` (`sglang serve`, OpenAI `/v1/images/generations` API)
  2. **`vllm-omni` (`vllm-torchtpu`):** `us-central1-docker.pkg.dev/diesel-patrol-382622/vllm/vllm-torchtpu-omni:v0.32.0` (`vllm serve --omni`, OpenAI `/v1/images/generations` & `/v1/chat/completions` API)
- **Status:** `IN_PROGRESS` — Staging models into `900 GiB` `/dev/shm/hf_cache` + regional GCS & probing sub-host `DP × Ulysses` topologies on `gke-tpu-58112ace-2qgx`.

---

## 1. Hardware & Memory Architecture (`1 × v7x-8` Single Host = `8` TensorCores / `760 GiB` HBM3e)

| Hardware / Model Spec | Value | Architectural Implication for `FLUX.1` & Small Diffusion |
|---|---|---|
| **Single `v7x` Chip** | `2` TensorCores, `190 GiB` HBM3e (`95 GiB`/TensorCore), `7.38 TB/s` BW | Fits **2 independent `BF16` `FLUX.1` replicas (`~33.8 GB` each)** or 1 intra-chip `Ulysses=2` replica with zero inter-chip ICI |
| **Single `v7x-8` Host (`2x2x1`)** | `4` `v7x` Chips = `8` TensorCores, `760 GiB` HBM3e, `29.5 TB/s` BW, `1,512 GiB` Host RAM | Supports **`DP=8 × Ulysses=1`** (`8` independent replicas), **`DP=4 × Ulysses=2`**, **`DP=2 × Ulysses=4`**, and **`DP=1 × Ulysses=8`** |
| **RAM-Disk Staging (`/dev/shm/hf_cache`)** | `900 GiB` tmpfs backed by `1,512 GiB` DDR5 host RAM | Stages `FLUX.1-schnell`, `FLUX.1-dev`, and `Z-Image-Turbo` simultaneously in host RAM for instant multi-replica weight loading |

---

## 2. Executive Summary & Sizing Leaderboard (Populated Continuously as Phases Complete)

| Config ID | Engine | Model | Precision & Attention | Topology (`DP × Ulysses`) | Resolution & Steps | Concurrency | Cold Compile + Infer (s) | Warm P50 Latency (s) | Throughput (img/s) | Throughput (IPM) | Peak HBM / Core (GiB) | `v7x-8` Hosts for `60 IPM` | `v7x-8` Hosts for `300 IPM` |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Pending Phase 1* | `sglang-torchtpu` | `FLUX.1-schnell` | `BF16` + Pallas Splash | `DP × Ulysses` sweep | `512²..2048²`, `1 & 4` steps | `1..32` | — | — | — | — | — | — | — |
| *Pending Phase 2* | `vllm-omni` | `FLUX.1-schnell` | `BF16` + TorchTPU | `DP × USP` sweep | `512²..2048²`, `1 & 4` steps | `1..32` | — | — | — | — | — | — | — |
| *Pending Phase 3* | `sglang-torchtpu` & `vllm-omni` | `FLUX.1-dev` / `Z-Image-Turbo` | `BF16` & `FP8` | `DP × Ulysses` sweep | `512²..2048²`, `8..50` steps | `1..32` | — | — | — | — | — | — | — |
