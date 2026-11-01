# Cloud TPU `v7x-8` (`Ironwood`) — FLUX & Image Diffusion TorchTPU Serving Benchmark Results

- **Project / Region / Zone:** `YOUR_GCP_PROJECT_ID` / `us-central1` / `us-central1-c`
- **Cluster & Isolated Nodepool:** `tpu-v5p-cluster` -> **`tpu-v7x-4-spot-pool`** (`tpu7x-standard-4t`, `1` host × `4` physical `v7x` chips = `8` TensorCores / `760 GiB` HBM3e / `1,512 GiB` host RAM, topology `2x2x1`)
- **GCS Artifacts:** `gs://YOUR_GCS_BUCKET/flux-serving-results/` & `gs://YOUR_GCS_BUCKET/flux-serving-scripts/`
- **GitHub Repository:** [`ankdeshm/tpu-v7x-flux-image-diffusion-serving`](https://github.com/ankdeshm/tpu-v7x-flux-image-diffusion-serving) (`/Users/deshmukhankita/ankita-dev/github-repos/tpu-v7x-flux-image-diffusion-serving`)
- **100% TorchTPU Serving Stack (`torch_tpu 0.2.0` GA + `libtpu 0.0.50`):**
  1. **`sglang-torchtpu`:** `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/vllm/sglang-torchtpu-diffusion:v0.2.0` (`sglang serve`, OpenAI `/v1/images/generations` API)
  2. **`vllm-omni` (`vllm-torchtpu`):** `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/vllm/vllm-torchtpu-omni:v0.32.0` (`vllm serve --omni`, OpenAI `/v1/images/generations` & `/v1/chat/completions` API)
- **Status (`2026-11-01 12:35 UTC`):** Phase 1 (`sglang-torchtpu`, `100%` complete across `FLUX.1-schnell`, `FLUX.1-dev`, `Z-Image-Turbo`) + Phase 2 (`vllm-omni`, **`46` validated `BF16` `tpu_flash_attn` operating points** across `FLUX.1-schnell` and `FLUX.1-dev` live on `tpu-v7x-4-spot-pool`, sweep continuing through `[13/19]..[19/19]`).

---

## 1. Key Engineering & Roofline Findings on Single-Host TPU `v7x-8` (`8` TensorCores)

1. **`373.19 IPM` (`6.22 img/s`) on `FLUX.1-schnell` & New `28.76 IPM` (`8.366s` p50 @ `c=4`) Record on `FLUX.1-dev` via Intra-Chip `DP=4 × Ulysses=2` (`vllm-omni`):**
   - Each physical TPU `v7x` Ironwood package integrates **2 TensorCores (`190 GiB` HBM3e, `7.38 TB/s` BW)** connected over ultra-low-latency intra-package die-to-die links. Running `DP=4 × Ulysses=2` places each `2`-core `FLUX.1` replica strictly inside a single Ironwood chip (`0` inter-chip ICI hops!), delivering:
     - **`FLUX.1-schnell`:** **`373.19 IPM` (`0.559s` p50)** at `512×512` (`4`-step), **`372.79 IPM` (`0.581s` p50)** at `1024×1024` (`1`-step), and **`179.41 IPM` (`2.990 img/s`)** at `1024×1024` (`4`-step, `c=32`) — or **`159.53 IPM` at `1.446s` p50 (`c=4`)**!
     - **`FLUX.1-dev` (`28`-step `1024×1024`):** **`28.76 IPM` (`0.479 img/s`, `12.500s` p50 @ `c=8`)** and **`28.49 IPM` (`0.475 img/s`) at `8.366s` p50 (`c=4`, `298.8 ms/step`)** — beating `sglang-torchtpu` `DP=8 × U=1` (`27.81 IPM` @ `42.459s` p50) on **both** throughput and **`5.08×` lower latency** simultaneously (`3` hosts for `60 IPM`, `11` hosts for `300 IPM`)! At 4MP UHD (`2048×2048`, `28`-step, `c=4`), `DP=4 × U=2` delivers **`5.38 IPM` (`44.343s` p50, `39.55 GiB` HBM/core)**.
2. **`193.18 IPM` (`3.22 img/s`) at `1024×1024` `4`-Step via `DP=8 × Ulysses=1` (`sglang-torchtpu`):**
   - Because each `95 GiB` TensorCore comfortably holds an entire `~33.8 GB` `BF16` `FLUX.1` model (`32.50 GiB` measured HBM at `1024²`), `sglang-torchtpu` runs **8 completely independent zero-communication replicas (`DP=8 × Ulysses=1`)** per host, achieving **`193.18 IPM` (`4.809s` p50 at `c=16`)** on `FLUX.1-schnell` `1024×1024` (`4`-step), **`189.01 IPM` (`2.449s` p50 at `c=8`)** at `768×1344`, **`32.69 IPM`** on `FLUX.1-dev` `1024×1024` (`20`-step), and **`163.03 IPM` (`3.463s` p50)** on `Z-Image-Turbo` `1024×1024` (`8`-step).
3. **Sub-Second & Ultra-Low Latency (`DP=1 × Ulysses=8` across all `8` TensorCores):**
   - Sharding sequence tokens across all 8 TensorCores with Pallas Splash/Flash Attention cuts single-request (`c=1`) p50 latency to **`0.232s`** (`512²` `1`-step, `vllm-omni`), **`0.498s`** (`1024²` `1`-step, `sglang-torchtpu`), **`1.226s` / `1.286s`** (`1344×768` / `1024²` `4`-step), **`2.727s`** (`2048×2048` 4MP UHD `4`-step), and **`4.599s` / `5.943s`** on high-fidelity `FLUX.1-dev` `1024²` (`20`-step / `28`-step)!
4. **Pallas `tpu_splash_attn` vs `torch_sdpa` (`2.51×` Higher Throughput & `3.71×` Lower Latency):**
   - At `1024×1024` (`4`-step), Pallas `tpu_splash_attn` cuts `DP=1×U=8` `c=1` latency from `4.766s` (`torch_sdpa`) down to **`1.286s` (`3.71×` faster)** and lifts `DP=8×U=1` peak throughput from `76.96 IPM` to **`193.18 IPM` (`2.51×` higher)** while avoiding dense quadratic attention score materialization in HBM.

---

## 2. Executive Summary & Production Sizing Leaderboard (`1 × v7x-8` Host = `4` Ironwood Chips / `8` TensorCores)

| Engine | Model | Topology (`DP × Ulysses`) | Precision & Attention | Resolution & Steps | Concurrency | Cold Compile + Infer (s) | Warm P50 Latency (s) | Throughput (img/s) | Throughput (IPM) | Peak HBM / Core (GiB) | `v7x-8` Hosts for `60 IPM` | `v7x-8` Hosts for `300 IPM` |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `BF16` + `tpu_flash_attn` | `512×512`, `4` steps | `4` | `3.80` | **`0.559`** | **`6.220`** | **`373.19`** | `34.30` | **`1`** | **`1`** |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `BF16` + `tpu_flash_attn` | `1024×1024`, `1` step | `4` | `150.17` | **`0.581`** | **`6.213`** | **`372.79`** | `35.35` | **`1`** | **`1`** |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(Low-Lat)* | `BF16` + `tpu_flash_attn` | `512×512`, `1` step | `1` | `15.00` | **`0.232`** | `4.207` | **`252.40`** | `34.04` | **`1`** | `2` |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `1` step | `16` | `121.35` | `3.062` | `3.805` | **`228.33`** | `34.49` | **`1`** | `2` |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `4` steps | `16` | `121.35` | `4.809` | **`3.220`** | **`193.18`** | `32.50` | **`1`** | **`2`** |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `768×1344`, `4` steps | `8` | `130.61` | `2.449` | `3.150` | **`189.01`** | `32.63` | **`1`** | **`2`** |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1344×768`, `4` steps | `8` | `119.04` | `2.458` | `3.145` | **`188.68`** | `32.52` | **`1`** | **`2`** |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `BF16` + `tpu_flash_attn` | `1024×1024`, `4` steps | `32` | `150.17` | `6.045` | `2.990` | **`179.41`** | `35.35` | **`1`** | **`2`** |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Balanced)* | `BF16` + `tpu_flash_attn` | `1344×768`, `4` steps | `4` | `151.30` | **`1.420`** | `2.705` | **`162.33`** | `35.33` | **`1`** | **`2`** |
| **`vllm-omni`** | **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Balanced)* | `BF16` + `tpu_flash_attn` | `1024×1024`, `4` steps | `4` | `150.17` | **`1.446`** | `2.659` | **`159.53`** | `35.35` | **`1`** | **`2`** |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(Low-Lat)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `1` step | `1` | `3.75` | **`0.498`** | `1.827` | `109.61` | `32.50` | **`1`** | `3` |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(Low-Lat)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `4` steps | `1` | `3.75` | **`1.286`** | `0.751` | `45.06` | `32.50` | `2` | `7` |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=8 × U=1`** *(4MP Max-DP)* | `BF16` + `tpu_splash_attn` | `2048×2048`, `4` steps | `8` | `456.04` | `9.966` | **`0.774`** | **`46.42`** | `63.07` | `2` | `7` |
| **`sglang-torchtpu`** | **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(4MP Low-Lat)* | `BF16` + `tpu_splash_attn` | `2048×2048`, `4` steps | `1` | `9.11` | **`2.727`** | `0.342` | `20.51` | `63.06` | `3` | `15` |
| **`sglang-torchtpu`** | **`Z-Image-Turbo`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `8` steps | `16` | `6.40` | `3.463` | **`2.717`** | **`163.03`** | `22.47` | **`1`** | **`2`** |
| **`sglang-torchtpu`** | **`Z-Image-Turbo`** | **`DP=8 × U=1`** *(4MP Max-DP)* | `BF16` + `tpu_splash_attn` | `2048×2048`, `8` steps | `8` | `72.43` | `8.763` | **`0.868`** | **`52.07`** | `22.81` | `2` | `6` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `20` steps | `8` | `12.18` | `12.127` | **`0.545`** | **`32.69`** | `34.53` | `2` | `10` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `768×1344`, `28` steps | `8` | `16.25` | `15.682` | `0.507` | **`30.41`** | `32.66` | `2` | `10` |
| **`vllm-omni`** | **`FLUX.1-dev`** | **`DP=4 × U=2`** *(Intra-Chip)* | `BF16` + `tpu_flash_attn` | `1024×1024`, `28` steps | `8` | `155.60` | **`12.500`** | **`0.479`** | **`28.76`** | `35.35` | **`3`** | **`11`** |
| **`vllm-omni`** | **`FLUX.1-dev`** | **`DP=4 × U=2`** *(Balanced)* | `BF16` + `tpu_flash_attn` | `1024×1024`, `28` steps | `4` | `155.60` | **`8.366`** | **`0.475`** | **`28.49`** | `35.35` | **`3`** | **`11`** |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(Max-DP)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `28` steps | `32` | `12.18` | `42.459` | `0.464` | `27.81` | `34.56` | `3` | `11` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=1 × U=8`** *(Low-Lat)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `20` steps | `1` | `67.52` | **`4.599`** | `0.208` | `12.49` | `32.53` | `5` | `25` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=1 × U=8`** *(Low-Lat)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `28` steps | `1` | `67.52` | **`5.943`** | `0.166` | `9.95` | `32.53` | `7` | `31` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(50-step)* | `BF16` + `tpu_splash_attn` | `1024×1024`, `50` steps | `8` | `12.18` | `28.280` | `0.282` | **`16.91`** | `32.53` | `4` | `18` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=1 × U=8`** *(4MP Low-Lat)* | `BF16` + `tpu_splash_attn` | `2048×2048`, `28` steps | `1` | `18.12` | **`14.056`** | `0.070` | `4.22` | `63.10` | `15` | `72` |
| **`sglang-torchtpu`** | **`FLUX.1-dev`** | **`DP=8 × U=1`** *(4MP Max-DP)* | `BF16` + `tpu_splash_attn` | `2048×2048`, `28` steps | `8` | `504.76` | `70.596` | **`0.112`** | **`6.74`** | `62.74` | `9` | `45` |
| **`vllm-omni`** | **`FLUX.1-dev`** | **`DP=4 × U=2`** *(4MP Balanced)* | `BF16` + `tpu_flash_attn` | `2048×2048`, `28` steps | `4` | `558.82` | **`44.343`** | `0.090` | **`5.38`** | `39.55` | `12` | `56` |

---

## 3. Detailed `vLLM-Omni` (`vllm-torchtpu v0.32.0`) `DP × Ulysses` Scaling Matrix (`FLUX.1-schnell` & `FLUX.1-dev`)

| Model | Topology | Resolution | Steps | `c=1` P50 Latency (s) | `c=1` Step Time (ms) | Peak Concurrency | Peak Throughput (IPM) | Peak Throughput (img/s) | Peak P50 Latency (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `512×512` | `4` | **`0.507`** | `126.7` | `4` | **`373.19`** | **`6.220`** | `0.559` |
| **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `1024×1024` | `1` | **`0.544`** | `544.4` | `4` | **`372.79`** | **`6.213`** | `0.581` |
| **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `1024×1024` | `4` | **`1.407`** | `351.8` | `32` *(c=4: `159.53` IPM)* | **`179.41`** | **`2.990`** | `6.045` *(c=4: `1.446s`)* |
| **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `1344×768` | `4` | **`1.369`** | `342.2` | `4` | **`162.33`** | **`2.705`** | `1.420` |
| **`FLUX.1-schnell`** | **`DP=4 × U=2`** *(Intra-Chip)* | `2048×2048` | `4` | **`7.043`** | `1,760.8` | `8` | **`16.77`** | `0.280` | `10.613` |
| **`FLUX.1-schnell`** | **`DP=2 × U=4`** *(2-Chip)* | `1024×1024` | `4` | `1.521` | `380.3` | `16` *(c=4: `81.60` IPM)* | `82.79` | `1.380` | `6.562` *(c=2: `1.530s`)* |
| **`FLUX.1-schnell`** | **`DP=2 × U=4`** *(2-Chip)* | `1344×768` | `4` | `1.490` | `372.6` | `2` | `77.14` | `1.286` | `1.512` |
| **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(4-Chip)* | `512×512` | `1` | **`0.232`** | `232.2` | `1` | **`252.40`** | `4.207` | `0.232` |
| **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(4-Chip)* | `1024×1024` | `4` | `1.935` | `483.8` | `4` | `32.33` | `0.539` | `7.311` |
| **`FLUX.1-schnell`** | **`DP=1 × U=8`** *(4-Chip)* | `1344×768` | `4` | `1.902` | `475.4` | `1` | `31.01` | `0.517` | `1.902` |
| **`FLUX.1-dev`** | **`DP=4 × U=2`** *(Intra-Chip)* | `1024×1024` | `20` | **`6.029`** | `301.5` | `1` | `9.89` | `0.165` | `6.029` |
| **`FLUX.1-dev`** | **`DP=4 × U=2`** *(Intra-Chip)* | `1024×1024` | `28` | **`8.333`** | **`297.6`** | `8` *(c=4: `28.49` IPM)* | **`28.76`** | **`0.479`** | `12.500` *(c=4: `8.366s`)* |
| **`FLUX.1-dev`** | **`DP=4 × U=2`** *(Intra-Chip)* | `1024×1024` | `50` | **`14.663`** | **`293.3`** | `1` | `4.08` | `0.068` | `14.663` |
| **`FLUX.1-dev`** | **`DP=4 × U=2`** *(Intra-Chip)* | `2048×2048` | `28` | **`44.175`** | `1,577.7` | `4` | **`5.38`** | `0.090` | `44.343` |
| **`FLUX.1-dev`** | **`DP=2 × U=4`** *(2-Chip)* | `1024×1024` | `28` | `9.166` | `327.4` | `4` *(c=2: `13.03` IPM)* | `13.19` | `0.220` | `18.065` *(c=2: `9.166s`)* |
