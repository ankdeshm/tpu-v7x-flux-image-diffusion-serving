#!/usr/bin/env python3
"""Phase 1: Comprehensive TorchTPU FLUX & Image Diffusion Serving Benchmark on TPU v7x-8 (`ank-v7-spot-pool`)
Engine: `sglang-torchtpu` (`us-central1-docker.pkg.dev/diesel-patrol-382622/vllm/sglang-torchtpu-diffusion:v0.2.0`)

Covers:
  1. Models:
     - `black-forest-labs/FLUX.1-schnell` (12B rectified-flow DiT, 1-step & 4-step)
     - `black-forest-labs/FLUX.1-dev` (12B guidance-distilled DiT, 20, 28, 50-step, guidance_scale=3.5)
     - `Tongyi-MAI/Z-Image-Turbo` (6B distilled single-stream DiT, 8-step, guidance_scale=0.0)
  2. Multi-Replica Parallelism Topologies across 8 TensorCores (4 physical v7x chips):
     - `DP=8 x Ulysses=1` (1 TensorCore/replica x 8 replicas/host)
     - `DP=4 x Ulysses=2` (2 TensorCores = 1 physical v7x chip x 4 replicas/host)
     - `DP=2 x Ulysses=4` (4 TensorCores = 2 physical v7x chips x 2 replicas/host)
     - `DP=1 x Ulysses=8` (8 TensorCores = 4 physical v7x chips x 1 replica/host)
  3. Resolutions: `512x512`, `1024x1024` (1MP), `1344x768` (16:9), `768x1344` (9:16), `2048x2048` (4MP UHD)
  4. Precision & Attention Ablations: `BF16` vs `FP8` (`--quantization fp8`), `tpu_splash_attn` (64 MiB VMEM Pallas) vs `torch_sdpa`
  5. Concurrency Scaling: `c = 1, 2, 4, 8, 16, 32`
  6. Spot-Resilient Incremental GCS JSONL Checkpointing (`phase1_sglang_results.jsonl`)
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import math
import os
import re
import signal
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROMPTS = [
    "A hyper-detailed architectural photograph of a futuristic glass conservatory nestled in misty Nordic fjords at sunrise, golden rim light, crisp reflections on calm water, 8k editorial photography.",
    "Close-up studio portrait of an ornate mechanical clockwork owl perched on moss-covered volcanic stone, brass gears, emerald glass eyes, shallow depth of field, dramatic rim lighting.",
    "Aerial view of vibrant bioluminescent waves washing onto a black sand beach in Iceland under the aurora borealis, long exposure, ultra-sharp stars and sea spray.",
    "A cozy sunlit artisan bakery in Kyoto with freshly baked croissants, steam rising from ceramic pour-over coffee, warm cedar woodwork, natural morning window light.",
    "An astronaut botanist tending glowing hydroponic orchids inside a geodesic dome on Mars, red desert canyon visible through panoramic windows, cinematic sci-fi realism.",
    "Macro wildlife photograph of a translucent glass frog resting on a raindrop-covered heliconia leaf in a Costa Rican cloud forest, vivid colors, razor-sharp focus.",
    "A sleek vintage silver streamliner train crossing a stone viaduct over an autumn alpine valley, turquoise glacial river below, crisp mountain air, medium format film look.",
    "Isometric cutaway illustration of a bustling subterranean solarpunk library with spiral staircases, hanging gardens, warm lanterns, and stained-glass skylights.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{utc_now()}][sglang-flux-v7x] {msg}", flush=True)


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def gcs_upload_bytes(bucket: str, obj: str, data: bytes, content_type: str = "application/json") -> None:
    for attempt in range(5):
        try:
            tok = vm_access_token()
            url = (
                f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
                f"?uploadType=media&name={urllib.parse.quote(obj, safe='')}"
            )
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": content_type},
            )
            with urllib.request.urlopen(req, timeout=120):
                return
        except Exception as exc:
            if attempt == 4:
                log(f"WARNING: GCS upload failed for {obj}: {exc}")
                return
            time.sleep(2 * (attempt + 1))


def gcs_download_bytes(bucket: str, obj: str) -> bytes | None:
    for attempt in range(4):
        try:
            tok = vm_access_token()
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == 3:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def gcs_list_prefix(bucket: str, prefix: str) -> list[dict[str, Any]]:
    tok = vm_access_token()
    items: list[dict[str, Any]] = []
    pt: str | None = None
    while True:
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={urllib.parse.quote(prefix, safe='')}"
        if pt:
            url += f"&pageToken={urllib.parse.quote(pt, safe='')}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        items.extend(d.get("items", []))
        pt = d.get("nextPageToken")
        if not pt:
            break
    return items


def fast_gcs_sync_dir(bucket: str, gcs_prefix: str, local_dir: Path) -> int:
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"
    items = gcs_list_prefix(bucket, gcs_prefix)
    blobs = [
        (it["name"][len(gcs_prefix) :], int(it["size"]))
        for it in items
        if not it["name"].endswith("/")
        and "/.cache/" not in it["name"]
        and not it["name"].endswith(".ready")
        and not it["name"].endswith("_FLUX_SERVE_STAGE_DONE.txt")
    ]
    if not blobs:
        return 0
    local_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, Path, int, int]] = []
    chunk = 512 * 1024 * 1024
    for rel, sz in blobs:
        dst = local_dir / rel
        if dst.exists() and dst.stat().st_size == sz:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            if sz > 0:
                f.truncate(sz)
        if sz == 0:
            continue
        for s in range(0, sz, chunk):
            tasks.append((rel, dst, s, min(s + chunk, sz)))
    if not tasks:
        return len(blobs)

    def dl_chunk(t: tuple[str, Path, int, int]) -> int:
        rel, dst, s, e = t
        for r in range(5):
            try:
                tok = vm_access_token()
                u = (
                    f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
                    f"{urllib.parse.quote(gcs_prefix + rel, safe='')}?alt=media"
                )
                req = urllib.request.Request(
                    u, headers={"Authorization": f"Bearer {tok}", "Range": f"bytes={s}-{e - 1}"}
                )
                fd = os.open(str(dst), os.O_WRONLY)
                try:
                    pos = s
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        while True:
                            buf = resp.read(16 * 1024 * 1024)
                            if not buf:
                                break
                            os.pwrite(fd, buf, pos)
                            pos += len(buf)
                finally:
                    os.close(fd)
                return e - s
            except Exception:
                if r == 4:
                    raise
                time.sleep(1.5 * (r + 1))
        return 0

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        total_b = sum(pool.map(dl_chunk, tasks))
    dt = max(time.time() - t0, 1e-3)
    log(f"Hydrated {total_b / 1e9:.2f} GB from gs://{bucket}/{gcs_prefix} -> {local_dir} in {dt:.1f}s ({(total_b / 1e9) / dt:.2f} GB/s)")
    return len(blobs)


def ensure_model_staged(bucket: str, repo_id: str, cache_root: Path) -> str:
    from huggingface_hub import snapshot_download

    short = repo_id.split("/")[-1]
    local_dir = cache_root / short
    marker = local_dir / ".complete_verified"
    if marker.exists():
        return str(local_dir)

    t0 = time.time()
    log(f"Staging {repo_id} into {local_dir} (checking regional GCS mirror first)...")
    try:
        fast_gcs_sync_dir(bucket, f"models/{repo_id}/", local_dir)
    except Exception as exc:
        log(f"GCS pre-sync warning for {repo_id}: {exc}")

    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    ignore = [
        "*.mp4",
        "*.mov",
        "*.gif",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.webp",
        "*.pdf",
        "flux1-schnell.safetensors",
        "flux1-dev.safetensors",
        "ae.safetensors",
    ]
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=ignore,
        token=os.environ.get("HF_TOKEN") or None,
        max_workers=16,
    )
    marker.write_text(utc_now() + "\n")
    sz_gb = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file() and "/.cache/" not in p.as_posix()) / 1e9
    log(f"Model {repo_id} ready at {local_dir} ({sz_gb:.2f} GB in {time.time() - t0:.1f}s)")
    return str(local_dir)


class LibtpuHbmMonitor:
    """Polls libtpu runtime metrics endpoints (`localhost:8431..8438/metrics`) and /proc to capture Peak HBM per TensorCore."""

    def __init__(self) -> None:
        ports_str = os.environ.get("TPU_RUNTIME_METRICS_PORTS", "8431,8432,8433,8434,8435,8436,8437,8438")
        self.ports = [int(p.strip()) for p in ports_str.split(",") if p.strip().isdigit()]
        self.peak_bytes_per_core: dict[int, float] = {}
        self.discovered_metric_lines: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._logged_sample = False

    def _poll_once(self) -> None:
        for idx, port in enumerate(self.ports):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=0.8) as r:
                    txt = r.read().decode("utf-8", errors="ignore")
                if not self._logged_sample and txt.strip():
                    self._logged_sample = True
                    sample = [ln for ln in txt.splitlines() if not ln.startswith("#")][:25]
                    self.discovered_metric_lines = sample
                    log(f"Discovered libtpu metrics on port {port} (sample): {sample[:6]}")
                for line in txt.splitlines():
                    if line.startswith("#"):
                        continue
                    low = line.lower()
                    if ("hbm" in low or "memory" in low) and ("usage" in low or "used" in low or "allocated" in low or "bytes" in low):
                        if "total" in low or "limit" in low or "capacity" in low:
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                val = float(parts[-1])
                                if val > 1024 * 1024:  # > 1 MiB
                                    prev = self.peak_bytes_per_core.get(idx, 0.0)
                                    if val > prev:
                                        self.peak_bytes_per_core[idx] = val
                            except ValueError:
                                pass
            except Exception:
                pass

    def start(self) -> None:
        self._stop.clear()
        self.peak_bytes_per_core.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                self._poll_once()
                self._stop.wait(1.5)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def snapshot_peak_gib(self) -> float | None:
        self._poll_once()
        if not self.peak_bytes_per_core:
            return None
        return round(max(self.peak_bytes_per_core.values()) / (1024.0**3), 2)

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return self.snapshot_peak_gib()


def parse_hbm_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    try:
        txt = log_path.read_text(errors="ignore")
    except Exception:
        return None
    vals_gib: list[float] = []
    for m in re.finditer(r"(?:memory|hbm|allocated|used)[^\n]{0,40}?([0-9]+\.[0-9]+)\s*(GiB|GB|MiB|MB)", txt, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).upper()
        if unit in ("MIB", "MB"):
            val /= 1024.0
        if 1.0 <= val <= 96.0:
            vals_gib.append(val)
    return round(max(vals_gib), 2) if vals_gib else None


def stop_server_cleanly(proc: subprocess.Popen[Any] | None = None) -> None:
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        t_wait = time.time()
        while proc.poll() is None and time.time() - t_wait < 6.0:
            time.sleep(0.3)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    subprocess.run(
        [
            "bash",
            "-lc",
            "pkill -15 -f 'sglang serve' 2>/dev/null || true; sleep 1; "
            "pkill -9 -f 'sglang serve' 2>/dev/null || true; "
            "pkill -9 -f 'sglang.multimodal_gen' 2>/dev/null || true; "
            "rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*/*.lock 2>/dev/null || true",
        ],
        check=False,
    )
    time.sleep(2.5)


def start_sglang_server(
    model_path: str,
    port: int,
    dp_size: int,
    ulysses_degree: int,
    attention_backend: str,
    quantization: str | None,
    log_path: Path,
) -> tuple[subprocess.Popen[Any], float]:
    stop_server_cleanly(None)
    Path("/dev/shm/xla_cache").mkdir(parents=True, exist_ok=True)
    Path("/dev/shm/sglang_outputs").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PJRT_DEVICE": "TPU",
            "TPU_SKIP_MDS_QUERY": "1",
            "TPU_PREMAPPED_BUFFER_SIZE": str(512 * 1024**2),
            "JAX_COMPILATION_CACHE_DIR": "/dev/shm/xla_cache",
            "SGLANG_TPU_SPLASH_VARLEN": "1",
        }
    )

    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--num-gpus",
        "8",
        "--dp-size",
        str(dp_size),
        "--ulysses-degree",
        str(ulysses_degree),
        "--attention-backend",
        attention_backend,
        "--warmup-mode",
        "off",
        "--output-path",
        "/dev/shm/sglang_outputs",
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])

    log(f"Launching SGLang server: {' '.join(cmd)}")
    lf = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

    while time.time() - t0 < 900:
        if proc.poll() is not None:
            lf.flush()
            tail = log_path.read_text(errors="ignore")[-3000:]
            raise RuntimeError(f"sglang serve exited early (rc={proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3.0) as r:
                if r.status == 200:
                    boot_s = round(time.time() - t0, 2)
                    log(f"SGLang ready on :{port} in {boot_s}s (dp={dp_size}, ulysses={ulysses_degree}, attn={attention_backend}, quant={quantization or 'bf16'})")
                    return proc, boot_s
        except Exception:
            pass
        time.sleep(2.0)

    stop_server_cleanly(proc)
    tail = log_path.read_text(errors="ignore")[-3000:]
    raise TimeoutError(f"Timed out waiting for SGLang server on :{port}:\n{tail}")


def generate_one_image(
    endpoint: str,
    prompt: str,
    size: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    exp_w, exp_h = (int(x) for x in size.split("x"))
    payload: dict[str, Any] = {
        "prompt": prompt,
        "size": size,
        "n": 1,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "generator_device": "cpu",
        "response_format": "b64_json",
        "output_format": "png",
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/v1/images/generations",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw_bytes = resp.read()
        e2e_s = time.perf_counter() - t0
        data = json.loads(raw_bytes.decode("utf-8"))
        items = data.get("data") or []
        if not items or not items[0].get("b64_json"):
            return {"ok": False, "e2e_s": round(e2e_s, 4), "error": f"Missing b64_json in response keys={list(data.keys())}"}
        png_bytes = base64.b64decode(items[0]["b64_json"])
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        arr = np.asarray(img, dtype=np.float32)
        finite = bool(np.isfinite(arr).all())
        p_mean = float(np.mean(arr))
        p_std = float(np.std(arr))
        valid = (w == exp_w and h == exp_h and finite and p_std > 4.0)
        # Clean up saved output file in /dev/shm/sglang_outputs if returned to keep /dev/shm tidy
        fp = items[0].get("file_path")
        if fp and os.path.exists(fp):
            try:
                os.remove(fp)
            except Exception:
                pass
        return {
            "ok": valid,
            "e2e_s": round(e2e_s, 4),
            "server_inference_s": data.get("inference_time_s"),
            "server_peak_memory_mb": data.get("peak_memory_mb"),
            "width": w,
            "height": h,
            "pixel_mean": round(p_mean, 2),
            "pixel_std": round(p_std, 2),
            "png_bytes": png_bytes,
            "error": None if valid else f"Invalid image dims=({w},{h}) expected=({exp_w},{exp_h}) finite={finite} std={p_std:.2f}",
        }
    except urllib.error.HTTPError as he:
        err_txt = ""
        try:
            err_txt = he.read().decode("utf-8", errors="ignore")[:500]
        except Exception:
            pass
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": f"HTTP {he.code}: {err_txt}"}
    except Exception as exc:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": repr(exc)}


def estimate_theoretic_hbm_gib(model_short: str, ulysses_degree: int, precision: str, width: int, height: int) -> float:
    """Fallback analytic HBM estimate ONLY if both server_peak_memory_mb and libtpu /metrics return None; flagged clearly."""
    # We record separate fields: `measured_peak_hbm_gib` and `analytic_est_hbm_gib`
    if "Z-Image" in model_short:
        dit_gb, enc_gb, vae_gb = (6.1 if precision == "FP8" else 12.3), 8.2, 0.35
    else:
        dit_gb, enc_gb, vae_gb = (11.9 if precision == "FP8" else 23.8), 9.8, 0.35
    # Under Ulysses SP in SGLang/vLLM-Omni, weights are replicated per rank unless TP/FSDP is enabled; activations shard by ulysses_degree
    mp = (width * height) / (1024.0 * 1024.0)
    act_gb = (2.8 * mp) / max(1, ulysses_degree)
    return round(dit_gb + enc_gb + vae_gb + act_gb, 2)


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def build_benchmark_groups() -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    # 1. FLUX.1-schnell | DP=8 x Ulysses=1 (1 TensorCore/replica x 8 replicas) | BF16 | tpu_splash_attn
    groups.append(
        {
            "group_id": "sglang_flux1_schnell_dp8_u1_bf16_splash",
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "model_short": "FLUX.1-schnell",
            "dp_size": 8,
            "ulysses_degree": 1,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 0.0,
            "points": [
                {"size": "512x512", "steps": 1, "c": 1, "n": 4},
                {"size": "512x512", "steps": 1, "c": 8, "n": 16},
                {"size": "512x512", "steps": 4, "c": 1, "n": 4, "save_sample": True},
                {"size": "512x512", "steps": 4, "c": 8, "n": 16},
                {"size": "1024x1024", "steps": 1, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 1, "c": 8, "n": 16},
                {"size": "1024x1024", "steps": 1, "c": 16, "n": 32},
                {"size": "1024x1024", "steps": 4, "c": 1, "n": 4, "save_sample": True},
                {"size": "1024x1024", "steps": 4, "c": 2, "n": 8},
                {"size": "1024x1024", "steps": 4, "c": 4, "n": 8},
                {"size": "1024x1024", "steps": 4, "c": 8, "n": 16},
                {"size": "1024x1024", "steps": 4, "c": 16, "n": 32},
                {"size": "1024x1024", "steps": 4, "c": 32, "n": 32},
                {"size": "1344x768", "steps": 4, "c": 1, "n": 4, "save_sample": True},
                {"size": "1344x768", "steps": 4, "c": 8, "n": 16},
                {"size": "768x1344", "steps": 4, "c": 1, "n": 4},
                {"size": "768x1344", "steps": 4, "c": 8, "n": 16},
                {"size": "2048x2048", "steps": 4, "c": 1, "n": 3, "save_sample": True},
                {"size": "2048x2048", "steps": 4, "c": 8, "n": 8},
            ],
        }
    )

    # 2. FLUX.1-schnell | DP=4 x Ulysses=2 (1 physical v7x chip = 2 TensorCores/replica x 4 replicas) | BF16 | tpu_splash_attn
    groups.append(
        {
            "group_id": "sglang_flux1_schnell_dp4_u2_bf16_splash",
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "model_short": "FLUX.1-schnell",
            "dp_size": 4,
            "ulysses_degree": 2,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 0.0,
            "points": [
                {"size": "512x512", "steps": 4, "c": 1, "n": 4},
                {"size": "512x512", "steps": 4, "c": 4, "n": 8},
                {"size": "1024x1024", "steps": 1, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 1, "c": 4, "n": 12},
                {"size": "1024x1024", "steps": 4, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 4, "c": 2, "n": 8},
                {"size": "1024x1024", "steps": 4, "c": 4, "n": 12},
                {"size": "1024x1024", "steps": 4, "c": 8, "n": 16},
                {"size": "1024x1024", "steps": 4, "c": 16, "n": 16},
                {"size": "1024x1024", "steps": 4, "c": 32, "n": 32},
                {"size": "1344x768", "steps": 4, "c": 1, "n": 4},
                {"size": "1344x768", "steps": 4, "c": 4, "n": 8},
                {"size": "768x1344", "steps": 4, "c": 1, "n": 4},
                {"size": "768x1344", "steps": 4, "c": 4, "n": 8},
                {"size": "2048x2048", "steps": 4, "c": 1, "n": 3},
                {"size": "2048x2048", "steps": 4, "c": 4, "n": 8},
                {"size": "2048x2048", "steps": 4, "c": 8, "n": 8},
            ],
        }
    )

    # 3. FLUX.1-schnell | DP=2 x Ulysses=4 (2 physical v7x chips = 4 TensorCores/replica x 2 replicas) | BF16 | tpu_splash_attn
    groups.append(
        {
            "group_id": "sglang_flux1_schnell_dp2_u4_bf16_splash",
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "model_short": "FLUX.1-schnell",
            "dp_size": 2,
            "ulysses_degree": 4,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 0.0,
            "points": [
                {"size": "512x512", "steps": 4, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 1, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 4, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 4, "c": 2, "n": 8},
                {"size": "1024x1024", "steps": 4, "c": 4, "n": 8},
                {"size": "1024x1024", "steps": 4, "c": 8, "n": 16},
                {"size": "1024x1024", "steps": 4, "c": 16, "n": 16},
                {"size": "1344x768", "steps": 4, "c": 1, "n": 4},
                {"size": "1344x768", "steps": 4, "c": 2, "n": 6},
                {"size": "768x1344", "steps": 4, "c": 1, "n": 4},
                {"size": "2048x2048", "steps": 4, "c": 1, "n": 3},
                {"size": "2048x2048", "steps": 4, "c": 2, "n": 6},
                {"size": "2048x2048", "steps": 4, "c": 4, "n": 8},
            ],
        }
    )

    # 4. FLUX.1-schnell | DP=1 x Ulysses=8 (all 4 physical v7x chips = 8 TensorCores/replica) | BF16 | tpu_splash_attn
    groups.append(
        {
            "group_id": "sglang_flux1_schnell_dp1_u8_bf16_splash",
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "model_short": "FLUX.1-schnell",
            "dp_size": 1,
            "ulysses_degree": 8,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 0.0,
            "points": [
                {"size": "512x512", "steps": 1, "c": 1, "n": 4},
                {"size": "512x512", "steps": 4, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 1, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 4, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 4, "c": 2, "n": 6},
                {"size": "1024x1024", "steps": 4, "c": 4, "n": 8},
                {"size": "1024x1024", "steps": 4, "c": 8, "n": 8},
                {"size": "1344x768", "steps": 4, "c": 1, "n": 4},
                {"size": "768x1344", "steps": 4, "c": 1, "n": 4},
                {"size": "2048x2048", "steps": 1, "c": 1, "n": 3},
                {"size": "2048x2048", "steps": 4, "c": 1, "n": 4},
                {"size": "2048x2048", "steps": 4, "c": 2, "n": 4},
                {"size": "2048x2048", "steps": 4, "c": 4, "n": 8},
            ],
        }
    )

    # 5a. Attention Ablation: Torch SDPA vs Pallas Splash Attention (DP=8 x Ulysses=1)
    groups.append(
        {
            "group_id": "sglang_flux1_schnell_dp8_u1_bf16_sdpa",
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "model_short": "FLUX.1-schnell",
            "dp_size": 8,
            "ulysses_degree": 1,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "torch_sdpa",
            "guidance_scale": 0.0,
            "points": [
                {"size": "1024x1024", "steps": 4, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 4, "c": 8, "n": 16},
                {"size": "2048x2048", "steps": 4, "c": 1, "n": 3},
                {"size": "2048x2048", "steps": 4, "c": 8, "n": 8},
            ],
        }
    )

    # 5b. Attention Ablation: Torch SDPA vs Pallas Splash Attention (DP=1 x Ulysses=8)
    groups.append(
        {
            "group_id": "sglang_flux1_schnell_dp1_u8_bf16_sdpa",
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "model_short": "FLUX.1-schnell",
            "dp_size": 1,
            "ulysses_degree": 8,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "torch_sdpa",
            "guidance_scale": 0.0,
            "points": [
                {"size": "1024x1024", "steps": 4, "c": 1, "n": 4},
                {"size": "2048x2048", "steps": 4, "c": 1, "n": 3},
            ],
        }
    )

    # 6a. Precision Ablation: Online FP8 (`--quantization fp8`) on DP=8 x Ulysses=1 & DP=1 x Ulysses=8
    groups.append(
        {
            "group_id": "sglang_flux1_schnell_dp8_u1_fp8_splash",
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "model_short": "FLUX.1-schnell",
            "dp_size": 8,
            "ulysses_degree": 1,
            "precision": "FP8",
            "quantization": "fp8",
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 0.0,
            "optional": True,
            "points": [
                {"size": "1024x1024", "steps": 4, "c": 1, "n": 4, "save_sample": True},
                {"size": "1024x1024", "steps": 4, "c": 8, "n": 16},
                {"size": "1024x1024", "steps": 4, "c": 16, "n": 32},
                {"size": "2048x2048", "steps": 4, "c": 1, "n": 3},
                {"size": "2048x2048", "steps": 4, "c": 8, "n": 8},
            ],
        }
    )
    groups.append(
        {
            "group_id": "sglang_flux1_schnell_dp1_u8_fp8_splash",
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "model_short": "FLUX.1-schnell",
            "dp_size": 1,
            "ulysses_degree": 8,
            "precision": "FP8",
            "quantization": "fp8",
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 0.0,
            "optional": True,
            "points": [
                {"size": "1024x1024", "steps": 4, "c": 1, "n": 4},
                {"size": "2048x2048", "steps": 4, "c": 1, "n": 3},
            ],
        }
    )

    # 7. FLUX.1-dev (12B guidance-distilled DiT, guidance_scale=3.5, 20/28/50 steps) across all 4 DP x Ulysses topologies
    groups.append(
        {
            "group_id": "sglang_flux1_dev_dp8_u1_bf16_splash",
            "repo_id": "black-forest-labs/FLUX.1-dev",
            "model_short": "FLUX.1-dev",
            "dp_size": 8,
            "ulysses_degree": 1,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 3.5,
            "points": [
                {"size": "512x512", "steps": 28, "c": 1, "n": 3},
                {"size": "512x512", "steps": 28, "c": 8, "n": 8},
                {"size": "1024x1024", "steps": 20, "c": 1, "n": 3},
                {"size": "1024x1024", "steps": 20, "c": 8, "n": 16},
                {"size": "1024x1024", "steps": 28, "c": 1, "n": 3, "save_sample": True},
                {"size": "1024x1024", "steps": 28, "c": 4, "n": 8},
                {"size": "1024x1024", "steps": 28, "c": 8, "n": 16},
                {"size": "1024x1024", "steps": 28, "c": 16, "n": 16},
                {"size": "1024x1024", "steps": 28, "c": 32, "n": 32},
                {"size": "1024x1024", "steps": 50, "c": 1, "n": 2},
                {"size": "1024x1024", "steps": 50, "c": 8, "n": 8},
                {"size": "1344x768", "steps": 28, "c": 1, "n": 3},
                {"size": "1344x768", "steps": 28, "c": 8, "n": 8},
                {"size": "768x1344", "steps": 28, "c": 1, "n": 3},
                {"size": "768x1344", "steps": 28, "c": 8, "n": 8},
                {"size": "2048x2048", "steps": 20, "c": 1, "n": 2},
                {"size": "2048x2048", "steps": 28, "c": 1, "n": 2, "save_sample": True},
                {"size": "2048x2048", "steps": 28, "c": 8, "n": 8},
            ],
        }
    )
    groups.append(
        {
            "group_id": "sglang_flux1_dev_dp4_u2_bf16_splash",
            "repo_id": "black-forest-labs/FLUX.1-dev",
            "model_short": "FLUX.1-dev",
            "dp_size": 4,
            "ulysses_degree": 2,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 3.5,
            "points": [
                {"size": "1024x1024", "steps": 20, "c": 1, "n": 3},
                {"size": "1024x1024", "steps": 28, "c": 1, "n": 3},
                {"size": "1024x1024", "steps": 28, "c": 4, "n": 8},
                {"size": "1024x1024", "steps": 28, "c": 8, "n": 8},
                {"size": "1024x1024", "steps": 28, "c": 16, "n": 16},
                {"size": "1024x1024", "steps": 50, "c": 1, "n": 2},
                {"size": "2048x2048", "steps": 28, "c": 1, "n": 2},
                {"size": "2048x2048", "steps": 28, "c": 4, "n": 4},
            ],
        }
    )
    groups.append(
        {
            "group_id": "sglang_flux1_dev_dp2_u4_bf16_splash",
            "repo_id": "black-forest-labs/FLUX.1-dev",
            "model_short": "FLUX.1-dev",
            "dp_size": 2,
            "ulysses_degree": 4,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 3.5,
            "points": [
                {"size": "1024x1024", "steps": 28, "c": 1, "n": 3},
                {"size": "1024x1024", "steps": 28, "c": 2, "n": 6},
                {"size": "1024x1024", "steps": 28, "c": 4, "n": 8},
                {"size": "2048x2048", "steps": 28, "c": 1, "n": 2},
                {"size": "2048x2048", "steps": 28, "c": 2, "n": 4},
            ],
        }
    )
    groups.append(
        {
            "group_id": "sglang_flux1_dev_dp1_u8_bf16_splash",
            "repo_id": "black-forest-labs/FLUX.1-dev",
            "model_short": "FLUX.1-dev",
            "dp_size": 1,
            "ulysses_degree": 8,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 3.5,
            "points": [
                {"size": "512x512", "steps": 28, "c": 1, "n": 3},
                {"size": "1024x1024", "steps": 20, "c": 1, "n": 3},
                {"size": "1024x1024", "steps": 28, "c": 1, "n": 3},
                {"size": "1024x1024", "steps": 28, "c": 2, "n": 4},
                {"size": "1024x1024", "steps": 28, "c": 4, "n": 4},
                {"size": "1024x1024", "steps": 50, "c": 1, "n": 3},
                {"size": "1344x768", "steps": 28, "c": 1, "n": 3},
                {"size": "768x1344", "steps": 28, "c": 1, "n": 3},
                {"size": "2048x2048", "steps": 20, "c": 1, "n": 3},
                {"size": "2048x2048", "steps": 28, "c": 1, "n": 3},
                {"size": "2048x2048", "steps": 50, "c": 1, "n": 2},
            ],
        }
    )

    # 8. Tongyi-MAI/Z-Image-Turbo (6B single-stream DiT, 8-step, guidance_scale=0.0) comparison
    groups.append(
        {
            "group_id": "sglang_zimage_turbo_dp8_u1_bf16_splash",
            "repo_id": "Tongyi-MAI/Z-Image-Turbo",
            "model_short": "Z-Image-Turbo",
            "dp_size": 8,
            "ulysses_degree": 1,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 0.0,
            "points": [
                {"size": "512x512", "steps": 8, "c": 1, "n": 4},
                {"size": "512x512", "steps": 8, "c": 8, "n": 16},
                {"size": "1024x1024", "steps": 8, "c": 1, "n": 4, "save_sample": True},
                {"size": "1024x1024", "steps": 8, "c": 8, "n": 16},
                {"size": "1024x1024", "steps": 8, "c": 16, "n": 32},
                {"size": "1024x1024", "steps": 8, "c": 32, "n": 32},
                {"size": "1344x768", "steps": 8, "c": 1, "n": 4},
                {"size": "1344x768", "steps": 8, "c": 8, "n": 16},
                {"size": "2048x2048", "steps": 8, "c": 1, "n": 3},
                {"size": "2048x2048", "steps": 8, "c": 8, "n": 8},
            ],
        }
    )
    groups.append(
        {
            "group_id": "sglang_zimage_turbo_dp1_u8_bf16_splash",
            "repo_id": "Tongyi-MAI/Z-Image-Turbo",
            "model_short": "Z-Image-Turbo",
            "dp_size": 1,
            "ulysses_degree": 8,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 0.0,
            "points": [
                {"size": "512x512", "steps": 8, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 8, "c": 1, "n": 4},
                {"size": "1344x768", "steps": 8, "c": 1, "n": 4},
                {"size": "2048x2048", "steps": 8, "c": 1, "n": 3},
            ],
        }
    )

    return groups


def make_point_key(grp: dict[str, Any], pt: dict[str, Any]) -> str:
    return (
        f"sglang|{grp['model_short']}|dp{grp['dp_size']}_u{grp['ulysses_degree']}"
        f"|{grp['precision']}|{grp['attention_backend']}|{pt['size']}|s{pt['steps']}|c{pt['c']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-bucket", default="ank-usc1-bucket-314837540096")
    ap.add_argument("--gcs-prefix", default="flux-serving-results")
    ap.add_argument("--port", type=int, default=30180)
    args = ap.parse_args()

    cache_root = Path("/dev/shm/hf_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    local_jsonl = Path("/dev/shm/phase1_sglang_results.jsonl")
    gcs_jsonl_obj = f"{args.gcs_prefix}/phase1_sglang_results.jsonl"
    gcs_summary_obj = f"{args.gcs_prefix}/phase1_sglang_summary.json"

    # Load existing checkpointed results from GCS so Spot preemption resumes seamlessly
    completed: dict[str, dict[str, Any]] = {}
    existing_raw = gcs_download_bytes(args.gcs_bucket, gcs_jsonl_obj)
    if existing_raw:
        local_jsonl.write_bytes(existing_raw)
        for line in existing_raw.decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pk = rec.get("point_key")
                if pk and (rec.get("status") == "ok" or rec.get("status") == "unsupported"):
                    completed[pk] = rec
            except Exception:
                pass
        log(f"Loaded {len(completed)} already-completed benchmark points from gs://{args.gcs_bucket}/{gcs_jsonl_obj}")

    def append_record(rec: dict[str, Any]) -> None:
        completed[rec["point_key"]] = rec
        with open(local_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        gcs_upload_bytes(args.gcs_bucket, gcs_jsonl_obj, local_jsonl.read_bytes(), content_type="application/x-ndjson")
        summary_payload = {
            "updated_at": utc_now(),
            "engine": "sglang-torchtpu",
            "image": "us-central1-docker.pkg.dev/diesel-patrol-382622/vllm/sglang-torchtpu-diffusion:v0.2.0",
            "node": os.uname().nodename,
            "total_points_recorded": len(completed),
            "results": list(completed.values()),
        }
        gcs_upload_bytes(args.gcs_bucket, gcs_summary_obj, json.dumps(summary_payload, indent=2).encode("utf-8"))

    groups = build_benchmark_groups()
    total_pts = sum(len(g["points"]) for g in groups)
    log(f"Phase 1 SGLang benchmark plan: {len(groups)} server groups, {total_pts} total points ({len(completed)} already done).")

    for g_idx, grp in enumerate(groups, 1):
        pending_pts = [pt for pt in grp["points"] if make_point_key(grp, pt) not in completed]
        if not pending_pts:
            log(f"[{g_idx}/{len(groups)}] Skipping group {grp['group_id']} (all {len(grp['points'])} points completed)")
            continue

        log(f"\n==================== [{g_idx}/{len(groups)}] Group: {grp['group_id']} ({len(pending_pts)}/{len(grp['points'])} pending) ====================")
        model_dir = ensure_model_staged(args.gcs_bucket, grp["repo_id"], cache_root)
        log_path = Path(f"/tmp/{grp['group_id']}.log")
        hbm_mon = LibtpuHbmMonitor()
        hbm_mon.start()
        proc: subprocess.Popen[Any] | None = None

        try:
            proc, boot_s = start_sglang_server(
                model_path=model_dir,
                port=args.port,
                dp_size=grp["dp_size"],
                ulysses_degree=grp["ulysses_degree"],
                attention_backend=grp["attention_backend"],
                quantization=grp["quantization"],
                log_path=log_path,
            )
            endpoint = f"http://127.0.0.1:{args.port}"

            # Track which (size) shapes have been cold-compiled and warmed across all DP replicas on this server
            warmed_shapes: dict[str, float] = {}

            for pt in pending_pts:
                pk = make_point_key(grp, pt)
                size = pt["size"]
                w, h = (int(x) for x in size.split("x"))
                steps = pt["steps"]
                conc = pt["c"]
                n_req = pt["n"]
                dp_size = grp["dp_size"]
                gs = grp["guidance_scale"]

                # 1) Cold compile + first inference measurement (and warm all DP replicas on this resolution!)
                if size not in warmed_shapes:
                    log(f"  [COLD+WARMUP] {pk}: running 1st cold request ({size}, steps={steps})...")
                    cold_res = generate_one_image(endpoint, PROMPTS[0], size, steps, gs, seed=1001)
                    if not cold_res["ok"]:
                        log(f"  [ERROR] Cold request failed for {pk}: {cold_res.get('error')}")
                        # Upload server log tail for debugging
                        gcs_upload_bytes(
                            args.gcs_bucket,
                            f"{args.gcs_prefix}/logs/{grp['group_id']}_error.log",
                            log_path.read_bytes()[-25000:] if log_path.exists() else b"",
                            content_type="text/plain",
                        )
                        status_tag = "unsupported" if grp.get("optional") else "error"
                        append_record(
                            {
                                "point_key": pk,
                                "status": status_tag,
                                "engine": "sglang-torchtpu",
                                "group_id": grp["group_id"],
                                "model": grp["model_short"],
                                "dp_size": dp_size,
                                "ulysses_degree": grp["ulysses_degree"],
                                "precision": grp["precision"],
                                "attention_backend": grp["attention_backend"],
                                "size": size,
                                "steps": steps,
                                "concurrency": conc,
                                "error": cold_res.get("error"),
                                "timestamp": utc_now(),
                            }
                        )
                        if grp.get("optional"):
                            break
                        continue

                    cold_s = cold_res["e2e_s"]
                    warmed_shapes[size] = cold_s
                    log(f"  [COLD OK] {size} steps={steps}: cold_compile_infer_s={cold_s:.2f}s (pixel_std={cold_res['pixel_std']})")

                    # Warm remaining DP replicas for this exact resolution so multi-replica runs have 0 compile stragglers
                    if dp_size > 1:
                        log(f"  [DP WARMUP] Warming all {dp_size} DP replicas concurrently on {size} (steps={min(steps, 4)})...")
                        warm_steps = min(steps, 4)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=dp_size) as wp:
                            futs = [
                                wp.submit(generate_one_image, endpoint, PROMPTS[i % len(PROMPTS)], size, warm_steps, gs, 1100 + i)
                                for i in range(dp_size * 2)
                            ]
                            for f in futs:
                                f.result()
                else:
                    cold_s = warmed_shapes[size]
                    # Also do 1 quick warm request if step count changed
                    _ = generate_one_image(endpoint, PROMPTS[0], size, steps, gs, seed=1299)

                # 2) Timed Steady-State Warm Benchmark Wave
                log(f"  [BENCH] {pk}: running n={n_req} warm requests at concurrency c={conc}...")
                t_wave = time.perf_counter()
                if conc == 1:
                    outs = [
                        generate_one_image(endpoint, PROMPTS[i % len(PROMPTS)], size, steps, gs, seed=4000 + i)
                        for i in range(n_req)
                    ]
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                        futs = [
                            pool.submit(
                                generate_one_image,
                                endpoint,
                                PROMPTS[i % len(PROMPTS)],
                                size,
                                steps,
                                gs,
                                5000 + i,
                            )
                            for i in range(n_req)
                        ]
                        outs = [f.result() for f in futs]
                wall_s = time.perf_counter() - t_wave

                oks = [o for o in outs if o.get("ok")]
                if not oks:
                    err_msg = outs[0].get("error") if outs else "no outputs"
                    log(f"  [ERROR] {pk} 0/{n_req} succeeded: {err_msg}")
                    append_record(
                        {
                            "point_key": pk,
                            "status": "error",
                            "engine": "sglang-torchtpu",
                            "group_id": grp["group_id"],
                            "model": grp["model_short"],
                            "dp_size": dp_size,
                            "ulysses_degree": grp["ulysses_degree"],
                            "precision": grp["precision"],
                            "attention_backend": grp["attention_backend"],
                            "size": size,
                            "steps": steps,
                            "concurrency": conc,
                            "error": err_msg,
                            "timestamp": utc_now(),
                        }
                    )
                    continue

                # Save sample PNG to GCS if flagged
                if pt.get("save_sample") and oks[0].get("png_bytes"):
                    sample_name = (
                        f"sglang_{grp['model_short'].lower().replace('.', '_')}_"
                        f"dp{dp_size}u{grp['ulysses_degree']}_{grp['precision'].lower()}_{size}_{steps}step.png"
                    )
                    gcs_upload_bytes(
                        args.gcs_bucket,
                        f"{args.gcs_prefix}/samples/{sample_name}",
                        oks[0]["png_bytes"],
                        content_type="image/png",
                    )
                    log(f"  [SAMPLE PNG] Uploaded gs://{args.gcs_bucket}/{args.gcs_prefix}/samples/{sample_name}")

                lats = [o["e2e_s"] for o in oks]
                srv_lats = [float(o["server_inference_s"]) for o in oks if o.get("server_inference_s") is not None]
                srv_mems = [float(o["server_peak_memory_mb"]) / 1024.0 for o in oks if o.get("server_peak_memory_mb")]

                p50_s = round(statistics.median(lats), 4)
                mean_s = round(statistics.mean(lats), 4)
                p90_s = round(percentile(lats, 90.0), 4)
                min_s = round(min(lats), 4)
                srv_p50_s = round(statistics.median(srv_lats), 4) if srv_lats else None

                img_per_s = round(len(oks) / wall_s, 4)
                ipm = round(img_per_s * 60.0, 2)
                ms_per_step = round((p50_s * 1000.0) / steps, 2)

                libtpu_hbm = hbm_mon.snapshot_peak_gib()
                log_hbm = parse_hbm_from_log(log_path)
                srv_hbm = round(max(srv_mems), 2) if srv_mems else None
                measured_hbm = srv_hbm or libtpu_hbm or log_hbm
                est_hbm = estimate_theoretic_hbm_gib(grp["model_short"], grp["ulysses_degree"], grp["precision"], w, h)

                rec = {
                    "point_key": pk,
                    "status": "ok",
                    "engine": "sglang-torchtpu",
                    "group_id": grp["group_id"],
                    "model": grp["model_short"],
                    "repo_id": grp["repo_id"],
                    "dp_size": dp_size,
                    "ulysses_degree": grp["ulysses_degree"],
                    "tensorcores_per_replica": grp["ulysses_degree"],
                    "chips_per_replica": round(grp["ulysses_degree"] / 2.0, 1),
                    "precision": grp["precision"],
                    "attention_backend": grp["attention_backend"],
                    "size": size,
                    "width": w,
                    "height": h,
                    "megapixels": round((w * h) / 1e6, 3),
                    "steps": steps,
                    "guidance_scale": gs,
                    "concurrency": conc,
                    "num_requests": n_req,
                    "ok_requests": len(oks),
                    "server_boot_s": boot_s,
                    "cold_compile_infer_s": round(cold_s, 3),
                    "warm_p50_s": p50_s,
                    "warm_mean_s": mean_s,
                    "warm_p90_s": p90_s,
                    "warm_min_s": min_s,
                    "server_inference_p50_s": srv_p50_s,
                    "ms_per_step": ms_per_step,
                    "wall_s": round(wall_s, 3),
                    "img_per_sec": img_per_s,
                    "ipm": ipm,
                    "hosts_for_60_ipm": math.ceil(60.0 / ipm) if ipm > 0 else None,
                    "hosts_for_300_ipm": math.ceil(300.0 / ipm) if ipm > 0 else None,
                    "peak_hbm_per_core_gib": measured_hbm if measured_hbm is not None else est_hbm,
                    "peak_hbm_source": "measured" if measured_hbm is not None else "analytic_estimate",
                    "measured_server_hbm_gib": srv_hbm,
                    "measured_libtpu_hbm_gib": libtpu_hbm,
                    "analytic_est_hbm_gib": est_hbm,
                    "pixel_mean": oks[0]["pixel_mean"],
                    "pixel_std": oks[0]["pixel_std"],
                    "timestamp": utc_now(),
                }
                append_record(rec)
                log(
                    f"  [OK] {pk} -> p50={p50_s:.3f}s ({ms_per_step:.1f} ms/step) | "
                    f"img/s={img_per_s:.3f} ({ipm:.1f} IPM) | cold={cold_s:.2f}s | "
                    f"HBM={rec['peak_hbm_per_core_gib']} GiB ({rec['peak_hbm_source']}) | "
                    f"hosts@60IPM={rec['hosts_for_60_ipm']} hosts@300IPM={rec['hosts_for_300_ipm']}"
                )

        except Exception as exc:
            log(f"Group {grp['group_id']} failed at server startup/execution: {exc}")
            if log_path.exists():
                gcs_upload_bytes(
                    args.gcs_bucket,
                    f"{args.gcs_prefix}/logs/{grp['group_id']}_startup_error.log",
                    log_path.read_bytes()[-30000:],
                    content_type="text/plain",
                )
            if grp.get("optional"):
                for pt in pending_pts:
                    pk = make_point_key(grp, pt)
                    if pk not in completed:
                        append_record(
                            {
                                "point_key": pk,
                                "status": "unsupported",
                                "engine": "sglang-torchtpu",
                                "group_id": grp["group_id"],
                                "model": grp["model_short"],
                                "dp_size": grp["dp_size"],
                                "ulysses_degree": grp["ulysses_degree"],
                                "precision": grp["precision"],
                                "attention_backend": grp["attention_backend"],
                                "size": pt["size"],
                                "steps": pt["steps"],
                                "concurrency": pt["c"],
                                "error": str(exc)[:600],
                                "timestamp": utc_now(),
                            }
                        )
        finally:
            hbm_mon.stop()
            stop_server_cleanly(proc)

    gcs_upload_bytes(args.gcs_bucket, f"{args.gcs_prefix}/PHASE1_SGLANG_COMPLETE.txt", f"{utc_now()}\n".encode("utf-8"), content_type="text/plain")
    log(f"ALL PHASE 1 SGLANG BENCHMARKS COMPLETE! Total points recorded: {len(completed)}")


if __name__ == "__main__":
    main()
