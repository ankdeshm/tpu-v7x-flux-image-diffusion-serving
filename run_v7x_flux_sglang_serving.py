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
     - `DP=4 x Ulysses=2` (2 TensorCores = 1 physical v7x chip x 4 isolated sub-host servers/host)
     - `DP=2 x Ulysses=4` (4 TensorCores = 2 physical v7x chips x 2 isolated sub-host servers/host, with lockstep fallback)
     - `DP=1 x Ulysses=8` (8 TensorCores = 4 physical v7x chips x 1 server/host)
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
import socket
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

METADATA_HOOK_C = r"""
#define _GNU_SOURCE
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <dlfcn.h>
#include <string.h>

static int (*real_connect)(int, const struct sockaddr*, socklen_t) = NULL;

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    if (!real_connect) {
        real_connect = dlsym(RTLD_NEXT, "connect");
    }
    if (addr && addr->sa_family == AF_INET) {
        const struct sockaddr_in *addr_in = (const struct sockaddr_in *)addr;
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(addr_in->sin_addr), ip, INET_ADDRSTRLEN);
        if (strcmp(ip, "169.254.169.254") == 0) {
            struct sockaddr_in redirected = *addr_in;
            redirected.sin_addr.s_addr = inet_addr("127.0.0.1");
            redirected.sin_port = htons(80);
            return real_connect(sockfd, (const struct sockaddr *)&redirected, sizeof(redirected));
        }
    }
    return real_connect(sockfd, addr, addrlen);
}
"""

MOCK_METADATA_PY = r"""#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import urllib.error
import urllib.parse
import urllib.request


class MockMetadataHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def forward_to_real_metadata(self):
        real_url = f"http://169.254.169.254{self.path}"
        req = urllib.request.Request(real_url)
        for key, val in self.headers.items():
            if key.lower() != "host":
                req.add_header(key, val)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                self.send_response(response.status)
                for key, val in response.headers.items():
                    self.send_header(key, val)
                self.end_headers()
                self.wfile.write(response.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, val in e.headers.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if self.headers.get("Metadata-Flavor") != "Google":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing Metadata-Flavor header")
            return

        response_body = None
        tpu_env_keys = {
            "TPU_WORKER_ID": "0",
            "ALT": "false",
            "CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "CONSUMER_PROJECT_ID": "tpu-prod-env-one-vm",
            "ENABLE_ICI_RESILIENCY": "false",
            "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
            "HOST_BOUNDS": "1,1,1",
            "INJECT_SLICE_BUILDER_FAULT": "",
            "NODE_ID": "0",
            "TOPOLOGY": "2x2x1",
            "TPU_ACCELERATOR_TYPE": "tpu7x-8",
            "WRAP": "false,false,false",
            "ZONE": "us-central1-c",
        }
        attr = path.rsplit("/attributes/", 1)[-1] if "/instance/attributes/" in path else None
        if attr in tpu_env_keys:
            response_body = tpu_env_keys[attr].encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/tpu-env"):
            response_body = (
                "TPU_WORKER_ID: '0'\n"
                "ALT: 'false'\n"
                "CHIPS_PER_HOST_BOUNDS: '2,2,1'\n"
                "CONSUMER_PROJECT_ID: 'tpu-prod-env-one-vm'\n"
                "ENABLE_ICI_RESILIENCY: 'false'\n"
                "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY: 'false'\n"
                "HOST_BOUNDS: '1,1,1'\n"
                "INJECT_SLICE_BUILDER_FAULT: ''\n"
                "NODE_ID: '0'\n"
                "TOPOLOGY: '2x2x1'\n"
                "TPU_ACCELERATOR_TYPE: 'tpu7x-8'\n"
                "WRAP: 'false,false,false'\n"
                "ZONE: 'us-central1-c'\n"
            ).encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/worker-network-endpoints") or path.endswith(
            "/computeMetadata/v1/instance/attributes/instance-id"
        ):
            req = urllib.request.Request(
                "http://169.254.169.254/computeMetadata/v1/instance/hostname",
                headers={"Metadata-Flavor": "Google"},
            )
            hn = urllib.request.urlopen(req, timeout=2).read().decode().strip().split(".")[0]
            response_body = hn.encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/accelerator-type"):
            response_body = b"tpu7x-8"
        elif path.endswith("/computeMetadata/v1/instance/attributes/agent-worker-number"):
            response_body = b"0"
        elif path.endswith("/computeMetadata/v1/project/project-id"):
            response_body = b"tpu-prod-env-one-vm"
        elif path.endswith("/computeMetadata/v1/project/numeric-project-id"):
            response_body = b"1234567890"
        elif path.endswith("/computeMetadata/v1/universe/universe-domain") or path.endswith("/universe-domain"):
            response_body = b"googleapis.com"

        if response_body is not None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.forward_to_real_metadata()


if __name__ == "__main__":
    httpd = ThreadingHTTPServer(("127.0.0.1", 80), MockMetadataHandler)
    print("[mock_metadata] Serving on 127.0.0.1:80", file=sys.stderr, flush=True)
    httpd.serve_forever()
"""

_MOCK_META_PROC: subprocess.Popen[Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{utc_now()}][sglang-flux-v7x] {msg}", flush=True)


def ensure_subhost_isolation_ready() -> None:
    """Compile metadata_hook.so, launch mock_metadata.py on 127.0.0.1:80, and patch sglang worker LOCAL_RANK pinning."""
    global _MOCK_META_PROC
    hook_c = Path("/tmp/metadata_hook.c")
    hook_so = Path("/tmp/metadata_hook.so")
    if not hook_so.exists():
        hook_c.write_text(METADATA_HOOK_C)
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", str(hook_c), "-o", str(hook_so), "-ldl"], check=True)
        log("Compiled /tmp/metadata_hook.so for fast GCE metadata interception")

    # Start mock_metadata.py if 127.0.0.1:80 is not yet listening
    port80_up = False
    try:
        with socket.create_connection(("127.0.0.1", 80), timeout=0.5):
            port80_up = True
    except Exception:
        port80_up = False

    if not port80_up:
        mock_py = Path("/tmp/mock_metadata.py")
        mock_py.write_text(MOCK_METADATA_PY)
        _MOCK_META_PROC = subprocess.Popen(
            ["python3", "-u", str(mock_py)],
            stdout=open("/tmp/mock_metadata.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(25):
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:80/computeMetadata/v1/instance/attributes/tpu-env",
                    headers={"Metadata-Flavor": "Google"},
                )
                with urllib.request.urlopen(req, timeout=1.0) as r:
                    if r.status == 200:
                        log("Started mock_metadata.py proxy on 127.0.0.1:80")
                        break
            except Exception:
                time.sleep(0.2)

    # Idempotently patch sglang worker_bootstrap.py & platforms/tpu.py for SGLANG_TPU_LOCAL_RANKS
    wb_path = Path("/usr/local/lib/python3.12/site-packages/sglang/multimodal_gen/runtime/managers/worker_bootstrap.py")
    if wb_path.exists():
        wb_txt = wb_path.read_text()
        if "SGLANG_TPU_LOCAL_RANKS" not in wb_txt:
            anchor = "def bootstrap_scheduler_process(spec: SchedulerProcessSpec) -> None:\n"
            inject = (
                anchor
                + "    import os as _os\n"
                + "    _os.environ['RANK'] = str(spec.rank)\n"
                + "    if _os.environ.get('SGLANG_TPU_LOCAL_RANKS'):\n"
                + "        _lrs = [x.strip() for x in _os.environ['SGLANG_TPU_LOCAL_RANKS'].split(',') if x.strip()]\n"
                + "        _os.environ['LOCAL_RANK'] = _lrs[spec.rank]\n"
            )
            if anchor in wb_txt:
                wb_path.write_text(wb_txt.replace(anchor, inject, 1))
                log("Patched worker_bootstrap.py for SGLANG_TPU_LOCAL_RANKS")

    tpu_path = Path("/usr/local/lib/python3.12/site-packages/sglang/multimodal_gen/runtime/platforms/tpu.py")
    if tpu_path.exists():
        tpu_txt = tpu_path.read_text()
        if "SGLANG_TPU_LOCAL_RANKS" not in tpu_txt:
            anchor = "_prepare_tpu_env(world_size=_world_size(), resolve_single_host_topology=True)"
            inject = (
                "if os.environ.get('SGLANG_TPU_LOCAL_RANKS'):\n"
                "            _lrs = [x.strip() for x in os.environ['SGLANG_TPU_LOCAL_RANKS'].split(',') if x.strip()]\n"
                "            os.environ['LOCAL_RANK'] = _lrs[int(os.environ.get('RANK', '0'))]\n"
                "            for _k in ('TPU_CHIPS_PER_HOST_BOUNDS', 'TPU_HOST_BOUNDS', 'TPU_WORKER_ID', 'TPU_WORKER_HOSTNAMES', 'CLOUD_TPU_TASK_ID'):\n"
                "                os.environ.pop(_k, None)\n"
                "        " + anchor
            )
            if anchor in tpu_txt:
                tpu_path.write_text(tpu_txt.replace(anchor, inject, 1))
                log("Patched platforms/tpu.py for SGLANG_TPU_LOCAL_RANKS sub-host VFIO pinning")


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
    """Polls libtpu runtime metrics endpoints (`localhost:8431..8438/metrics`) to capture Peak HBM per TensorCore."""

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
                                if val > 1024 * 1024:
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


def parse_hbm_from_logs(log_paths: list[Path]) -> float | None:
    vals_gib: list[float] = []
    for lp in log_paths:
        if not lp.exists():
            continue
        try:
            txt = lp.read_text(errors="ignore")
        except Exception:
            continue
        for m in re.finditer(r"(?:memory|hbm|allocated|used)[^\n]{0,40}?([0-9]+\.[0-9]+)\s*(GiB|GB|MiB|MB)", txt, re.IGNORECASE):
            val = float(m.group(1))
            unit = m.group(2).upper()
            if unit in ("MIB", "MB"):
                val /= 1024.0
            if 1.0 <= val <= 96.0:
                vals_gib.append(val)
    return round(max(vals_gib), 2) if vals_gib else None


def find_vfio_holder_pids() -> list[int]:
    my_pid = os.getpid()
    mock_pid = _MOCK_META_PROC.pid if _MOCK_META_PROC is not None else -1
    holders: set[int] = set()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid in (my_pid, mock_pid, 1):
            continue
        fd_dir = p / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                    if "/dev/vfio/" in target:
                        holders.add(pid)
                        break
                except Exception:
                    pass
        except Exception:
            pass
    return sorted(holders)


def stop_cluster_cleanly(procs: list[subprocess.Popen[Any]] | None = None) -> None:
    if procs:
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
        t_wait = time.time()
        while any(p is not None and p.poll() is None for p in procs) and time.time() - t_wait < 6.0:
            time.sleep(0.3)
        for proc in procs:
            if proc is not None and proc.poll() is None:
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
            "pkill -15 -f '[s]glang serve' 2>/dev/null || true; "
            "pkill -15 -f '[s]glang.multimodal_gen' 2>/dev/null || true; "
            "sleep 1.0; "
            "pkill -9 -f '[s]glang serve' 2>/dev/null || true; "
            "pkill -9 -f '[s]glang.multimodal_gen' 2>/dev/null || true; "
            "pkill -9 -f '[m]ultiprocessing.spawn' 2>/dev/null || true; "
            "pkill -9 -f '[m]ultiprocessing.resource_tracker' 2>/dev/null || true; "
            "rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*/*.lock 2>/dev/null || true",
        ],
        check=False,
    )

    # Guarantee 0 processes hold /dev/vfio/* before returning
    t_vfio = time.time()
    while time.time() - t_vfio < 20.0:
        holders = find_vfio_holder_pids()
        if not holders:
            break
        log(f"Killing lingering VFIO holder PIDs: {holders}")
        for pid in holders:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(1.0)
    time.sleep(1.5)


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
        valid = w == exp_w and h == exp_h and finite and p_std > 4.0
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


class SGLangCluster:
    """Manages 1 or more isolated SGLang server replicas across the 8 v7x TensorCores with least-inflight load balancing."""

    def __init__(
        self,
        procs: list[subprocess.Popen[Any]],
        endpoints: list[str],
        log_paths: list[Path],
        boot_s: float,
        mode: str,
        lockstep_dp: int = 1,
    ) -> None:
        self.procs = procs
        self.endpoints = endpoints
        self.log_paths = log_paths
        self.boot_s = boot_s
        self.mode = mode
        self.lockstep_dp = lockstep_dp
        self._lock = threading.Lock()
        self._inflight = [0] * len(endpoints)
        self._rr = 0

    def _acquire_endpoint(self) -> tuple[int, str]:
        with self._lock:
            n = len(self.endpoints)
            best_idx = min(range(n), key=lambda i: (self._inflight[i], (i - self._rr) % n))
            self._inflight[best_idx] += 1
            self._rr = (best_idx + 1) % n
            return best_idx, self.endpoints[best_idx]

    def _release_endpoint(self, idx: int) -> None:
        with self._lock:
            self._inflight[idx] = max(0, self._inflight[idx] - 1)

    def generate_single(
        self,
        prompt: str,
        size: str,
        steps: int,
        gs: float,
        seed: int,
        endpoint_override: str | None = None,
        timeout_s: float = 900.0,
    ) -> dict[str, Any]:
        if self.lockstep_dp > 1 and endpoint_override is None:
            # Lockstep fallback (only used if 8-core single slice has dp_size > 1 and ulysses > 1):
            # fire 1 target + (lockstep_dp - 1) shadow companion requests concurrently so all DP groups enter AllToAll together.
            ep = self.endpoints[0]
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.lockstep_dp) as pool:
                f_main = pool.submit(generate_one_image, ep, prompt, size, steps, gs, seed, timeout_s)
                f_helpers = [
                    pool.submit(generate_one_image, ep, PROMPTS[(seed + j) % len(PROMPTS)], size, steps, gs, seed + 10000 + j, timeout_s)
                    for j in range(1, self.lockstep_dp)
                ]
                res = f_main.result()
                for fh in f_helpers:
                    try:
                        fh.result()
                    except Exception:
                        pass
                return res

        if endpoint_override is not None:
            return generate_one_image(endpoint_override, prompt, size, steps, gs, seed, timeout_s)

        idx, ep = self._acquire_endpoint()
        try:
            return generate_one_image(ep, prompt, size, steps, gs, seed, timeout_s)
        finally:
            self._release_endpoint(idx)

    def run_wave(self, conc: int, n_req: int, size: str, steps: int, gs: float, seed_base: int) -> tuple[list[dict[str, Any]], float]:
        if self.lockstep_dp > 1 and conc > 1:
            # Ensure n_req is a multiple of lockstep_dp so no tail request is stranded without a peer
            rem = n_req % self.lockstep_dp
            eff_n = n_req if rem == 0 else (n_req + self.lockstep_dp - rem)
            eff_conc = max(self.lockstep_dp, (conc // self.lockstep_dp) * self.lockstep_dp)
            ep = self.endpoints[0]
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=eff_conc) as pool:
                futs = [
                    pool.submit(generate_one_image, ep, PROMPTS[i % len(PROMPTS)], size, steps, gs, seed_base + i)
                    for i in range(eff_n)
                ]
                outs = [f.result() for f in futs]
            return outs, time.perf_counter() - t0

        t0 = time.perf_counter()
        if conc == 1:
            outs = [
                self.generate_single(PROMPTS[i % len(PROMPTS)], size, steps, gs, seed_base + i)
                for i in range(n_req)
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                futs = [
                    pool.submit(self.generate_single, PROMPTS[i % len(PROMPTS)], size, steps, gs, seed_base + i)
                    for i in range(n_req)
                ]
                outs = [f.result() for f in futs]
        return outs, time.perf_counter() - t0

    def stop(self) -> None:
        stop_cluster_cleanly(self.procs)


def _launch_one_sglang_instance(
    model_path: str,
    port: int,
    scheduler_port: int,
    master_port: int,
    num_gpus: int,
    dp_size: int,
    ulysses_degree: int,
    attention_backend: str,
    quantization: str | None,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
    boot_timeout_s: float = 900.0,
) -> tuple[subprocess.Popen[Any], float]:
    out_dir = Path(f"/dev/shm/sglang_outputs_{port}")
    out_dir.mkdir(parents=True, exist_ok=True)
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
    if Path("/tmp/metadata_hook.so").exists():
        env["LD_PRELOAD"] = "/tmp/metadata_hook.so"
    if extra_env:
        env.update(extra_env)

    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--scheduler-port",
        str(scheduler_port),
        "--master-port",
        str(master_port),
        "--num-gpus",
        str(num_gpus),
        "--dp-size",
        str(dp_size),
        "--ulysses-degree",
        str(ulysses_degree),
        "--attention-backend",
        attention_backend,
        "--warmup-mode",
        "off",
        "--output-path",
        str(out_dir),
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])

    lrs_tag = env.get("SGLANG_TPU_LOCAL_RANKS", "all-8")
    topo_tag = env.get("TORCH_TPU_TOPOLOGY", "auto")
    log(f"Launching SGLang instance :{port} (local_ranks={lrs_tag}, topo={topo_tag}): {' '.join(cmd)}")
    lf = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

    while time.time() - t0 < boot_timeout_s:
        if proc.poll() is not None:
            lf.flush()
            tail = extract_clean_error_tail(log_path)
            raise RuntimeError(f"sglang serve :{port} exited early (rc={proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3.0) as r:
                if r.status == 200:
                    boot_s = round(time.time() - t0, 2)
                    log(f"SGLang ready on :{port} in {boot_s}s (local_ranks={lrs_tag}, dp={dp_size}, ulysses={ulysses_degree})")
                    return proc, boot_s
        except Exception:
            pass
        time.sleep(2.0)

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    tail = extract_clean_error_tail(log_path)
    raise TimeoutError(f"Timed out waiting for SGLang server on :{port}:\n{tail}")


def extract_clean_error_tail(log_path: Path, max_chars: int = 3500) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="ignore").splitlines()
    filtered = [
        ln
        for ln in lines
        if "coredump_hook.cc" not in ln
        and "Coroner client" not in ln
        and "remote_coredump.socket" not in ln
        and "process_state.cc:794" not in ln
    ]
    return "\n".join(filtered)[-max_chars:]


def start_sglang_cluster(
    model_path: str,
    base_port: int,
    dp_size: int,
    ulysses_degree: int,
    attention_backend: str,
    quantization: str | None,
    group_id: str,
) -> SGLangCluster:
    stop_cluster_cleanly(None)
    Path("/dev/shm/xla_cache").mkdir(parents=True, exist_ok=True)
    t_cluster0 = time.time()

    # Case 1: DP=4 x Ulysses=2 -> 4 independent 1-chip (2-TensorCore) servers on [0,1], [2,3], [4,5], [6,7]
    if dp_size == 4 and ulysses_degree == 2:
        procs: list[subprocess.Popen[Any]] = []
        endpoints: list[str] = []
        log_paths: list[Path] = []
        try:
            for r in range(4):
                p_http = base_port + r * 20
                p_sched = 6100 + r * 30
                p_master = 31100 + r * 30
                sb0 = 28100 + r * 10
                tmp_r = Path(f"/tmp/sgl_u2_rep{r}")
                tmp_r.mkdir(parents=True, exist_ok=True)
                lp = Path(f"/tmp/{group_id}_rep{r}.log")
                extra_env = {
                    "SGLANG_TPU_LOCAL_RANKS": f"{2 * r},{2 * r + 1}",
                    "TORCH_TPU_TOPOLOGY": "1,1,1,2",
                    "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"localhost:{sb0},localhost:{sb0 + 1}",
                    "TPU_RUNTIME_METRICS_PORTS": f"{8431 + 2 * r},{8432 + 2 * r}",
                    "TMPDIR": str(tmp_r),
                }
                proc, _ = _launch_one_sglang_instance(
                    model_path=model_path,
                    port=p_http,
                    scheduler_port=p_sched,
                    master_port=p_master,
                    num_gpus=2,
                    dp_size=1,
                    ulysses_degree=2,
                    attention_backend=attention_backend,
                    quantization=quantization,
                    log_path=lp,
                    extra_env=extra_env,
                    boot_timeout_s=420.0,
                )
                procs.append(proc)
                endpoints.append(f"http://127.0.0.1:{p_http}")
                log_paths.append(lp)
            total_boot = round(time.time() - t_cluster0, 2)
            log(f"DP=4 x Ulysses=2 isolated 4-server cluster ready in {total_boot}s: {endpoints}")
            return SGLangCluster(procs, endpoints, log_paths, total_boot, mode="subhost_4x_u2", lockstep_dp=1)
        except Exception as exc:
            stop_cluster_cleanly(procs)
            raise exc

    # Case 2: DP=2 x Ulysses=4 -> Try 2 independent 2-chip (4-TensorCore) servers first; fallback cleanly to 8-core lockstep if needed
    if dp_size == 2 and ulysses_degree == 4:
        for topo_try in ("1,2,1,2", "2,1,1,2"):
            procs = []
            endpoints = []
            log_paths = []
            try:
                for r, lrs in enumerate(("0,1,2,3", "4,5,6,7")):
                    p_http = base_port + r * 20
                    p_sched = 6200 + r * 30
                    p_master = 31200 + r * 30
                    sb0 = 28200 + r * 10
                    tmp_r = Path(f"/tmp/sgl_u4_rep{r}")
                    tmp_r.mkdir(parents=True, exist_ok=True)
                    lp = Path(f"/tmp/{group_id}_rep{r}.log")
                    extra_env = {
                        "SGLANG_TPU_LOCAL_RANKS": lrs,
                        "TORCH_TPU_TOPOLOGY": topo_try,
                        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{sb0 + k}" for k in range(4)),
                        "TPU_RUNTIME_METRICS_PORTS": ",".join(str(8431 + 4 * r + k) for k in range(4)),
                        "TMPDIR": str(tmp_r),
                    }
                    proc, _ = _launch_one_sglang_instance(
                        model_path=model_path,
                        port=p_http,
                        scheduler_port=p_sched,
                        master_port=p_master,
                        num_gpus=4,
                        dp_size=1,
                        ulysses_degree=4,
                        attention_backend=attention_backend,
                        quantization=quantization,
                        log_path=lp,
                        extra_env=extra_env,
                        boot_timeout_s=150.0,
                    )
                    procs.append(proc)
                    endpoints.append(f"http://127.0.0.1:{p_http}")
                    log_paths.append(lp)
                total_boot = round(time.time() - t_cluster0, 2)
                log(f"DP=2 x Ulysses=4 isolated 2-server cluster ({topo_try}) ready in {total_boot}s: {endpoints}")
                return SGLangCluster(procs, endpoints, log_paths, total_boot, mode=f"subhost_2x_u4_{topo_try}", lockstep_dp=1)
            except Exception as sub_exc:
                log(f"Sub-host 2x U=4 ({topo_try}) did not bind ({str(sub_exc)[:240]}); cleaning up before fallback...")
                stop_cluster_cleanly(procs)

        # Fallback: single 8-TensorCore server with lockstep_dp=2 wave pairing (0% hang risk!)
        lp = Path(f"/tmp/{group_id}.log")
        proc, boot_s = _launch_one_sglang_instance(
            model_path=model_path,
            port=base_port,
            scheduler_port=6300,
            master_port=31300,
            num_gpus=8,
            dp_size=2,
            ulysses_degree=4,
            attention_backend=attention_backend,
            quantization=quantization,
            log_path=lp,
            extra_env={"TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438"},
            boot_timeout_s=600.0,
        )
        return SGLangCluster([proc], [f"http://127.0.0.1:{base_port}"], [lp], boot_s, mode="single_8core_lockstep_dp2_u4", lockstep_dp=2)

    # Case 3: DP=8 x Ulysses=1 or DP=1 x Ulysses=8 -> Single 8-TensorCore server
    lp = Path(f"/tmp/{group_id}.log")
    proc, boot_s = _launch_one_sglang_instance(
        model_path=model_path,
        port=base_port,
        scheduler_port=6000,
        master_port=31000,
        num_gpus=8,
        dp_size=dp_size,
        ulysses_degree=ulysses_degree,
        attention_backend=attention_backend,
        quantization=quantization,
        log_path=lp,
        extra_env={"TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438"},
        boot_timeout_s=900.0,
    )
    return SGLangCluster([proc], [f"http://127.0.0.1:{base_port}"], [lp], boot_s, mode=f"single_8core_dp{dp_size}_u{ulysses_degree}", lockstep_dp=1)


def estimate_theoretic_hbm_gib(model_short: str, ulysses_degree: int, precision: str, width: int, height: int) -> float:
    if "Z-Image" in model_short:
        dit_gb, enc_gb, vae_gb = (6.1 if precision == "FP8" else 12.3), 8.2, 0.35
    else:
        dit_gb, enc_gb, vae_gb = (11.9 if precision == "FP8" else 23.8), 9.8, 0.35
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

    # 2. FLUX.1-schnell | DP=4 x Ulysses=2 (1 physical v7x chip = 2 TensorCores/replica x 4 isolated servers) | BF16 | tpu_splash_attn
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
            "group_id": "sglang_zimage_turbo_dp4_u2_bf16_splash",
            "repo_id": "Tongyi-MAI/Z-Image-Turbo",
            "model_short": "Z-Image-Turbo",
            "dp_size": 4,
            "ulysses_degree": 2,
            "precision": "BF16",
            "quantization": None,
            "attention_backend": "tpu_splash_attn",
            "guidance_scale": 0.0,
            "points": [
                {"size": "1024x1024", "steps": 8, "c": 1, "n": 4},
                {"size": "1024x1024", "steps": 8, "c": 4, "n": 12},
                {"size": "1024x1024", "steps": 8, "c": 8, "n": 16},
                {"size": "2048x2048", "steps": 8, "c": 1, "n": 3},
                {"size": "2048x2048", "steps": 8, "c": 4, "n": 8},
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

    ensure_subhost_isolation_ready()

    cache_root = Path("/dev/shm/hf_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    local_jsonl = Path("/dev/shm/phase1_sglang_results.jsonl")
    gcs_jsonl_obj = f"{args.gcs_prefix}/phase1_sglang_results.jsonl"
    gcs_summary_obj = f"{args.gcs_prefix}/phase1_sglang_summary.json"

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
        hbm_mon = LibtpuHbmMonitor()
        hbm_mon.start()
        cluster: SGLangCluster | None = None

        try:
            cluster = start_sglang_cluster(
                model_path=model_dir,
                base_port=args.port,
                dp_size=grp["dp_size"],
                ulysses_degree=grp["ulysses_degree"],
                attention_backend=grp["attention_backend"],
                quantization=grp["quantization"],
                group_id=grp["group_id"],
            )

            cold_by_size: dict[str, float] = {}
            warmed_shape_steps: set[tuple[str, int]] = set()

            for pt in pending_pts:
                pk = make_point_key(grp, pt)
                size = pt["size"]
                w, h = (int(x) for x in size.split("x"))
                steps = pt["steps"]
                conc = pt["c"]
                n_req = pt["n"]
                dp_size = grp["dp_size"]
                gs = grp["guidance_scale"]

                # 1) Cold compile + warmup across all DP replicas on (size, steps)
                if size not in cold_by_size:
                    log(f"  [COLD+WARMUP] {pk}: running 1st cold request ({size}, steps={steps}, cluster={cluster.mode})...")
                    cold_res = cluster.generate_single(PROMPTS[0], size, steps, gs, seed=1001)
                    if not cold_res["ok"]:
                        log(f"  [ERROR] Cold request failed for {pk}: {cold_res.get('error')}")
                        err_blobs = b"\n\n".join(
                            lp.read_bytes()[-20000:] for lp in cluster.log_paths if lp.exists()
                        )
                        gcs_upload_bytes(
                            args.gcs_bucket,
                            f"{args.gcs_prefix}/logs/{grp['group_id']}_error.log",
                            err_blobs,
                            content_type="text/plain",
                        )
                        status_tag = "unsupported" if grp.get("optional") else "error"
                        append_record(
                            {
                                "point_key": pk,
                                "status": status_tag,
                                "engine": "sglang-torchtpu",
                                "group_id": grp["group_id"],
                                "cluster_mode": cluster.mode,
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
                    cold_by_size[size] = cold_s
                    log(f"  [COLD OK] {size} steps={steps}: cold_compile_infer_s={cold_s:.2f}s (pixel_std={cold_res['pixel_std']})")
                else:
                    cold_s = cold_by_size[size]

                if (size, steps) not in warmed_shape_steps:
                    if len(cluster.endpoints) > 1:
                        log(f"  [MULTI-SERVER WARMUP] Warming all {len(cluster.endpoints)} isolated SGLang servers on ({size}, steps={steps})...")
                        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cluster.endpoints)) as wp:
                            futs = [
                                wp.submit(
                                    cluster.generate_single,
                                    PROMPTS[i % len(PROMPTS)],
                                    size,
                                    steps,
                                    gs,
                                    1100 + i,
                                    ep,
                                )
                                for i, ep in enumerate(cluster.endpoints)
                            ]
                            for f in futs:
                                f.result()
                    elif dp_size > 1:
                        log(f"  [DP WARMUP] Warming all {dp_size} DP replicas on ({size}, steps={steps})...")
                        _, _ = cluster.run_wave(dp_size, dp_size * 2, size, steps, gs, seed_base=1100)
                    else:
                        _ = cluster.generate_single(PROMPTS[1], size, steps, gs, seed=1199)
                    warmed_shape_steps.add((size, steps))

                # 2) Timed Steady-State Warm Benchmark Wave
                log(f"  [BENCH] {pk}: running n={n_req} warm requests at concurrency c={conc} (mode={cluster.mode})...")
                outs, wall_s = cluster.run_wave(conc, n_req, size, steps, gs, seed_base=4000 + conc * 100)

                oks = [o for o in outs if o.get("ok")]
                if not oks:
                    err_msg = outs[0].get("error") if outs else "no outputs"
                    log(f"  [ERROR] {pk} 0/{len(outs)} succeeded: {err_msg}")
                    append_record(
                        {
                            "point_key": pk,
                            "status": "error",
                            "engine": "sglang-torchtpu",
                            "group_id": grp["group_id"],
                            "cluster_mode": cluster.mode,
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
                log_hbm = parse_hbm_from_logs(cluster.log_paths)
                srv_hbm = round(max(srv_mems), 2) if srv_mems else None
                measured_hbm = srv_hbm or libtpu_hbm or log_hbm
                est_hbm = estimate_theoretic_hbm_gib(grp["model_short"], grp["ulysses_degree"], grp["precision"], w, h)

                rec = {
                    "point_key": pk,
                    "status": "ok",
                    "engine": "sglang-torchtpu",
                    "group_id": grp["group_id"],
                    "cluster_mode": cluster.mode,
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
                    "num_requests": len(outs),
                    "ok_requests": len(oks),
                    "server_boot_s": cluster.boot_s,
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
            for lp in Path("/tmp").glob(f"{grp['group_id']}*.log"):
                gcs_upload_bytes(
                    args.gcs_bucket,
                    f"{args.gcs_prefix}/logs/{lp.name}",
                    lp.read_bytes()[-30000:],
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
                                "error": str(exc)[:800],
                                "timestamp": utc_now(),
                            }
                        )
        finally:
            hbm_mon.stop()
            if cluster is not None:
                cluster.stop()
            else:
                stop_cluster_cleanly(None)

    gcs_upload_bytes(
        args.gcs_bucket,
        f"{args.gcs_prefix}/PHASE1_SGLANG_COMPLETE.txt",
        f"{utc_now()}\n".encode("utf-8"),
        content_type="text/plain",
    )
    log(f"ALL PHASE 1 SGLANG BENCHMARKS COMPLETE! Total points recorded: {len(completed)}")


if __name__ == "__main__":
    main()
