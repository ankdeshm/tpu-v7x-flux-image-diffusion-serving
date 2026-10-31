#!/usr/bin/env python3
"""Stage FLUX & Diffusion models into /dev/shm/hf_cache (1,512 GiB host RAM) + GCS mirror,
and probe TorchTPU serving stack (sglang-torchtpu + vllm-omni) & multi-replica DP x Ulysses topologies on TPU v7x-8.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata as md
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{utc_now()}][flux-stage-probe] {msg}", flush=True)


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def gcs_upload_bytes(bucket: str, obj: str, data: bytes, content_type: str = "application/json") -> None:
    for attempt in range(4):
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
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


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
        if not it["name"].endswith("/") and "/.cache/" not in it["name"]
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
    log(f"Synced {total_b / 1e9:.2f} GB from gs://{bucket}/{gcs_prefix} in {dt:.1f}s ({(total_b / 1e9) / dt:.2f} GB/s)")
    return len(blobs)


def mirror_dir_to_gcs(bucket: str, local_dir: Path, gcs_prefix: str) -> None:
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"
    existing = {
        it["name"][len(gcs_prefix) :]: int(it["size"])
        for it in gcs_list_prefix(bucket, gcs_prefix)
        if not it["name"].endswith("/")
    }
    to_upload: list[tuple[Path, str, int]] = []
    for p in sorted(local_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        if rel.startswith(".cache/") or "/.cache/" in rel:
            continue
        sz = p.stat().st_size
        if existing.get(rel) == sz:
            continue
        to_upload.append((p, gcs_prefix + rel, sz))

    if not to_upload:
        log(f"GCS mirror gs://{bucket}/{gcs_prefix} already up-to-date ({len(existing)} files).")
        return

    total_bytes = sum(x[2] for x in to_upload)
    log(f"Mirroring {len(to_upload)} files ({total_bytes / 1e9:.2f} GB) to gs://{bucket}/{gcs_prefix}...")

    def upload_file(item: tuple[Path, str, int]) -> int:
        p, obj, sz = item
        for attempt in range(4):
            try:
                tok = vm_access_token()
                url = (
                    f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
                    f"?uploadType=media&name={urllib.parse.quote(obj, safe='')}"
                )
                with open(p, "rb") as f:
                    req = urllib.request.Request(
                        url,
                        data=f,
                        method="POST",
                        headers={
                            "Authorization": f"Bearer {tok}",
                            "Content-Type": "application/octet-stream",
                            "Content-Length": str(sz),
                        },
                    )
                    with urllib.request.urlopen(req, timeout=600):
                        return sz
            except Exception as exc:
                if attempt == 3:
                    log(f"WARNING: failed uploading {obj}: {exc}")
                    return 0
                time.sleep(2 * (attempt + 1))
        return 0

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        up_b = sum(pool.map(upload_file, to_upload))
    dt = max(time.time() - t0, 1e-3)
    log(f"Mirrored {up_b / 1e9:.2f} GB to gs://{bucket}/{gcs_prefix} in {dt:.1f}s ({(up_b / 1e9) / dt:.2f} GB/s)")


def collect_pkg_versions() -> dict[str, str]:
    pkgs = [
        "torch",
        "torch-tpu",
        "libtpu",
        "jax",
        "jaxlib",
        "diffusers",
        "transformers",
        "accelerate",
        "sglang",
        "sglang-tpu",
        "vllm",
        "vllm-omni",
        "vllm-torchtpu",
    ]
    out: dict[str, str] = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except Exception:
            pass
    return out


def probe_sglang_internals() -> dict[str, Any]:
    info: dict[str, Any] = {"packages": collect_pkg_versions()}
    try:
        import sglang
        import sglang_tpu

        info["sglang_file"] = sglang.__file__
        info["sglang_tpu_file"] = sglang_tpu.__file__
        sg_root = Path(sglang.__file__).parent
        sgt_root = Path(sglang_tpu.__file__).parent

        # Find diffusion / multimodal_gen files, server args, image endpoints, and flux support
        mm_dir = sg_root / "multimodal_gen"
        if mm_dir.exists():
            info["multimodal_gen_files"] = [
                str(p.relative_to(mm_dir))
                for p in sorted(mm_dir.rglob("*.py"))
                if any(k in p.name.lower() for k in ("flux", "zimage", "z_image", "image", "server_args", "openai", "http", "parallel", "quant", "sd3", "attn"))
            ][:80]
            # Inspect ServerArgs fields
            sa_files = list(mm_dir.rglob("*server_args*.py"))
            if sa_files:
                info["server_args_src"] = sa_files[0].read_text()[:12000]
            # Inspect image API routes
            ep_files = [p for p in mm_dir.rglob("*.py") if "image" in p.name.lower() or "http_server" in p.name.lower() or "openai" in p.name.lower()]
            info["endpoint_snippets"] = {str(p.relative_to(mm_dir)): p.read_text()[:6000] for p in ep_files[:6]}
            # Inspect model registry
            reg_files = [p for p in mm_dir.rglob("*.py") if "registry" in p.name.lower() or "flux" in p.name.lower()]
            info["registry_snippets"] = {str(p.relative_to(mm_dir)): p.read_text()[:6000] for p in reg_files[:6]}

        info["sglang_tpu_files"] = [str(p.relative_to(sgt_root)) for p in sorted(sgt_root.rglob("*.py"))][:80]
        for p in sorted(sgt_root.rglob("*.py")):
            if any(k in p.name.lower() for k in ("splash", "flux", "platform", "parallel", "quant", "diffusion")):
                info.setdefault("sglang_tpu_snippets", {})[str(p.relative_to(sgt_root))] = p.read_text()[:5000]
    except Exception as exc:
        info["error"] = repr(exc)
    return info


def probe_tpu_hardware_and_partitioning() -> dict[str, Any]:
    hw: dict[str, Any] = {}
    hw["vfio_devices"] = sorted(str(p) for p in Path("/dev/vfio").glob("*")) if Path("/dev/vfio").exists() else []
    hw["vbar_sock_exists"] = Path("/var/run/tpu-plugin/vbar.sock").is_socket()
    hw["env_tpu_vars"] = {k: v for k, v in os.environ.items() if "TPU" in k or "PJRT" in k or "VBAR" in k}

    # Test 1: Default 8-TensorCore single-host visibility
    r_full = subprocess.run(
        [
            "python3",
            "-c",
            (
                "import os, torch, torch_tpu; "
                "os.environ['TPU_PREMAPPED_BUFFER_SIZE']=str(512*1024**2); "
                "n = torch.tpu.device_count(); "
                "x = torch.ones((1024,1024), device='tpu:0', dtype=torch.bfloat16); "
                "mem = getattr(torch.tpu, 'max_memory_allocated', lambda d=0: -1)(0); "
                "mem_info = getattr(torch.tpu, 'mem_get_info', lambda d=0: (-1,-1))(0); "
                "print('FULL_OK', n, float(x.sum().cpu()), mem, mem_info)"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    hw["test_full_8core"] = {"rc": r_full.returncode, "stdout": r_full.stdout[-1500:], "stderr": r_full.stderr[-1500:]}

    # Test 2: Can we run 4 simultaneous 1-chip (2-TensorCore) processes via TPU_VISIBLE_CHIPS=0..3?
    sub_script = (
        "import os, sys, time, torch, torch_tpu; "
        "c = int(sys.argv[1]); "
        "os.environ['TPU_VISIBLE_CHIPS'] = str(c); "
        "os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,1,1'; "
        "os.environ['TPU_HOST_BOUNDS'] = '1,1,1'; "
        "os.environ['TPU_SKIP_MDS_QUERY'] = '1'; "
        "os.environ['TPU_RUNTIME_METRICS_PORTS'] = f'{8441+2*c},{8442+2*c}'; "
        "os.environ['TPU_PREMAPPED_BUFFER_SIZE'] = str(512*1024**2); "
        "n = torch.tpu.device_count(); "
        "t0 = torch.ones((512,512), device='tpu:0', dtype=torch.bfloat16); "
        "t1 = torch.ones((512,512), device='tpu:1', dtype=torch.bfloat16) if n > 1 else t0; "
        "time.sleep(2.0); "
        "print(f'CHIP_{c}_OK count={n} sum0={float(t0.sum().cpu())} sum1={float(t1.sum().cpu())}')"
    )
    procs = [subprocess.Popen(["python3", "-c", sub_script, str(c)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for c in range(4)]
    outs = []
    for c, p in enumerate(procs):
        so, se = p.communicate(timeout=90)
        outs.append({"chip": c, "rc": p.returncode, "stdout": so[-600:], "stderr": se[-600:]})
    hw["test_4x_1chip_concurrent"] = outs

    # Test 3: Can we run 2 simultaneous 2-chip (4-TensorCore) processes via TPU_VISIBLE_CHIPS=0,1 and 2,3?
    sub2_script = (
        "import os, sys, time, torch, torch_tpu; "
        "idx = int(sys.argv[1]); chips = '0,1' if idx == 0 else '2,3'; "
        "os.environ['TPU_VISIBLE_CHIPS'] = chips; "
        "os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,2,1'; "
        "os.environ['TPU_HOST_BOUNDS'] = '1,1,1'; "
        "os.environ['TPU_SKIP_MDS_QUERY'] = '1'; "
        "os.environ['TPU_RUNTIME_METRICS_PORTS'] = ','.join(str(8451 + 4*idx + k) for k in range(4)); "
        "os.environ['TPU_PREMAPPED_BUFFER_SIZE'] = str(512*1024**2); "
        "n = torch.tpu.device_count(); "
        "ts = [torch.ones((512,512), device=f'tpu:{k}', dtype=torch.bfloat16) for k in range(n)]; "
        "time.sleep(2.0); "
        "print(f'PAIR_{idx}_OK chips={chips} count={n} sums={[float(t.sum().cpu()) for t in ts]}')"
    )
    procs2 = [subprocess.Popen(["python3", "-c", sub2_script, str(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(2)]
    outs2 = []
    for i, p in enumerate(procs2):
        so, se = p.communicate(timeout=90)
        outs2.append({"pair": i, "rc": p.returncode, "stdout": so[-600:], "stderr": se[-600:]})
    hw["test_2x_2chip_concurrent"] = outs2

    return hw


def stage_models(bucket: str, cache_root: Path) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    hf_token = os.environ.get("HF_TOKEN") or None
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    cache_root.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=hf_token)

    candidates = [
        ("black-forest-labs/FLUX.1-schnell", True),
        ("black-forest-labs/FLUX.1-dev", True),
        ("Tongyi-MAI/Z-Image-Turbo", True),
        ("stabilityai/stable-diffusion-3.5-large", False),
    ]
    results: dict[str, Any] = {}
    for repo_id, stage_now in candidates:
        entry: dict[str, Any] = {"repo_id": repo_id}
        try:
            mi = api.model_info(repo_id=repo_id, token=hf_token)
            entry["accessible"] = True
            entry["gated"] = getattr(mi, "gated", False)
            siblings = [s.rfilename for s in (mi.siblings or [])]
            entry["num_files"] = len(siblings)
        except Exception as exc:
            entry["accessible"] = False
            entry["error"] = str(exc)
            log(f"Model {repo_id} access check failed: {exc}")
            results[repo_id] = entry
            continue

        if not stage_now:
            results[repo_id] = entry
            continue

        short_name = repo_id.split("/")[-1]
        local_dir = cache_root / short_name
        marker = local_dir / ".ready"
        t0 = time.time()
        if not marker.exists():
            log(f"Checking GCS cache gs://{bucket}/models/{repo_id}/ ...")
            fast_gcs_sync_dir(bucket, f"models/{repo_id}/", local_dir)
            log(f"Downloading / verifying {repo_id} into {local_dir} ...")
            # Ignore single-file monolithic safetensors (`flux1-schnell.safetensors` / `flux1-dev.safetensors` / `ae.safetensors`)
            # at repo root because Diffusers uses `transformer/`, `text_encoder/`, `text_encoder_2/`, `vae/`!
            ignore = [
                "*.mp4",
                "*.mov",
                "*.gif",
                "*.png",
                "*.jpg",
                "*.jpeg",
                "flux1-schnell.safetensors",
                "flux1-dev.safetensors",
                "ae.safetensors",
            ]
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                ignore_patterns=ignore,
                token=hf_token,
                max_workers=16,
            )
            marker.write_text(utc_now() + "\n")
        dt = round(time.time() - t0, 2)
        total_sz = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file() and "/.cache/" not in p.as_posix())
        entry["local_dir"] = str(local_dir)
        entry["size_gb"] = round(total_sz / 1e9, 2)
        entry["stage_s"] = dt
        log(f"Staged {repo_id} -> {local_dir} ({entry['size_gb']} GB in {dt}s)")
        # Mirror cleanly to regional GCS so Spot restarts hydrate at >10 GB/s without hitting HF rate limits!
        mirror_dir_to_gcs(bucket, local_dir, f"models/{repo_id}/")
        results[repo_id] = entry

    # If FLUX.1-dev was gated/inaccessible on hf-token, also stage SD3.5-large if accessible
    if not results.get("black-forest-labs/FLUX.1-dev", {}).get("accessible"):
        log("FLUX.1-dev gated on current HF_TOKEN; Tongyi-MAI/Z-Image-Turbo staged as un-gated second model!")

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-bucket", default="ank-usc1-bucket-314837540096")
    args = ap.parse_args()

    cache_root = Path("/dev/shm/hf_cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    vllm_probe_path = Path("/dev/shm/probe_vllm_omni.json")
    vllm_probe = json.loads(vllm_probe_path.read_text()) if vllm_probe_path.exists() else {"missing": True}

    log("1/3 Probing sglang-torchtpu internals...")
    sglang_probe = probe_sglang_internals()

    log("2/3 Probing TPU v7x-8 hardware & sub-host multi-process partitioning...")
    hw_probe = probe_tpu_hardware_and_partitioning()

    log("3/3 Staging FLUX.1-schnell, FLUX.1-dev, and Z-Image-Turbo into /dev/shm/hf_cache + GCS...")
    staged = stage_models(args.gcs_bucket, cache_root)

    report = {
        "timestamp": utc_now(),
        "hostname": os.uname().nodename,
        "shm_df": subprocess.check_output(["df", "-h", "/dev/shm"], text=True),
        "staged_models": staged,
        "hardware_probe": hw_probe,
        "sglang_probe": sglang_probe,
        "vllm_omni_probe": vllm_probe,
    }
    raw = json.dumps(report, indent=2).encode("utf-8")
    Path("/dev/shm/flux_stage_probe_report.json").write_bytes(raw)
    gcs_upload_bytes(args.gcs_bucket, "flux-serving-results/probe/stage_probe_report.json", raw)
    log("Uploaded full probe & staging report to gs://ank-usc1-bucket-314837540096/flux-serving-results/probe/stage_probe_report.json")


if __name__ == "__main__":
    main()
