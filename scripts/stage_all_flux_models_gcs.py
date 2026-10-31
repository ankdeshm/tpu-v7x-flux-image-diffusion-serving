#!/usr/bin/env python3
"""Parallel Hugging Face Hub -> regional GCS (`gs://ank-usc1-bucket-314837540096/models/`) stager
for FLUX.1-schnell, FLUX.1-dev, and Z-Image-Turbo (Diffusers shards only, skipping duplicate monolithic root files).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
import requests
from google.cloud import storage
from huggingface_hub import HfApi, hf_hub_url

BUCKET = "ank-usc1-bucket-314837540096"
PREFIX = "models"
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Skip duplicate root-level monolithic single-file weights (`flux1-schnell.safetensors`, `flux1-dev.safetensors`, `ae.safetensors`)
# since Diffusers loads `transformer/`, `text_encoder/`, `text_encoder_2/`, `vae/` (~33.8 GB instead of ~58 GB!).
SKIP_EXACT = {
    "flux1-schnell.safetensors",
    "flux1-dev.safetensors",
    "ae.safetensors",
}
DROP_EXTS = (".mp4", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".mov")

TARGETS = [
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "Tongyi-MAI/Z-Image-Turbo",
]


def stage_repo(repo_id: str) -> dict:
    api = HfApi(token=HF_TOKEN)
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    except Exception as exc:
        print(f"[SKIP/GATED] {repo_id}: {exc}", flush=True)
        return {"repo_id": repo_id, "accessible": False, "error": str(exc)}

    # Also test reading 1 small file (`model_index.json`) to verify gated license acceptance!
    test_url = hf_hub_url(repo_id=repo_id, filename="model_index.json")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    r_test = requests.get(test_url, headers=headers, timeout=30)
    if r_test.status_code != 200:
        msg = f"HTTP {r_test.status_code}: {r_test.text[:300]}"
        print(f"[GATED/DENIED] {repo_id}: {msg}", flush=True)
        return {"repo_id": repo_id, "accessible": False, "gated": getattr(info, "gated", False), "error": msg}

    siblings = []
    for s in info.siblings:
        fn = s.rfilename
        if fn in SKIP_EXACT or fn.lower().endswith(DROP_EXTS) or "/docs/" in fn or fn.startswith("docs/"):
            continue
        siblings.append(s)

    total_bytes = sum((s.size or 0) for s in siblings)
    print(
        f"\n=== Staging {repo_id}: {len(siblings)} files ({total_bytes / 1e9:.2f} GB) -> gs://{BUCKET}/{PREFIX}/{repo_id}/ ===",
        flush=True,
    )

    def xfer(s):
        client = storage.Client()
        blob = client.bucket(BUCKET).blob(f"{PREFIX}/{repo_id}/{s.rfilename}")
        if blob.exists():
            blob.reload()
            if (s.size is None and blob.size is not None) or ((s.size or 0) > 0 and blob.size == s.size):
                return (s.rfilename, "SKIP", blob.size or 0)
        url = hf_hub_url(repo_id=repo_id, filename=s.rfilename)
        t0 = time.time()
        for attempt in range(1, 5):
            written = 0
            try:
                with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
                    resp.raise_for_status()
                    with blob.open("wb", chunk_size=64 * 1024 * 1024) as f:
                        for chunk in resp.iter_content(chunk_size=16 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                dt = max(time.time() - t0, 1e-3)
                if written >= 50 * 1024 * 1024:
                    print(
                        f"  [OK] {repo_id}/{s.rfilename}: {written / 1e9:.2f} GB in {dt:.1f}s ({(written / 1048576) / dt:.0f} MiB/s)",
                        flush=True,
                    )
                return (s.rfilename, "OK", written)
            except Exception as exc:
                print(f"  [RETRY {attempt}] {repo_id}/{s.rfilename}: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(3 * attempt)

    t_repo = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(xfer, s) for s in siblings]
        done = 0
        bytes_done = 0
        for fut in concurrent.futures.as_completed(futs):
            _, _, sz = fut.result()
            done += 1
            bytes_done += sz
            if done % 10 == 0 or done == len(siblings):
                print(
                    f"  [{repo_id}] {done}/{len(siblings)} files ({bytes_done / 1e9:.2f}/{total_bytes / 1e9:.2f} GB) in {time.time() - t_repo:.1f}s",
                    flush=True,
                )

    storage.Client().bucket(BUCKET).blob(f"{PREFIX}/{repo_id}/_FLUX_SERVE_STAGE_DONE.txt").upload_from_string("ok\n")
    dt_tot = round(time.time() - t_repo, 2)
    print(f"=== COMPLETE {repo_id} in {dt_tot}s ===", flush=True)
    return {
        "repo_id": repo_id,
        "accessible": True,
        "files": len(siblings),
        "size_gb": round(total_bytes / 1e9, 2),
        "stage_s": dt_tot,
    }


if __name__ == "__main__":
    summary = {}
    for repo in TARGETS:
        summary[repo] = stage_repo(repo)
    # If FLUX.1-dev is gated on this HF_TOKEN, check stabilityai/stable-diffusion-3.5-large or Qwen/Qwen-Image too
    if not summary.get("black-forest-labs/FLUX.1-dev", {}).get("accessible"):
        for fallback in ("Qwen/Qwen-Image", "stabilityai/stable-diffusion-3.5-large"):
            res = stage_repo(fallback)
            summary[fallback] = res
            if res.get("accessible"):
                break
    storage.Client().bucket(BUCKET).blob("flux-serving-results/probe/gcs_model_stage_summary.json").upload_from_string(
        json.dumps(summary, indent=2), content_type="application/json"
    )
    print("Uploaded staging summary to gs://ank-usc1-bucket-314837540096/flux-serving-results/probe/gcs_model_stage_summary.json")
