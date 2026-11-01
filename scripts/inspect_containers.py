#!/usr/bin/env python3
"""Inspect sglang-torchtpu-diffusion:v0.2.0 and vllm-torchtpu-omni:v0.32.0 internals for FLUX & Image Diffusion."""

from __future__ import annotations

import importlib.metadata as md
import json
import sys
from pathlib import Path


def collect_versions() -> dict[str, str]:
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


def inspect_sglang() -> dict:
    info: dict = {"mode": "sglang", "versions": collect_versions()}
    import sglang
    import sglang_tpu

    sg_root = Path(sglang.__file__).parent
    sgt_root = Path(sglang_tpu.__file__).parent
    mm_dir = sg_root / "multimodal_gen"
    info["mm_files"] = [str(p.relative_to(mm_dir)) for p in sorted(mm_dir.rglob("*.py"))] if mm_dir.exists() else []
    info["sgt_files"] = [str(p.relative_to(sgt_root)) for p in sorted(sgt_root.rglob("*.py"))]

    snippets = {}
    if mm_dir.exists():
        for p in sorted(mm_dir.rglob("*.py")):
            rel = str(p.relative_to(mm_dir))
            low = rel.lower()
            if any(
                k in low
                for k in (
                    "server_args",
                    "image",
                    "openai",
                    "http_server",
                    "flux",
                    "zimage",
                    "z_image",
                    "registry",
                    "parallel",
                    "quant",
                    "attention",
                    "scheduler",
                )
            ):
                snippets[f"sglang/{rel}"] = p.read_text(errors="replace")[:15000]
    for p in sorted(sgt_root.rglob("*.py")):
        rel = str(p.relative_to(sgt_root))
        snippets[f"sglang_tpu/{rel}"] = p.read_text(errors="replace")[:15000]
    info["snippets"] = snippets
    return info


def inspect_vllm_omni() -> dict:
    info: dict = {"mode": "vllm_omni", "versions": collect_versions()}
    import vllm_omni

    vo_root = Path(vllm_omni.__file__).parent
    info["vo_files"] = [str(p.relative_to(vo_root)) for p in sorted(vo_root.rglob("*.py"))]
    snippets = {}
    for p in sorted(vo_root.rglob("*.py")):
        rel = str(p.relative_to(vo_root))
        low = rel.lower()
        if any(
            k in low
            for k in (
                "flux",
                "zimage",
                "z_image",
                "image",
                "serving",
                "openai",
                "cli",
                "args",
                "engine_args",
                "omni",
                "parallel",
                "quant",
                "attention",
                "tpu",
                "registry",
                "diffusion",
            )
        ):
            snippets[f"vllm_omni/{rel}"] = p.read_text(errors="replace")[:15000]
    info["snippets"] = snippets
    return info


if __name__ == "__main__":
    mode = sys.argv[1]
    out_path = Path(sys.argv[2])
    data = inspect_sglang() if mode == "sglang" else inspect_vllm_omni()
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {mode} inspection ({out_path.stat().st_size} bytes) to {out_path}")
