# VERSIONS
sglang: {'torch': '2.13.0+cpu', 'torch-tpu': '0.2.0', 'libtpu': '0.0.50', 'jax': '0.11.2', 'jaxlib': '0.11.2', 'diffusers': '0.37.0', 'transformers': '5.19.0', 'sglang': '0.0.0.dev1+gb3d718cb0.d20261031', 'sglang-tpu': '0.0.0.dev1+gb3d718cb0.d20261031', 'vllm': '0.32.0+empty'}
vllm_omni: {'torch': '2.13.0+cpu', 'torch-tpu': '0.2.0', 'libtpu': '0.0.50', 'jax': '0.11.2', 'jaxlib': '0.11.2', 'diffusers': '0.40.0', 'transformers': '5.19.0', 'accelerate': '1.12.0', 'vllm': '0.32.0+empty', 'vllm-omni': '0.32.0rc1'}

## sglang_tpu/multimodal_gen/__init__.py
```python
# SPDX-License-Identifier: Apache-2.0
"""TPU implementations for sglang.multimodal_gen.

:func:`activate` is the ``tpu_oot`` entry point of
``sglang.multimodal_gen.platforms``.
"""

PLATFORM_CLASS_PATH = "sglang_tpu.multimodal_gen.platform.TpuOOTPlatform"


def activate() -> str | None:
    """Select :class:`~sglang_tpu.multimodal_gen.platform.TpuOOTPlatform` on TPU."""
    import torch

    # torch_tpu autoloads with torch and registers torch.tpu only when it finds
    # hardware; the built-in TPU detection checks the same.
    if getattr(torch, "tpu", None) is None:
        return None
    return PLATFORM_CLASS_PATH

```

## sglang_tpu/multimodal_gen/platform.py
```python
# SPDX-License-Identifier: Apache-2.0
"""multimodal_gen's TPU platform as an out-of-tree plugin.

The ``tpu_oot`` entry point of ``sglang.multimodal_gen.platforms`` selects
:class:`TpuOOTPlatform` on TPU hosts (see :func:`sglang_tpu.multimodal_gen.activate`).
It is the in-tree ``TpuPlatform`` with ``PlatformEnum.OOT`` plus the plugin's
TPU behavior. ``is_tpu()`` stays true, so core TPU paths keep applying.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang_tpu.multimodal_gen import worker

logger = init_logger(__name__)

# The --attention-backend name of the Pallas backend (AttentionBackendEnum.OOT).
SPLASH_ATTN_BACKEND_NAME = "tpu_splash_attn"
_SPLASH_ATTN_BACKEND_CLS = (
    "sglang_tpu.multimodal_gen.tpu_splash_attn.TpuSplashAttentionBackend"
)


class TpuOOTPlatform(TpuPlatform):
    _enum = PlatformEnum.OOT

    def is_tpu(self) -> bool:
        # Kernel fallbacks, CustomOp.forward_tpu and the TPU argument checks
        # key on is_tpu().
        return True

    def get_dispatch_key_name(self) -> str:
        # CustomOp dispatch takes forward_tpu where an op defines it, as in-tree.
        return "tpu"

    @classmethod
    def get_all_to_all_communicator_cls(cls) -> str:
        # The base communicator's all-to-all, as the in-tree TPU platform uses.
        return Platform.get_device_communicator_cls()

    @classmethod
    def get_oot_attn_backend_name(cls) -> str | None:
        return SPLASH_ATTN_BACKEND_NAME

    @classmethod
    def supports_safetensors_device_load(cls) -> bool:
        # safetensors rejects 'tpu:N'.
        return False

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        if selected_backend is AttentionBackendEnum.OOT:
            logger.info(
                "Using TPU Pallas splash attention backend (falls back to SDPA "
                "per shape, or entirely off TPU v7x)."
            )
            return _SPLASH_ATTN_BACKEND_CLS
        return super().get_attn_backend_cls_str(selected_backend, head_size, dtype)

    @classmethod
    def _init_worker_process(cls) -> None:
        super()._init_worker_process()
        # The torch_tpu backend compiles static shapes only. The out-of-tree
        # compile path leaves dynamic=None, so keep a recompile for a new shape
        # static too.
        import torch._dynamo

        torch._dynamo.config.automatic_dynamic_shapes = False
        worker.install_runtime_patches()

    def get_compile_backend(self, mode: str | None = None) -> str:
        if mode is not None:
            logger.info_once(
                "torch.compile mode=%s is ignored by the torch_tpu backend.", mode
            )
        return super().get_compile_backend(mode)

    def apply_server_args_defaults(self, server_args) -> None:
        # The compile warmup layerwise-offloads the DiT, which TPU cannot run.
        if (
            server_args.is_arg_explicitly_set("offload_during_compile")
            and server_args.offload_during_compile
        ):
            logger.warning(
                "--offload-during-compile is not supported on TPU (the compile "
                "warmup cannot layerwise-offload the DiT); disabling it."
            )
        server_args.offload_during_compile = False

    def supports_teacache_with_torch_compile(self) -> bool:
        # Every TeaCache state update is a full XLA recompile under the
        # torch_tpu backend, up to the recompile limit.
        return False

    @classmethod
    def supports_vae_torch_compile(cls) -> bool:
        # Off for every VAE: the torch_tpu backend fails to compile the diffusers
        # AutoencoderKL decoder (torch_tpu 0.2.0rc4, libtpu 0.0.50: view() cannot
        # turn a channels-last [1, 512, 128, 128] activation into [1, 32, 262144]),
        # and the other VAEs are untested under it.
        return False

```

## sglang_tpu/multimodal_gen/worker.py
```python
# SPDX-License-Identifier: Apache-2.0
"""Worker-side steps of the diffusion TPU platform (``TpuOOTPlatform``)."""


def install_runtime_patches() -> None:
    """Patch a worker; call after ``torch_tpu`` is imported."""
    import torch

    from sglang_tpu.multimodal_gen import rope

    # torch_tpu runs float32 matmuls and convolutions as a single bf16 pass
    # ("medium", 2.3e-3 relative error); CUDA runs them in float32. Keep the
    # float32 paths (VAE decode, embeddings) float32.
    torch.set_float32_matmul_precision("highest")
    rope.install()

```

## sglang_tpu/multimodal_gen/tpu_splash_attn.py
```python
# SPDX-License-Identifier: Apache-2.0
"""TPU Pallas attention backend (``--attention-backend tpu_splash_attn``).

A flash attention kernel with an online exp2 softmax and a v7x block-size
model for non-causal attention over at least 512 keys, and JAX's
``make_splash_mha`` for causal attention, queries shorter than 256 rows and
cross-attention to fewer than 512 aligned keys. Both run as ``cpu_export_jax_op`` torch ops; block sizes,
padding and query scaling are resolved inside the jitted JAX function from the
static shapes.
"""

import functools
import math
import threading

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    SDPABackend,
    SDPAImpl,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

LOG2_E = math.log2(math.e)
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
# Block sizes and the VMEM model below are tuned and validated on v7x only.
_SUPPORTED_PALLAS_DEVICE_KINDS = frozenset({"TPU7x"})

_SPLASH_AVAILABLE: bool | None = None
_logged_sdpa_fallbacks: set[tuple] = set()

_FLASH_OP = None
_SPLASH_MHA_OP = None
_OPS_LOCK = threading.Lock()
_resolved_mode_logged = False


def _log_resolved_mode_once(use_splash: bool, dropout: float) -> None:
    global _resolved_mode_logged
    if _resolved_mode_logged:
        return
    _resolved_mode_logged = True
    if use_splash:
        logger.info("TPU splash attention: splash=on (per-shape SDPA fallback).")
    else:
        reason = f"dropout={dropout}" if dropout != 0.0 else "kernels unavailable"
        logger.info(
            "TPU splash attention: splash=off (%s); every call uses SDPA.", reason
        )


def _check_splash_available() -> bool:
    global _SPLASH_AVAILABLE
    if _SPLASH_AVAILABLE is not None:
        return _SPLASH_AVAILABLE
    try:
        import jax  # noqa: F401
        from jax.experimental import pallas  # noqa: F401
        from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa: F401
            splash_attention_kernel,
        )

        from sglang_tpu.pallas_export import (
            get_pallas_tpu_device_kind,
        )
        from sglang_tpu.pallas_jax_op import (  # noqa: F401
            cpu_export_jax_op,
        )

        device_kind = get_pallas_tpu_device_kind()
        if device_kind not in _SUPPORTED_PALLAS_DEVICE_KINDS:
            raise RuntimeError(f"kernels are tuned for TPU v7x only, not {device_kind}")
        _SPLASH_AVAILABLE = True
    except Exception as exc:
        logger.warning("TPU Pallas splash attention unavailable (%s); using SDPA.", exc)
        _SPLASH_AVAILABLE = False
    return _SPLASH_AVAILABLE


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _round_down(x: int, multiple: int) -> int:
    """Round ``x`` down to a multiple, but never below one ``multiple``.

    For ``x < multiple`` this rounds *up* to ``multiple``.
    """
    return max((x // multiple) * multiple, multiple)


# TPU v7x VMEM per TensorCore. The kernel passes the live
# ``get_tpu_info().vmem_capacity_bytes`` to both the block-size model and
# Mosaic's ``vmem_limit_bytes``; this default serves CPU-side callers.
_V7X_VMEM_CAPACITY_BYTES = 64 * 2**20
# Headroom for buffers the model does not count, chiefly the double-buffered
# bf16 output block (3.1 MiB at block_q=6400).
_VMEM_BUDGET_FRACTION = 0.93


def _score_tile_bytes(
    block_q: int,
    block_kv_compute: int,
    head_dim: int = 128,
    block_kv: int = 2048,
    bytes_per_element: int = 2,
) -> int:
    score_f32 = block_kv_compute * block_q * 4
    qkv_2x = 2 * (block_q + 2 * block_kv) * head_dim * bytes_per_element
    output_f32 = block_q * head_dim * 4
    max_sum_scratch = 2 * 8 * block_q * 4
    return score_f32 + qkv_2x + output_f32 + max_sum_scratch


def _autotune_block_sizes(
    q_seq: int,
    kv_seq: int,
    head_dim: int = 128,
    vmem_capacity_bytes: int = _V7X_VMEM_CAPACITY_BYTES,
) -> tuple[int, int, int]:
    """``(block_q, block_kv, block_kv_compute)`` of the flash kernel on TPU v7x.

    Minimizes grid-cell cost under a VMEM budget, keeping at least two query
    blocks per head. The VMEM model holds on v7x only.
    """
    mxu_dim = 256
    safe_kv_block = 2048
    vmem_budget = int(vmem_capacity_bytes * _VMEM_BUDGET_FRACTION)
    # Cap block_q: the VMEM model leaves out the double-buffered bf16 output
    # block (4 * block_q * head_dim bytes). For short KV the model alone would
    # allow block_q up to ~19.7k at block_kv=512 and overflow once that block
    # is added. 9472 is the largest size validated on device (37800x512,
    # 18900x600).
    max_block_q = 9472

    if q_seq < mxu_dim or kv_seq < mxu_dim:
        return 256, 256, 256

    block_kv = _round_down(min(safe_kv_block, kv_seq), mxu_dim)
    block_kv_compute = block_kv

    vmem_max = mxu_dim
    bq = mxu_dim
    while bq <= min(q_seq, max_block_q):
        if _score_tile_bytes(bq, block_kv_compute, head_dim, block_kv) <= vmem_budget:
            vmem_max = bq
            bq += mxu_dim
        else:
            break

    candidates = [
        c for c in range(mxu_dim, vmem_max + 1, mxu_dim) if math.ceil(q_seq / c) >= 2
    ]
    if not candidates:
        candidates = [min(vmem_max, _round_down(q_seq, mxu_dim))]

    cell_overhead = mxu_dim * safe_kv_block

    def _cost(cand_bq: int) -> tuple[int, int]:
        cells = math.ceil(q_seq / cand_bq) * math.ceil(kv_seq / block_kv)
        return (cells * (cand_bq * block_kv + cell_overhead), -cand_bq)

    best_bq = min(candidates, key=_cost)
    return best_bq, block_kv, block_kv_compute


def _use_custom_kernel(q_seq: int, kv_seq: int, is_causal: bool) -> bool:
    """Non-causal attention with at least 512 keys: long-KV self-attention, SP
    query shards and text cross-attention. The kernel masks ragged edge
    blocks, so no query or KV padding is needed."""
    return (not is_causal) and q_seq >= 256 and kv_seq >= 512


def _cross_attn_block_kv(kv_seq: int) -> int:
    """KV block dividing ``kv_seq``, a multiple of 128 below 512.

    Only kv in {128, 256, 384} reaches here: 512 or more keys go to the
    custom kernel.
    """
    return 256 if kv_seq % 256 == 0 else 128


def _splash_mha_block_sizes(
    q_seq: int, kv_seq: int, is_causal: bool
) -> tuple[int, int]:
    """``(block_q, block_kv)`` of ``make_splash_mha`` for a non-custom shape."""
    # Long query against short 128-aligned KV: block_kv divides kv_seq.
    if not is_causal and q_seq >= 2048 and kv_seq % 128 == 0:
        return 2048, _cross_attn_block_kv(kv_seq)
    # Causal attention, and short queries against long KV.
    if q_seq % 1024 == 0 and kv_seq % 1024 == 0:
        return 1024, 1024
    if q_seq >= 512 and kv_seq >= 512:
        return 512, 512
    return 256, 256


def _flash_attention(q, k, v, scale):
    """Flash attention with an online exp2 softmax on ``[B, H, S, D]`` arrays.

    Takes the unscaled query: the softmax scale is applied to the float32
    scores. Inputs are not padded; Mosaic masks the ragged query edge block and
    the last KV block reads only real keys.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    num_q_heads, q_seq, head_dim = q.shape[1:]
    num_kv_heads, kv_seq = k.shape[1], k.shape[2]
    q_heads_per_kv_head = num_q_heads // num_kv_heads
    # One VMEM figure feeds both the block-size model and Mosaic's limit.
    vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    block_q, block_kv, block_kv_compute = _autotune_block_sizes(
        q_seq, kv_seq, head_dim, vmem_capacity_bytes
    )
    grid_width = pl.cdiv(kv_seq, block_kv)
    qk_scale = scale * LOG2_E
    mask_value = -0.7 * float(jnp.finfo(jnp.float32).max)
    logger.info(
        "TPU splash attention: q_seq=%d kv_seq=%d -> flash kernel blocks=%s",
        q_seq,
        kv_seq,
        (block_q, block_kv, block_kv_compute),
    )

    def kernel(q_ref, k_ref, v_ref, o_ref, m_scratch_ref, l_scratch_ref, o_scratch_ref):
        float32 = jnp.float32
        sv_dims = (((0,), (0,)), ((), ()))
        j = pl.program_id(2)

        @pl.when(j == 0)
        def init():
            o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
            l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

        def update(slice_k):
            q_tile = q_ref[...]
            k_chunk = k_ref[slice_k, :]
            v_chunk = v_ref[slice_k, :]
            qk = lax.dot_general(
                k_chunk, q_tile, NT_DIM_NUMBERS, preferred_element_type=float32
            )
            m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
            # Max of raw scores (qk_scale > 0): qk * c - m is one FMA.
            m_curr = qk.max(axis=0)[None, :] * qk_scale
            m_next = jnp.maximum(m_prev, m_curr)
            s_curr = jnp.exp2(qk * qk_scale - m_next[0:1])
            l_curr = s_curr.sum(axis=0, keepdims=True)
            alpha = jnp.exp2(m_prev - m_next)
            o_curr = lax.dot_general(
                v_chunk,
                s_curr.astype(jnp.bfloat16),
                sv_dims,
                preferred_element_type=float32,
            )
            o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
            m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

        def compute_body(kv_compute_index, _):
            update(pl.ds(kv_compute_index * block_kv_compute, block_kv_compute))

        @pl.when(j != grid_width - 1)
        def body():
            lax.fori_loop(
                0, block_kv // block_kv_compute, compute_body, None, unroll=True
            )

        # The last KV block reads only real keys.
        @pl.when(j == grid_width - 1)
        def last_body():
            remain = kv_seq - (grid_width - 1) * block_kv
            full, tail = divmod(remain, block_kv_compute)
            lax.fori_loop(0, full, compute_body, None, unroll=True)
            if tail:
                update(pl.ds(full * block_kv_compute, tail))

        @pl.when(j == grid_width - 1)
        def end():
            # l is replicated across the sublanes; tile it to the [D, block_q]
            # scratch and write the [block_q, D] output tile.
            l_inv = jnp.tile(1.0 / l_scratch_ref[...], (head_dim // NUM_SUBLANES, 1))
            o_ref[...] = (o_scratch_ref[...] * l_inv).T.astype(o_ref.dtype)

    def attend(q_3d, k_3d, v_3d):
        kv_spec = pl.BlockSpec(
            (None, block_kv, head_dim),
            lambda h, i, j: (h // q_heads_per_kv_head, j, 0),
        )
        return pl.pallas_call(
            kernel,
            grid=(num_q_heads, pl.cdiv(q_seq, block_q), grid_width),
            in_specs=[
                pl.BlockSpec((None, block_q, head_dim), lambda h, i, j: (h, i, 0)),
                kv_spec,
                kv_spec,
            ],
            out_specs=pl.BlockSpec(
                (None, block_q, head_dim), lambda h, i, j: (h, i, 0)
            ),
            scratch_shapes=[
                pltpu.VMEM((NUM_SUBLANES, block_q), jnp.float32),
                pltpu.VMEM((NUM_SUBLANES, block_q), jnp.float32),
                pltpu.VMEM((head_dim, block_q), jnp.float32),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
                flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
                # Autotuned score tiles exceed Mosaic's 32 MiB default.
                vmem_limit_bytes=vmem_capacity_bytes,
            ),
            out_shape=jax.ShapeDtypeStruct((num_q_heads, q_seq, head_dim), q_3d.dtype),
            name="diffusion_flash_attention",
        )(q_3d, k_3d, v_3d)

    return jax.vmap(attend)(q, k, v)


def _splash_mha(q, k, v, scale, is_causal):
    """Stock ``make_splash_mha`` on ``[B, H, S, D]`` arrays.

    The query is scaled and padded to a block multiple; zero-padded keys are
    excluded by segment ids. Causal attention only reaches here with
    ``q_seq == kv_seq`` and equal blocks, so real rows never see padded keys.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
    )
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_mask as sam,
    )

    num_q_heads, q_seq = q.shape[1], q.shape[2]
    num_kv_heads, kv_seq = k.shape[1], k.shape[2]
    block_q, block_kv = _splash_mha_block_sizes(q_seq, kv_seq, is_causal)
    q_padded, kv_padded = _round_up(q_seq, block_q), _round_up(kv_seq, block_kv)
    logger.info(
        "TPU splash attention: q_seq=%d kv_seq=%d causal=%s -> make_splash_mha "
        "blocks=(%d, %d)",
        q_seq,
        kv_seq,
        is_causal,
        block_q,
        block_kv,
    )

    # make_splash_mha takes no scale, so q is pre-scaled and rounded to bf16:
    # up to 2^-9 relative error per element, which the flash kernel avoids by
    # scaling its f32 scores. The extra error grows with the logit spread
    # (test_splash_high_logit_gain bounds it).
    q = (q.astype(jnp.float32) * scale).astype(k.dtype)
    if num_q_heads != num_kv_heads:
        k = jnp.repeat(k, num_q_heads // num_kv_heads, axis=1)
        v = jnp.repeat(v, num_q_heads // num_kv_heads, axis=1)
    q = jnp.pad(q, ((0, 0), (0, 0), (0, q_padded - q_seq), (0, 0)))
    k = jnp.pad(k, ((0, 0), (0, 0), (0, kv_padded - kv_seq), (0, 0)))
    v = jnp.pad(v, ((0, 0), (0, 0), (0, kv_padded - kv_seq), (0, 0)))

    mask_cls = sam.CausalMask if is_causal else sam.FullMask
    kernel = splash.make_splash_mha(
        sam.MultiHeadMask([mask_cls((q_padded, kv_padded))] * num_q_heads),
        head_shards=1,
        q_seq_shards=1,
        block_sizes=splash.BlockSizes(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=min(block_kv, 512),
        ),
    )
    if kv_padded == kv_seq:
        out = jax.vmap(kernel)(q, k, v)
    else:
        segment_ids = splash.SegmentIds(
            q=jnp.ones((q_padded,), jnp.int32),
            kv=(jnp.arange(kv_padded) < kv_seq).astype(jnp.int32),
        )
        out = jax.vmap(functools.partial(kernel, segment_ids=segment_ids))(q, k, v)
    return out[:, :, :q_seq]


def _register_ops() -> None:
    """Create the kernel ops; must run before torch.compile traces a model.

    A failure marks the kernels unavailable, so callers fall back to SDPA
    instead of retrying a half-registered ``torch.library`` op.
    """
    global _FLASH_OP, _SPLASH_MHA_OP, _SPLASH_AVAILABLE
    if _FLASH_OP is not None or _SPLASH_AVAILABLE is False:
        return
    with _OPS_LOCK:
        if _FLASH_OP is n
```

## sglang/runtime/server_args/server_args.py
```python
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of sglang-diffusion Inference."""

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import field
from enum import Enum
from typing import Any, List, Literal, Optional

import addict
import yaml

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    LTX2PipelineConfig,
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.configs.quantization.nunchaku import NunchakuSVDQuantArgs
from sglang.multimodal_gen.configs.quantization.qvg_kv import QVGKVQuantArgs
from sglang.multimodal_gen.configs.utils import expand_path_fields
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.method_names import (
    canonical_quantization_method,
)
from sglang.multimodal_gen.runtime.loader.utils import BYTES_PER_GB
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
    SNAPSHOT_OFFLOAD,
    normalize_component_residency,
    resolve_component_residency_mode,
    resolve_diffusers_pipeline_offload,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    LAYERWISE_OFFLOAD_ALL_COMPONENTS,
    LAYERWISE_OFFLOAD_DIT_GROUP,
    LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_VAE_GROUP,
    RESIDENCY_LIFETIME_FORWARD,
    RESIDENCY_LIFETIME_PERMANENT,
    RESIDENCY_LIFETIMES,
    RESIDENCY_POLICIES,
    RESIDENCY_POLICY_LEADING,
    cpu_offload_component_matches,
    is_dit_component_name,
    is_image_encoder_component_name,
    is_legacy_dit_offload_component_name,
    is_text_encoder_component_name,
    is_vae_component_name,
    layerwise_component_matches_any_selection,
    normalize_cpu_offload_components,
    normalize_layerwise_offload_components,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args.auto_tune import (
    PERFORMANCE_MODES,
    ServerArgsAutoTuner,
)
from sglang.multimodal_gen.runtime.server_args.disagg import DisaggServerArgsMixin
from sglang.multimodal_gen.runtime.utils.argparse import (
    FlexibleArgumentParser,
    StoreBoolean,
)
from sglang.multimodal_gen.runtime.utils.common import (
    is_port_available,
    is_valid_ipv6_address,
    normalize_gpu_ids,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    _sanitize_for_logging,
    configure_logger,
    init_logger,
)
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE
from sglang.multimodal_gen.runtime.weights.source import (
    is_explicit_weight_file_reference,
)

logger = init_logger(__name__)

LTX2_TWO_STAGE_DEVICE_MODES = ("original", "resident")
LTX2_TWO_STAGE_DEVICE_MODE_CHOICES = LTX2_TWO_STAGE_DEVICE_MODES
LTX2_TWO_STAGE_PIPELINE_NAMES = ("LTX2TwoStagePipeline", "LTX2TwoStageHQPipeline")
# H200-class GPUs (>=130 GiB total) can usually keep both LTX2 DiTs resident.
LTX2_RESIDENT_AUTO_ENABLE_MEM_GB = 130
LORA_MERGE_MODES = ("auto", "merge", "dynamic")
MAX_SCHEDULER_RPC_TIMEOUT_S = 2_147_483
# Mirrors AttentionBackend.supports_ring_rotation; the name-level check
# runs before backend classes are importable on every platform.
RING_CAPABLE_ATTENTION_BACKENDS = ("fa", "sage_attn")


def _normalize_ltx2_two_stage_device_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    return mode.lower()


def is_ltx2_two_stage_pipeline_name(pipeline_class_name: str | None) -> bool:
    return pipeline_class_name in LTX2_TWO_STAGE_PIPELINE_NAMES


def _infer_direct_constructor_explicit_arg_names(server_args) -> set[str]:
    explicit_arg_names: set[str] = set()
    for attr in dataclasses.fields(server_args):
        if not attr.init or attr.name == "_explicit_arg_names":
            continue

        value = getattr(server_args, attr.name)
        if attr.default is not dataclasses.MISSING:
            default = attr.default
        elif attr.default_factory is not dataclasses.MISSING:
            default = attr.default_factory()
        else:
            explicit_arg_names.add(attr.name)
            continue

        if value != default:
            explicit_arg_names.add(attr.name)

    return explicit_arg_names


def _normalize_component_precisions(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("component_precisions must be a mapping")

    normalized: dict[str, str] = {}
    for component, precision in value.items():
        component_name = str(component).strip().replace("-", "_")
        precision_name = str(precision).strip().lower()
        if not component_name or precision_name not in PRECISION_TO_TYPE:
            raise ValueError(
                "Component precision entries require a component and one of "
                f"{sorted(PRECISION_TO_TYPE)}"
            )
        normalized[component_name] = precision_name
    return normalized


class Backend(str, Enum):
    """
    Enumeration for different model backends.
    - AUTO: Automatically select backend (prefer sglang native, fallback to diffusers)
    - SGLANG: Use sglang's native optimized implementation
    - DIFFUSERS: Use vanilla diffusers pipeline (supports all diffusers models)
    """

    AUTO = "auto"
    SGLANG = "sglang"
    DIFFUSERS = "diffusers"

    @classmethod
    def from_string(cls, value: str) -> "Backend":
        """Convert string to Backend enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid backend: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [backend.value for backend in cls]


WARMUP_MODES = ("off", "request", "server")

# Default prompt sequence-length buckets for breakable CUDA graph (BCG) padding.
# Prompt-conditioning is padded up to the smallest bucket that fits so prompts
# of different lengths share one captured graph.
DEFAULT_BCG_TEXT_BUCKETS = (64, 128, 256, 512, 1024)

# torch.distributed rendezvous port for a TPU slice, matching srt's DIST_PORT.
_TPU_DIST_PORT = 10011

BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS = frozenset(
    {
        "anima-base-v1.0-diffusers",
        "black-forest-labs/flux.1-dev",
        "circlestone-labs/anima-base-v1.0-diffusers",
        "comfy-org/ideogram-4",
        "efficient-large-model/sana1.5_1.6b_1024px_diffusers",
        "efficient-large-model/sana-video_2b_480p_diffusers",
        "sana1.5_1.6b_1024px_diffusers",
        "sana-video_2b_480p_diffusers",
        "fal/ideogram-v4-fast",
        "fal/ideogram-v4-instant",
        "flux.1-dev",
        "glm-image",
        "ideogram-4",
        "ideogram-4-fp8",
        "ideogram-4-nf4",
        "ideogram-v4-fast",
        "ideogram-v4-instant",
        "ideogram-ai/ideogram-4-fp8",
        "ideogram-ai/ideogram-4-nf4",
        "jdopensource/joyai-echo",
        "joyai-echo",
        "lightricks/ltx-2",
        "lightricks/ltx-2.3",
        "meituan-longcat/longcat-image",
        "meituan-longcat/longcat-image-edit-turbo",
        "ltx-2",
        "ltx-2.3",
        "minimax-h3",
        "minimaxai/minimax-h3",
        "inclusionai/ming-image-0.1-design",
        "inclusionai/ming-image-0.1-design-layer",
        "ming-image-0.1-design",
        "ming-image-0.1-design-layer",
        "qwen/qwen-image",
        "qwen/qwen-image-2512",
        "qwen/qwen-image-2.1",
        "qwen-image",
        "qwen-image-2512",
        "qwen-image-2.1",
        "tongyi-mai/z-image",
        "tongyi-mai/z-image-turbo",
        "zai-org/glm-image",
        "z-image",
        "z-image-turbo",
    }
)

BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS = frozenset(
    {
        "AnimaPipelineConfig",
        "FluxPipelineConfig",
        "GlmImagePipelineConfig",
        "Ideogram4PipelineConfig",
        "JoyEchoPipelineConfig",
        "LTX2PipelineConfig",
        "LTX23PipelineConfig",
        "LongCatImagePipelineConfig",
        "LongCatImageEditPipelineConfig",
        "MiniMaxH3PipelineConfig",
        "MingImagePipelineConfig",
        "MingImageLayerPipelineConfig",
        "QwenImagePipelineConfig",
        "QwenImage21PipelineConfig",
        "SanaPipelineConfig",
        "SanaVideoPipelineConfig",
        "ZImagePipelineConfig",
    }
)


def _normalized_bcg_model_refs(model_ref: str | None) -> set[str]:
    if not model_ref:
        return set()

    normalized = str(model_ref).strip().rstrip("/").lower()
    refs = {normalized, os.path.basename(normalized)}

    if "models--" in normalized:
        hf_cache_name = normalized.split("models--", 1)[1].split("/", 1)[0]
        refs.add(hf_cache_name.replace("--", "/"))

    return refs


@dataclasses.dataclass
class ServerArgs(DisaggServerArgsMixin):
    # Model and path configuration (for convenience)
    model_path: str
    model_subfolder: str | None = None
    model_variant: str | None = None

    # explicit model ID override (e.g. "Qwen-Image")
    model_id: str | None = None

    # served model name exposed via /v1/models and generation responses
    served_model_name: str | None = None

    # Model backend (sglang native or diffusers)
    backend: Backend = Backend.AUTO

    # Attention
    attention_backend: str = None
    # Time the viable attention backends on the model's own tensors during
    # warmup and keep the fastest per layer. Off by default: it has only been
    # measured on sm90 and sm12x.
    enable_attention_backend_autotune: bool = False
    attention_backend_config: addict.Dict | None = None
    component_attention_backends: dict[str, str] | str | None = field(
        default_factory=dict
    )
    _requested_component_attention_backends: dict[str, str] | None = field(
        default=None, repr=False, compare=False
    )
    cache_dit_config: str | dict[str, Any] | None = (
        None  # cache-dit config for diffusers
    )

    # Distributed executor backend
    nccl_port: Optional[int] = None
    enable_nccl_nvls: bool = False

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: str | None = None

    # Parallelism
    num_gpus: int = 1
    performance_mode: str = "auto"
    base_gpu_id: int = 0
    gpu_ids: list[int] | None = None
    # cross-node: num_gpus is the total world size across all nodes; each
    # node runs num_gpus // nnodes local GPU workers (mirrors srt's
    # tp_size_per_node convention)
    nnodes: int = 1
    node_rank: int = 0
    dist_init_addr: str | None = None
    tp_size: Optional[int] = None
    sp_degree: Optional[int] = None
    # sequence parallelism
    ulysses_degree: Optional[int] = None
    ring_degree: Optional[int] = None
    # rows split inside attention, exchanged with one K/V all-gather instead
    # of Ulysses a2a or ring rotation; auto-assigned at sp_degree=2 when no SP
    # degree is set explicitly
    kv_gather_degree: Optional[int] = None
    # whether the SP split was auto-assigned (lets layers fall back per call)
    sp_split_auto: bool = False
    # data parallelism
    # number of data parallelism groups
    dp_size: int = 1
    # number of gpu in a dp group
    # cfg parallel (None = auto-decide based on num_gpus)
    enable_cfg_parallel: Optional[bool] = None
    # number of GPUs in each CFG parallel group (None = auto, 1 = disabled, N > 1 = enabled)
    cfg_parallel_degree: Optional[int] = None

    # encoder layout across a multi-rank replica: auto | fold | dp | replicate
    # (see --encoder-parallel); fold shards the weights at load time, so it is
    # mutually exclusive with dp/replicate for the lifetime of the model
    encoder_parallel: str = "auto"

    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: Optional[int] = None
    dist_timeout: int | None = 3600  # 1 hour
    scheduler_rpc_timeout: int | None = None

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, repr=False)

    # Pipeline override
    pipeline_class_name: str | None = (
        None  # Override pipeline class from model_index.json
    )

    # LoRA parameters
    # (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated.
    lora_path: str | None = None
    lora_nickname: str = "default"  # for swapping adapters in the pipeline
    lora_scale: float = 1.0  # LoRA scale for merging (e.g., 0.125 for Hyper-SD)
    lora_alpha: int | None = None  # Override training alpha when metadata omits it
    lora_merge_mode: str = "auto"
    lora_weight_name: str | None = None

    # Component path overrides (key = model_index.json component name, value = path)
    component_paths: dict[str, str] = field(default_factory=dict)
    # Exact weight-file overrides retain the base component configuration.
    component_weights_paths: dict[str, str] = field(default_factory=dict)
    # Opt in one component to a loader-specific direct-GPU weight path.  The
    # existing --direct-gpu-weight-loading remains the primary-DiT control.
    component_direct_gpu_weight_loading: dict[str, bool] = field(default_factory=dict)
    # Explicit quantization override for one component. Self-describing
    # checkpoints remain auto-detected and do not need this override.
    component_quantizations: dict[str, str] = field(default_factory=dict)
    # Exact load and execution precision overrides for components whose native
    # loader advertises this capability.
    component_precisions: dict[str, str] = field(default_factory=dict)
    # Component-local layer name patterns to skip during online quantization.
    component_quantization_ignored_layers: dict[str, list[str]] = field(
        default_factory=dict
    )
    # Optional LTX-2.5 decoder is large enough to load only when requested.
    load_diffusion_decoder: bool = False

    # Pre-quantized transformer weights: safetensors file/directory or GGUF file.
    transformer_weights_path: str | None = None
    # path to precomputed MiniMax H3 AdaLN outputs for inference-only serving.
    minimax_h3_adaln_cache_path: str | None = None
    # Rebuild AdaLN outputs per request from the checkpoint, no sidecar needed.
    minimax_h3_adaln_online: bool = False
    # Widest timestep plan the rebuild slab is sized for; see
    # MINIMAX_H3_ADALN_MAX_PLAN_WIDTH.
    minimax_h3_adaln_plan_
```

## sglang/runtime/utils/quantization_utils.py
```python
import glob
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_fp8 import ComfyFp8Config
from sglang.multimodal_gen.runtime.layers.quantization.comfy_nvfp4 import (
    ComfyNvfp4Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
    ConvRotInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a4_config import (
    KitchenW4A4Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a8_config import (
    KitchenW4A8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.layers.linear import LinearBase as SrtLinearBase
from sglang.srt.layers.modelopt_utils import canonicalize_modelopt_quant_algo
from sglang.srt.layers.quantization.unquant import (
    UnquantizedEmbeddingMethod as SrtUnquantizedEmbeddingMethod,
)
from sglang.srt.layers.quantization.unquant import (
    UnquantizedLinearMethod as SrtUnquantizedLinearMethod,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as SrtVocabParallelEmbedding,
)
from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)
from sglang.srt.model_loader.post_load import stage_module_for_post_load
from sglang.srt.utils import is_npu

logger = init_logger(__name__)


def process_model_weights_after_loading(
    model: nn.Module,
    process_device: torch.device | None = None,
    *,
    quantized_only: bool = False,
) -> int:
    """Process native and SRT layers once, optionally staging one layer at a time."""
    processed_layers = 0
    for module in model.modules():
        if not isinstance(
            module,
            (
                LinearBase,
                SrtLinearBase,
                VocabParallelEmbedding,
                SrtVocabParallelEmbedding,
            ),
        ):
            continue
        method = module.quant_method
        if method is None:
            continue
        unquantized = isinstance(
            method,
            (
                UnquantizedLinearMethod,
                SrtUnquantizedLinearMethod,
                UnquantizedEmbeddingMethod,
                SrtUnquantizedEmbeddingMethod,
            ),
        )
        if quantized_only and unquantized:
            continue
        if is_npu() and not unquantized:
            torch.npu.config.allow_internal_format = True
        if process_device is None:
            method.process_weights_after_loading(module)
        else:
            with stage_module_for_post_load(module, process_device):
                method.process_weights_after_loading(module)
        if is_npu():
            torch.npu.empty_cache()
        processed_layers += 1
    return processed_layers


def _merge_comfy_quant_marker(
    markers: dict[str, dict[str, Any]], prefix: str, marker: dict[str, Any]
) -> None:
    # Headers may summarize a layer whose tensor marker includes more fields.
    previous = markers.setdefault(prefix, {})
    if any(key in previous and previous[key] != value for key, value in marker.items()):
        raise ValueError(f"Conflicting Comfy quantization markers for {prefix!r}")
    previous.update(marker)


def inspect_comfy_quant_markers(
    safetensors_list: list[str],
    param_name_mapper: Callable[[str], str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Read and validate Comfy's tensor-level quantization markers."""
    checkpoint_meta: dict[str, tuple[str, tuple[int, ...]]] = {}
    raw_markers: dict[str, dict[str, Any]] = {}
    marked_dtype_weight_prefixes: set[str] = set()
    global_quant_formats: set[str] = set()

    for path in safetensors_list:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if quant_format := metadata.get("quant_format"):
                global_quant_formats.add(quant_format.lower())
            serialized_metadata = metadata.get("_quantization_metadata")
            if serialized_metadata is not None:
                try:
                    metadata_config = json.loads(serialized_metadata)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid _quantization_metadata in {path}"
                    ) from exc
                if not isinstance(metadata_config, dict):
                    raise ValueError(
                        f"_quantization_metadata in {path} must contain an object"
                    )
                metadata_layers = metadata_config.get("layers")
                if not isinstance(metadata_layers, dict):
                    raise ValueError(
                        f"_quantization_metadata in {path} must contain a layers object"
                    )
                for prefix, marker in metadata_layers.items():
                    if not isinstance(marker, dict):
                        raise ValueError(
                            f"Comfy quantization metadata for {prefix!r} must be an object"
                        )
                    _merge_comfy_quant_marker(raw_markers, prefix, marker)
            for key in checkpoint.keys():
                tensor_slice = checkpoint.get_slice(key)
                checkpoint_meta[key] = (
                    tensor_slice.get_dtype(),
                    tuple(tensor_slice.get_shape()),
                )
                if key.endswith(".weight") and tensor_slice.get_dtype() in (
                    "F8_E4M3",
                    "I8",
                    "U8",
                ):
                    marked_dtype_weight_prefixes.add(key.removesuffix(".weight"))
                if not key.endswith(".comfy_quant"):
                    continue
                try:
                    marker = json.loads(checkpoint.get_tensor(key).numpy().tobytes())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"Invalid Comfy quantization marker {key!r} in {path}"
                    ) from exc
                if not isinstance(marker, dict):
                    raise ValueError(
                        f"Comfy quantization marker {key!r} must contain a JSON object"
                    )
                prefix = key.removesuffix(".comfy_quant")
                _merge_comfy_quant_marker(raw_markers, prefix, marker)

    if global_quant_formats == {"mxfp8"}:
        for prefix in marked_dtype_weight_prefixes:
            weight_meta = checkpoint_meta[f"{prefix}.weight"]
            scale_meta = checkpoint_meta.get(f"{prefix}.weight_scale")
            weight_dtype, weight_shape = weight_meta
            if (
                weight_dtype != "F8_E4M3"
                or len(weight_shape) != 2
                or weight_shape[1] % 32 != 0
                or scale_meta != ("U8", (weight_shape[0], weight_shape[1] // 32))
            ):
                raise ValueError(
                    f"MXFP8 layer {prefix!r} has incompatible weight/scale metadata: "
                    f"{weight_meta} and {scale_meta}"
                )
            raw_markers.setdefault(prefix, {"format": "mxfp8"})

    missing_markers = marked_dtype_weight_prefixes - raw_markers.keys()
    if missing_markers:
        raise ValueError(
            "Quantized weights are missing comfy_quant metadata: "
            f"{sorted(missing_markers)[:5]}"
        )

    for prefix, marker in raw_markers.items():
        marker_format = marker.get("format")
        required = {f"{prefix}.weight", f"{prefix}.weight_scale"}
        if marker_format == "asym_w4a8_int8":
            required = {
                f"{prefix}.weight",
                f"{prefix}.weight_s_rel",
                f"{prefix}.weight_s_channel",
            }
        if marker_format == "nvfp4":
            required.add(f"{prefix}.weight_scale_2")
        if marker_format not in (
            "float8_e4m3fn",
            "int8_tensorwise",
            "asym_w4a8_int8",
            "convrot_w4a4",
            "nvfp4",
        ):
            continue
        missing = required - checkpoint_meta.keys()
        if missing:
            raise ValueError(
                f"Comfy layer {prefix!r} is missing checkpoint tensors: "
                f"{sorted(missing)}"
            )
        if marker_format == "float8_e4m3fn":
            marker["_activation_scheme"] = (
                "static" if f"{prefix}.input_scale" in checkpoint_meta else "dynamic"
            )
            continue
        if marker_format == "asym_w4a8_int8":
            weight_dtype, weight_shape = checkpoint_meta[f"{prefix}.weight"]
            scale_dtype, scale_shape = checkpoint_meta[f"{prefix}.weight_s_rel"]
            channel_dtype, channel_shape = checkpoint_meta[f"{prefix}.weight_s_channel"]
            group_size = int(marker.get("group_size", 16))
            if group_size < 4:
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} has invalid group_size={group_size}"
                )
            if weight_dtype != "I8" or scale_dtype != "F8_E4M3":
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} needs I8 weights and FP8 "
                    f"group scales, got {weight_dtype} and {scale_dtype}"
                )
            if channel_dtype != "F32":
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} needs F32 channel scales, "
                    f"got {channel_dtype}"
                )
            if len(weight_shape) != 2:
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} needs a 2D packed weight, "
                    f"got {weight_shape}"
                )
            logical_input_size = weight_shape[1] * 2
            expected_scale_shape = (weight_shape[0], logical_input_size // group_size)
            if scale_shape != expected_scale_shape or channel_shape != (
                weight_shape[0],
            ):
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} has incompatible weight/scale "
                    f"shapes: {weight_shape}, {scale_shape}, and {channel_shape}"
                )
            codebook_key = f"{prefix}.weight_codebook"
            correction_key = f"{prefix}.weight_correction"
            marker["_has_codebook"] = codebook_key in checkpoint_meta
            marker["_has_correction"] = correction_key in checkpoint_meta
            if marker["_has_codebook"] and checkpoint_meta[codebook_key] != (
                "F32",
                (16,),
            ):
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} needs an F32[16] codebook"
                )
            expected_correction = (
                logical_input_size // group_size,
                weight_shape[0],
            )
            if marker["_has_correction"] and checkpoint_meta[correction_key] != (
                "F32",
                expected_correction,
            ):
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} has an incompatible correction tensor"
                )
            continue
        if marker_format == "convrot_w4a4":
            weight_dtype, weight_shape = checkpoint_meta[f"{prefix}.weight"]
            scale_dtype, scale_shape = checkpoint_meta[f"{prefix}.weight_scale"]
            if weight_dtype != "I8" or scale_dtype != "F32":
                raise ValueError(
                    f"Comfy W4A4 layer {prefix!r} needs I8 packed weights and "
                    f"F32 scales, got {weight_dtype} and {scale_dtype}"
                )
            if len(weight_shape) != 2 or scale_shape != (weight_shape[0],):
                raise ValueError(
                    f"Comfy W4A4 layer {prefix!r} has incompatible weight/scale "
                    f"shapes: {weight_shape} and {scale_shape}"
                )
            logical_input_size = weight_shape[1] * 2
            convrot_group_size = int(marker.get("convrot_groupsize", 256))
            if convrot_group_size not in (16, 64, 256):
                raise ValueError(
                    f"Comfy W4A4 layer {prefix!r} has unsupported "
                    f"convrot_groupsize={convrot_group_size}"
                )
            if logical_input_size % 64 or logical_input_size % convrot_group_size:
                raise ValueError(
                    f"Comfy W4A4 layer {prefix!r} has input size "
                    f"{logical_input_size}, incompatible with quant_group_size=64 "
                    f"and convrot_groupsize={convrot_group_size}"
                )
            continue
        if marker_format == "nvfp4":
            weight_dtype, weight_shape = checkpoint_meta[f"{prefix}.weight"]
            scale_dtype, scale_shape = checkpoint_meta[f"{prefix}.weight_scale"]
            scale_2_dtype, scale_2_shape = checkpoint_meta[f"{prefix}.weight_scale_2"]
            if weight_dtype != "U8" or scale_dtype != "F8_E4M3":
                raise ValueError(
                    f"Comfy NVFP4 layer {prefix!r} needs U8 packed weights and "
                    f"FP8 block scales, got {weight_dtype} and {scale_dtype}"
                )
            if scale_2_dtype != "F32" or scale_2_shape not in ((), (1,)):
                raise ValueError(
                    f"Comfy NVFP4 layer {prefix!r} needs a scalar F32 "
                    f"weight_scale_2, got {scale_2_dtype}{scale_2_shape}"
                )
            if len(weight_shape) != 2:
                raise ValueError(
                    f"Comfy NVFP4 layer {prefix!r} needs a 2D packed weight, "
                    f"got {weight_shape}"
                )
            logical_input_size = weight_shape[1] * 2
            expected_scale_shape = (weight_shape[0], logical_input_size // 16)
            if logical_input_size % 16 or scale_shape != expected_scale_shape:
                raise ValueError(
                    f"Comfy NVFP4 layer {prefix!r} has incompatible weight/scale "
                    f"shapes: {weight_shape} and {scale_shape}"
                )
            pre_quant_scale_key = f"{prefix}.pre_quant_scale"
            marker["_has_pre_quant_scale"] = pre_quant_scale_key in checkpoint_meta
            if marker["_has_pre_quant_scale"]:
             
```

## sglang/runtime/pipelines/flux.py
```python
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig
from sglang.multimodal_gen.configs.sample.flux import FluxSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    calculate_linear_shift,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux import (
    FluxProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def prepare_mu(batch: Req, server_args: ServerArgs):
    height = batch.height
    width = batch.width
    vae_scale_factor = (
        server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
    )
    image_seq_len = (int(height) // (vae_scale_factor * 2)) * (
        int(width) // (vae_scale_factor * 2)
    )

    return "mu", calculate_linear_shift(image_seq_len)


class FluxPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "FluxPipeline"

    # Used when the checkpoint is a single safetensors file with no
    # model_index.json to derive these from.
    pipeline_config_cls = FluxPipelineConfig
    sampling_params_cls = FluxSamplingParams

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_standard_t2i_stages(
            text_encoder_key=["text_encoder", "text_encoder_2"],
            tokenizer_key=["tokenizer", "tokenizer_2"],
            text_encoding_stage_name="prompt_encoding_stage_primary",
            prepare_extra_timestep_kwargs=[prepare_mu],
            progressive_denoising_stage_cls=FluxProgressiveDenoisingStage,
        )


EntryClass = FluxPipeline

```

## sglang/runtime/entrypoints/http_server.py
```python
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import base64
import os
import signal
import uuid
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

import httpx
import torch
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.action import api as action_api
from sglang.multimodal_gen.runtime.entrypoints.action import openpi
from sglang.multimodal_gen.runtime.entrypoints.openai import image_api, video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.prompt_enhancement import (
    PromptEnhancer,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VertexGenerateReqInput,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_video_api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.entrypoints.post_training import (
    rollout_api,
    weights_api,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    prepare_request,
    save_outputs,
)
from sglang.multimodal_gen.runtime.observability.metrics import configure_metrics
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.server_warmup import (
    run_async_client_warmup,
    should_run_synthetic_server_warmup,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    globally_suppress_loggers,
    init_logger,
)
from sglang.srt.utils.common import (
    add_prometheus_middleware,
    add_prometheus_track_response_middleware,
)
from sglang.srt.utils.json_response import orjson_response
from sglang.version import __version__

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)

VERTEX_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate")
SERVER_WARMUP_BYPASS_PATHS = (
    "/metrics",
    "/liveness",
    "/health",
    "/health_generate",
    "/model_info",
    "/server_info",
)


async def _wait_until_http_live(server_args: ServerArgs) -> None:
    """for server warmup"""
    liveness_url = f"{server_args.url()}/liveness"
    # Probe the local server directly: a loopback liveness check must never be
    # routed through an HTTP proxy. trust_env=False also avoids crashing startup
    # on a malformed proxy env var, since httpx parses *_PROXY/NO_PROXY when the
    # client is constructed (raising httpx.InvalidURL before any request). See #28493.
    async with httpx.AsyncClient(trust_env=False) as client:
        for _ in range(120):
            try:
                response = await client.get(liveness_url, timeout=5.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError(f"HTTP server did not become live at {liveness_url}")


async def _run_server_warmup_after_http_live(
    server_args: ServerArgs, warmup_done: asyncio.Event
) -> None:
    try:
        if not should_run_synthetic_server_warmup(server_args):
            warmup_done.set()
            return

        await _wait_until_http_live(server_args)

        await run_async_client_warmup(
            server_args,
            async_scheduler_client.forward,
            fail_open=server_args.warmup_resolutions is None,
        )
        logger.info("The server is fired up and ready to roll!")
        warmup_done.set()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error("Server warmup failed; aborting startup: %s", e, exc_info=True)
        os.kill(os.getpid(), signal.SIGTERM)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
        shutdown_video_jobs,
    )
    from sglang.multimodal_gen.runtime.scheduler_client import (
        async_scheduler_client,
        run_zeromq_broker,
    )

    # 1. Initialize the singleton client that connects to the backend Scheduler
    server_args = app.state.server_args
    async_scheduler_client.initialize(server_args)
    warmup_done = asyncio.Event()
    app.state.server_warmup_done = warmup_done

    # 2. Start the ZMQ Broker in the background to handle offline requests
    broker_task = asyncio.create_task(run_zeromq_broker(server_args))
    warmup_task = None
    if server_args.warmup_mode == "server":
        warmup_task = asyncio.create_task(
            _run_server_warmup_after_http_live(server_args, warmup_done)
        )
    else:
        warmup_done.set()

    try:
        app.state.prompt_enhancer = (
            PromptEnhancer.from_file(server_args.prompt_enhancer_config)
            if server_args.prompt_enhancer_config is not None
            else None
        )
        yield
    finally:
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task

        # On shutdown
        logger.info("FastAPI app is shutting down...")
        await shutdown_video_jobs()
        if app.state.prompt_enhancer is not None:
            await app.state.prompt_enhancer.close()
        broker_task.cancel()
        with suppress(asyncio.CancelledError):
            await broker_task
        async_scheduler_client.close()


# Health router
health_router = APIRouter()


@health_router.get("/liveness")
async def liveness():
    """Report that the HTTP server is accepting requests."""
    return {"status": "ok"}


@health_router.get("/health")
async def health(request: Request):
    """Report readiness for normal inference traffic."""
    if not request.app.state.server_warmup_done.is_set():
        return Response(status_code=503)
    return {"status": "ok"}


@health_router.get("/models", deprecated=True)
async def get_models(request: Request):
    """
    Get information about the model served by this server.

    .. deprecated::
        Use /v1/models instead for OpenAI-compatible model discovery.
        This endpoint will be removed in a future version.
    """
    from sglang.multimodal_gen.runtime.entrypoints.openai.common_api import (
        get_served_pipeline_class,
    )

    server_args: ServerArgs = request.app.state.server_args
    pipeline_cls = get_served_pipeline_class(server_args)

    response = {
        "model_path": server_args.model_path,
        "num_gpus": server_args.num_gpus,
        "task_type": server_args.pipeline_config.task_type.name,
        "supported_task_types": [
            task.name for task in server_args.pipeline_config.get_supported_task_types()
        ],
        "dit_precision": server_args.pipeline_config.dit_precision,
        "vae_precision": server_args.pipeline_config.vae_precision,
        "vae_decode_precision": server_args.pipeline_config.vae_decode_precision,
    }

    if pipeline_cls:
        response["pipeline_name"] = pipeline_cls.pipeline_name
        response["pipeline_class"] = pipeline_cls.__name__

    return response


@health_router.get("/server_info")
async def server_info_endpoint(request: Request):
    """Get server information.

    Returns fields compatible with the LLM engine's /server_info so that
    the model gateway can discover diffusion workers.
    """
    server_args: ServerArgs = request.app.state.server_args

    return {
        "model_path": server_args.model_path,
        "served_model_name": server_args.served_model_name,
        "tp_size": server_args.tp_size,
        "dp_size": server_args.dp_size,
        "version": __version__,
    }


@health_router.get("/model_info")
async def model_info_en
```

## sglang/runtime/entrypoints/openai/common_api.py
```python
import time
from typing import Any, List, Optional, Union

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from sglang.multimodal_gen.registry import get_model_info, get_pipeline_class
from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    UnmergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    format_lora_message,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils.json_response import orjson_response

router = APIRouter(prefix="/v1")
logger = init_logger(__name__)


class ModelCard(BaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sglang"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None


class DiffusionModelCard(ModelCard):
    """Extended ModelCard with diffusion-specific fields."""

    num_gpus: Optional[int] = None
    task_type: Optional[str] = None
    supported_task_types: list[str] | None = None
    dit_precision: Optional[str] = None
    vae_precision: Optional[str] = None
    pipeline_name: Optional[str] = None
    pipeline_class: Optional[str] = None


def get_served_pipeline_class(server_args: ServerArgs):
    """Resolve discovery metadata using the same explicit pipeline as serving."""
    if server_args.pipeline_class_name:
        return get_pipeline_class(server_args.pipeline_class_name)
    model_info = get_model_info(
        server_args.model_path,
        backend=server_args.backend,
        model_id=server_args.model_id,
    )
    return model_info.pipeline_cls if model_info else None


def _build_model_card(
    server_args: ServerArgs, served_model_name: str
) -> DiffusionModelCard:
    pipeline_cls = get_served_pipeline_class(server_args)
    card_kwargs: dict[str, Any] = {
        "id": served_model_name,
        "root": served_model_name,
        "num_gpus": server_args.num_gpus,
        "task_type": server_args.pipeline_config.task_type.name,
        "supported_task_types": [
            task.name for task in server_args.pipeline_config.get_supported_task_types()
        ],
        "dit_precision": server_args.pipeline_config.dit_precision,
        "vae_precision": server_args.pipeline_config.vae_precision,
    }
    if pipeline_cls:
        card_kwargs["pipeline_name"] = pipeline_cls.pipeline_name
        card_kwargs["pipeline_class"] = pipeline_cls.__name__
    return DiffusionModelCard(**card_kwargs)


async def _handle_lora_request(req: Any, success_msg: str, failure_msg: str):
    try:
        output: OutputBatch = await async_scheduler_client.forward(req)
        if output.error is None:
            return {"status": "ok", "message": success_msg}
        else:
            error_msg = output.error
            raise HTTPException(status_code=500, detail=f"{failure_msg}: {error_msg}")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error during '{failure_msg}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_lora")
async def set_lora(
    lora_nickname: Union[str, List[str]] = Body(..., embed=True),
    lora_path: Optional[Union[str, List[Optional[str]]]] = Body(None, embed=True),
    target: Union[str, List[str]] = Body("all", embed=True),
    strength: Union[float, List[float]] = Body(1.0, embed=True),
    merge_mode: Optional[str] = Body(None, embed=True),
    lora_alpha: Optional[Union[int, List[Optional[int]]]] = Body(None, embed=True),
):
    """
    Set LoRA adapter(s) for the specified transformer(s).
    Supports both single LoRA (backward compatible) and multiple LoRA adapters.

    Args:
        lora_nickname: The nickname(s) of the adapter(s). Can be a string or a list of strings.
        lora_path: Path(s) to the LoRA adapter(s) (local path or HF repo id).
            Can be a string, None, or a list of strings/None. Must match the length of lora_nickname.
        target: Which transformer(s) to apply the LoRA to. Can be a string or a list of strings.
            If a list, must match the length of lora_nickname. Valid values:
            - "all": Apply to all transformers (default)
            - "transformer": Apply only to the primary transformer (high noise for Wan2.2)
            - "transformer_2": Apply only to transformer_2 (low noise for Wan2.2)
            - "critic": Apply only to the critic model
        strength: LoRA strength(s) for merge, default 1.0. Can be a float or a list of floats.
            If a list, must match the length of lora_nickname. Values < 1.0 reduce the effect,
            values > 1.0 amplify the effect.
        merge_mode: Optional LoRA merge mode: "auto", "merge", or "dynamic".
        lora_alpha: Training alpha override for adapters that omit it from metadata.
    """
    req = SetLoraReq(
        lora_nickname=lora_nickname,
        lora_path=lora_path,
        target=target,
        strength=strength,
        merge_mode=merge_mode,
        lora_alpha=lora_alpha,
    )
    nickname_str, target_str, strength_str = format_lora_message(
        lora_nickname, target, strength
    )

    return await _handle_lora_request(
        req,
        f"Successfully set LoRA adapter(s): {nickname_str} (target: {target_str}, strength: {strength_str})",
        "Failed to set LoRA adapter",
    )


@router.post("/merge_lora_weights")
async def merge_lora_weights(
    target: str = Body("all", embed=True),
    strength: float = Body(1.0, embed=True),
):
    """
    Merge LoRA weights into the base model.

    Args:
        target: Which transformer(s) to merge. One of "all", "transformer",
                "transformer_2", "critic".
        strength: LoRA strength for merge, default 1.0. Values < 1.0 reduce the effect,
            values > 1.0 amplify the effect.
    """
    req = MergeLoraWeightsReq(target=target, strength=strength)
    return await _handle_lora_request(
        req,
        f"Successfully merged LoRA weights (target: {target}, strength: {strength})",
        "Failed to merge LoRA weights",
    )


@router.post("/unmerge_lora_weights")
async def unmerge_lora_weights(
    target: str = Body("all", embed=True),
):
    """
    Unmerge LoRA weights from the base model.

    Args:
        target: Which transformer(s) to unmerge. One of "all", "transformer",
                "transformer_2", "critic".
    """
    req = UnmergeLoraWeightsReq(target=target)
    return await _handle_lora_request(
        req,
        f"Successfully unmerged LoRA weights (target: {target})",
        "Failed to unmerge LoRA weights",
    )


@router.get("/model_info")
async def model_info():
    """Get the model information."""
    server_args = get_global_server_args()
    if not server_args:
        raise HTTPException(status_code=500, detail="Server args not initialized")

    result = {
        "model_path": server_args.model_path,
    }
    return result


@router.get("/list_loras")
async def list_loras():
    """List loaded LoRA adapters and current application status per module."""
    try:
        req = ListLorasReq()
        output: OutputBatch = await async_scheduler_client.forward(req)
        if output.error is None:
            return output.output or {}
        else:
            raise HTTPException(status_code=500, detail=output.error)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error during 'list_loras': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@ro
```

## sglang/runtime/entrypoints/openai/image_api.py
```python
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import base64
import contextlib
import os
import time
from typing import Any, List, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from PIL import Image

from sglang.multimodal_gen.configs.sample.glmimage import GlmImageSamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.entrypoints.openai.prompt_enhancement import (
    maybe_enhance_prompt,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import IMAGE_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    add_common_data_to_response,
    build_sampling_params,
    choose_output_image_ext,
    get_sampling_request_extra_fields,
    merge_image_input_list,
    process_generation_batch,
    request_extra_value,
    resolve_sampling_params_cls,
    save_image_to_path,
    temp_dir_if_disabled,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.srt.observability.trace import extract_trace_headers

router = APIRouter(prefix="/v1/images", tags=["images"])


def _get_extra_field(request, field_name):
    return request_extra_value(request, field_name)


def _get_request_field_or_extra(request, field_name):
    value = getattr(request, field_name, None)
    if value is not None:
        return value
    return _get_extra_field(request, field_name)


def _image_request_model_kwargs(
    request: ImageGenerationsRequest,
    sampling_params_cls: type[SamplingParams],
) -> dict[str, Any]:
    """Extract fields owned and declared by the active model contract."""

    kwargs = {}
    for field_name in get_sampling_request_extra_fields(sampling_params_cls, "image"):
        value = _get_extra_field(request, field_name)
        if value is not None:
            kwargs[field_name] = value
    return kwargs


def _runtime_sampling_quality(quality: str | None) -> str | None:
    """Keep OpenAI's automatic default out of SGLang's sampling contract."""
    return None if quality in (None, "auto") else quality


def _resolve_image_output_format(
    output_format: str | None, sampling_params_cls: type[SamplingParams]
) -> str | None:
    if output_format is not None:
        return output_format
    return sampling_params_cls.default_image_output_format()


def _read_b64_for_paths(paths: list[str]) -> list[str]:
    """Read and base64-encode each file. Must be called before cloud upload deletes them."""
    result = []
    for path in paths:
        with open(path, "rb") as f:
            result.append(base64.b64encode(f.read()).decode("utf-8"))
    return result


async def _upload_and_cleanup_images(paths: list[str]) -> list[str | None]:
    return await asyncio.gather(
        *(cloud_storage.upload_and_cleanup(path) for path in paths)
    )


def _fallback_image_urls(
    request_id: str, num_outputs: int, is_persistent: bool
) -> list[str] | None:
    if not is_persistent:
        return None
    if num_outputs <= 1:
        return [f"/v1/images/{request_id}/content"]
    return [
        f"/v1/images/{request_id}/content?variant={idx}" for idx in range(num_outputs)
    ]


def _select_image_variant_path(item: dict, variant: str | None) -> str | None:
    file_paths = item.get("file_paths")
    if file_paths:
        variant_idx = _image_variant_index(variant)
        if variant_idx is None:
            return None
        if variant_idx < 0 or variant_idx >= len(file_paths):
            return None
        return file_paths[variant_idx]

    if variant not in (None, "0", 0):
        return None
    return item.get("file_path")


def _image_variant_index(variant: str | None) -> int | None:
    try:
        return 0 if variant is None else int(variant)
    except (TypeError, ValueError):
        return None


def _select_image_variant_cloud_url(item: dict, variant: str | None) -> str | None:
    variant_idx = _image_variant_index(variant)
    if variant_idx is None:
        return None

    urls = item.get("urls")
    if urls and 0 <= variant_idx < len(urls):
        return urls[variant_idx]
    if variant_idx == 0:
        return item.get("url")
    return None


def _raise_if_image_variant_not_found(item: dict, variant: str | None) -> None:
    file_paths = item.get("file_paths")
    if not file_paths:
        return

    variant_idx = _image_variant_index(variant)
    if variant_idx is None or variant_idx < 0 or variant_idx >= len(file_paths):
        raise HTTPException(
            status_code=404,
            detail=f"Image variant {variant} not found",
        )


def _build_image_response_kwargs(
    save_file_path_list: list[str],
    resp_format: str,
    prompt: str,
    request_id: str,
    result: OutputBatch,
    *,
    b64_list: list[str] | None = None,
    cloud_url: str | None = None,
    cloud_urls: list[str | None] | None = None,
    fallback_url: str | None = None,
    fallback_urls: list[str] | None = None,
    is_persistent: bool = True,
    resize: str | None = None,
) -> dict:
    """Build ImageResponse data list.

    For b64_json: uses pre-read b64_list (call _read_b64_for_paths first).
    For url: uses cloud_url or fallback_url.
    file_path is omitted when is_persistent=False to avoid exposing stale temp paths.
    """
    ret = None
    if resp_format == "b64_json":
        if not b64_list:
            raise ValueError("b64_list required for b64_json response_format")
        data = [
            ImageResponseData(
                b64_json=b64,
                revised_prompt=prompt,
                file_path=os.path.abspath(path) if is_persistent else None,
                resize=resize,
            )
            for b64, path in zip(b64_list, save_file_path_list)
        ]
        ret = {"data": data}
    elif resp_format == "url":
        if cloud_urls is None and cloud_url is not None:
            cloud_urls = [cloud_url]
        if fallback_urls is None and fallback_url is not None:
            fallback_urls = [fallback_url]

        data = []
        for idx, path in enumerate(save_file_path_list):
            url = None
            if cloud_urls is not None and idx < len(cloud_urls):
                url = cloud_urls[idx]
            if not url and fallback_urls is not None and idx < len(fallback_urls):
                url = fallback_urls[idx]
            if not url:
                break
            data.append(
                ImageResponseData(
                    url=url,
                    revised_prompt=prompt,
                    file_path=os.path.abspath(path) if is_persistent else None,
                    resize=resize,
                )
            )

        if len(data) != len(save_file_path_list):
            raise HTTPException(
                status_code=400,
                detail="response_format='url' requires cloud storage to be configured.",
            )
        ret = {"data": data}
    else:
        raise HTTPException(
            status_code=400, detail=f"response_format={resp_format} is not supported"
        )

    ret = add_common_data_to_response(ret, request_id=request_id, result=result)
    if ret.get("usage") is not None:
        ret["usage"]["image_count"] = len(save_file_path_list)

    return ret


def _get_response_resize
```

## sglang/runtime/entrypoints/openai/mesh_api.py
```python
import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    MeshGenerationsRequest,
    MeshListResponse,
    MeshResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import MESH_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    add_common_data_to_response,
    merge_image_input_list,
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/meshes", tags=["meshes"])


def _normalize_format(fmt: Optional[str]) -> str:
    fmt = (fmt or "glb").lower()
    return fmt if fmt in ("glb", "obj") else "glb"


def _build_sampling_params_from_request(
    request_id: str, req: MeshGenerationsRequest, image_path: Optional[str] = None
) -> SamplingParams:
    ext = _normalize_format(req.output_format)

    server_args = get_global_server_args()
    sampling_kwargs: Dict[str, Any] = {
        "request_id": request_id,
        "task_type": ModelTaskType.I2M,
        "prompt": req.prompt,
        "num_frames": 1,
        "image_path": [image_path] if image_path else None,
        "save_output": True,
        "output_file_name": f"{request_id}.{ext}",
        "seed": req.seed,
        "generator_device": req.generator_device,
    }
    if req.num_inference_steps is not None:
        sampling_kwargs["num_inference_steps"] = req.num_inference_steps
    if req.guidance_scale is not None:
        sampling_kwargs["guidance_scale"] = req.guidance_scale
    if req.negative_prompt is not None:
        sampling_kwargs["negative_prompt"] = req.negative_prompt

    return SamplingParams.from_user_sampling_params_args(
        model_path=server_args.model_path,
        server_args=server_args,
        **sampling_kwargs,
    )


def _mesh_job_from_sampling(
    request_id: str, req: MeshGenerationsRequest, sampling: SamplingParams
) -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "mesh",
        "model": req.model or "",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "format": _normalize_format(req.output_format),
        "file_path": os.path.abspath(sampling.output_file_path()),
    }


async def _dispatch_job_async(job_id: str, batch: Req) -> None:
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    try:
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        save_file_path = save_file_path_list[0]

        file_size = None
        if os.path.exists(save_file_path):
            file_size = os.path.getsize(save_file_path)

        cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)

        update_fields: Dict[str, Any] = {
            "status": "completed",
            "progress": 100,
            "completed_at": int(time.time()),
            "url": cloud_url,
            "file_path": save_file_path if not cloud_url else None,
            "file_size_bytes": file_size,
        }
        update_fields = add_common_data_to_response(
            update_fields, request_id=job_id, result=result
        )
        await MESH_STORE.update_fields(job_id, update_fields)
    except Exception as e:
        logger.exception("Mesh job %s failed", job_id)
        await MESH_STORE.update_fields(
            job_id, {"status": "failed", "error": {"message": str(e)}}
        )


@router.post("", response_model=MeshResponse)
async def create_mesh(
    request: Request,
    image: Optional[List[UploadFile]] = File(None),
    image_array: Optional[List[UploadFile]] = File(None, alias="image[]"),
    url: Optional[List[str]] = Form(None),
    url_array: Optional[List[str]] = Form(None, alias="url[]"),
    prompt: Optional[str] = Form("generate 3d mesh"),
    model: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    generator_device: Optional[str] = Form("cuda"),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    output_format: Optional[str] = Form("glb"),
):
    content_type = request.headers.get("content-type", "").lower()
    request_id = generate_request_id()
    server_args = get_global_server_args()

    input_path = None

    if "multipart/form-data" in content_type:
        images = image or image_array
        urls = url or url_array
        image_list = merge_image_input_list(images, urls)

        if not image_list:
            raise HTTPException(
                status_code=422,
                detail="Field 'image' or 'url' is required for mesh generation",
            )

        uploads_dir = os.path.join("outputs", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        img = image_list[0]
        filename = img.filename if hasattr(img, "filename") else "input_image"
        try:
            input_path = await save_image_to_path(
                img, os.path.join(uploads_dir, f"{request_id}_{filename}")
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to process image source: {str(e)}"
            )

        req = MeshGenerationsRequest(
            prompt=prompt or "generate 3d mesh",
            model=model,
            seed=seed,
            generator_device=generator_device,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            output_format=output_format,
            **(
                {"guidance_scale": guidance_scale} if guidance_scale is not None else {}
            ),
        )
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            payload: Dict[str, Any] = dict(body or {})

            if payload.get("input_image"):
                img_src = payload.pop("input_image")
                uploads_dir = os.path.join("outputs", "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                input_path = await save_image_to_path(
                    img_src,
                    os.path.join(uploads_dir, f"{request_id}_input_image"),
                )

            req = MeshGenerationsRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    if not input_path:
        raise HTTPException(
            status_code=422,
            detail="An input image is required for mesh generation",
        )

    sampling_params = _build_sampling_params_from_request(request_id, req, input_path)
    job = _mesh_job_from_sampling(request_id, req, sampling_params)
    await MESH_STORE.upsert(request_id, job)

    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling_params,
    )

    asyncio.create_task(_dispatch_job_async(request_id, batch))
    return MeshResponse(**job)


@router.get("", response_model=MeshListResponse)
async def list_meshes(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
 
```

## sglang/runtime/entrypoints/openai/prompt_enhancement.py
```python
import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import HTTPException, Request
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class PromptEnhancerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: HttpUrl
    model: str = Field(min_length=1)
    system_prompt: str = Field(
        default=(
            "You expand prompts for image and video generation. The user supplies "
            "JSON with task and prompt fields. Expand the prompt into a concise, "
            "visually detailed description, clarifying composition, lighting and "
            "appearance where appropriate instead of simply repeating it. Preserve "
            "the requested subjects, count, quoted text, actions, style and language. "
            "Preserve special tokens and reference labels exactly. "
            "Clarify visual details without inventing new subjects or changing intent. "
            "For image editing, describe only the requested edits and preserve "
            "everything else. For video, preserve the requested motion and timing. "
            "Do not invent facts about reference media you cannot see. "
            "Return only the rewritten prompt, without surrounding quotes, "
            "commentary or markdown."
        ),
        min_length=1,
    )
    include_images: bool = False
    timeout: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    api_key_env: str | None = None
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: HttpUrl) -> HttpUrl:
        if value.username or value.password or value.query or value.fragment:
            raise ValueError(
                "base_url must not contain credentials, a query or fragment"
            )
        return value

    @field_validator("generation_kwargs")
    @classmethod
    def validate_generation_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        # one non-streaming completion per diffusion request
        reserved = value.keys() & {
            "model",
            "messages",
            "stream",
            "n",
            "tools",
            "tool_choice",
        }
        if reserved:
            raise ValueError(f"generation_kwargs cannot override {sorted(reserved)}")
        return value


class PromptEnhancer:
    """HTTP frontend client; owns its connection pool, never runs in GPU workers."""

    def __init__(self, config: PromptEnhancerConfig):
        self.config = config
        headers = {}
        if config.api_key_env is not None:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Prompt enhancer API key env {config.api_key_env} is unset"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(
            base_url=str(config.base_url).rstrip("/") + "/",
            headers=headers,
            timeout=config.timeout,
            trust_env=False,
        )

    @classmethod
    def from_file(cls, path: str) -> "PromptEnhancer":
        return cls(PromptEnhancerConfig.model_validate_json(Path(path).read_text()))

    async def close(self) -> None:
        await self.client.aclose()

    async def enhance(
        self,
        prompt: str,
        *,
        task: Literal["image", "image_edit", "video"],
        image_paths: list[str],
    ) -> str:
        content: str | list[dict[str, Any]] = json.dumps(
            {"task": task, "prompt": prompt}, ensure_ascii=False
        )
        if self.config.include_images and image_paths:
            content = [{"type": "text", "text": content}]
            for path in image_paths:
                if path.startswith(("https://", "http://", "data:image/")):
                    url = path
                else:
                    url = await asyncio.to_thread(_local_image_url, path)
                content.append({"type": "image_url", "image_url": {"url": url}})

        payload = {
            "max_tokens": 512,
            "temperature": 0,
            **self.config.generation_kwargs,
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": content},
            ],
            "stream": False,
            "n": 1,
        }
        try:
            response = await self.client.post("chat/completions", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise HTTPException(504, "Prompt enhancer timed out") from e
        except httpx.HTTPError as e:
            # do not expose upstream response bodies, prompts or auth headers
            raise HTTPException(502, "Prompt enhancer request failed") from e

        try:
            choice = response.json()["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("Incomplete completion")
            rewritten = choice["message"]["content"]
            if not isinstance(rewritten, str) or not rewritten.strip():
                raise ValueError("Empty completion")
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise HTTPException(
                502,
                "Prompt enhancer did not return a complete, nonempty text response; "
                "check its template and token limit",
            ) from e
        return rewritten.strip()


def _local_image_url(path: str) -> str:
    # uploads live on the diffusion host, not necessarily on the SRT host
    try:
        with Image.open(path) as image:
            mime = image.get_format_mimetype()
            image.verify()
        if mime is None:
            raise ValueError("Unknown image MIME type")
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except (OSError, ValueError, SyntaxError) as e:
        raise HTTPException(
            400, "Cannot read reference image for prompt enhancement"
        ) from e
    return f"data:{mime};base64,{encoded}"


async def maybe_enhance_prompt(
    request: Request,
    prompt: str,
    *,
    enabled: bool,
    task: Literal["image", "image_edit", "video"],
    image_paths: list[str] | None = None,
) -> str:
    if not enabled:
        return prompt
    enhancer: PromptEnhancer | None = request.app.state.prompt_enhancer
    if enhancer is None:
        raise HTTPException(400, "Prompt enhancement requires --prompt-enhancer-config")
    return await enhancer.enhance(prompt, task=task, image_paths=image_paths or [])

```

## sglang/runtime/entrypoints/openai/protocol.py
```python
import time
import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# Image API protocol models
class ImageResponseData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None
    file_path: Optional[str] = None
    resize: Optional[str] = None


class ImagePromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class ImageUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    prompt_tokens_details: Optional[ImagePromptTokensDetails] = None
    reasoning_tokens: Optional[int] = 0
    image_count: Optional[int] = None


class ImageResponse(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageResponseData]
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None
    usage: Optional[ImageUsage] = None


# Keep request schemas limited to OpenAI fields and stable cross-model SGLang
# extensions. Model-owned controls travel as allowed extras and are interpreted
# only after the active SamplingParams subclass is resolved; do not add them to
# these shared protocol models.
class ImageGenerationsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    task_type: Optional[str] = Field(
        default=None, description="Requested task from the server supported_task_types."
    )
    enhance_prompt: bool = False
    model: Optional[str] = None
    n: Optional[int] = 1
    quality: Optional[str] = "auto"
    response_format: Optional[str] = "url"  # url | b64_json
    size: Optional[str] = "1024x1024"  # e.g., 1024x1024
    style: Optional[str] = "vivid"
    background: Optional[str] = "auto"  # transparent | opaque | auto
    output_format: Optional[str] = None  # png | jpeg | webp
    user: Optional[str] = None
    # SGLang extensions
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    true_cfg_scale: Optional[float] = (
        None  # for CFG vs guidance distillation (e.g., QwenImage)
    )
    seed: Optional[Union[int, List[int]]] = None
    generator_device: Optional[str] = "cuda"
    negative_prompt: Optional[str] = None
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    enable_teacache: Optional[bool] = False
    max_sequence_length: Optional[int] = None
    flow_shift: Optional[float] = None
    # Upscaling
    enable_upscaling: Optional[bool] = False
    upscaling_model_path: Optional[str] = None
    upscaling_scale: Optional[int] = 4
    diffusers_kwargs: Optional[Dict[str, Any]] = None  # kwargs for diffusers backend
    # Performance profiling
    perf_dump_path: Optional[str] = None
    # Progressive resolution generation
    progressive_mode: Optional[str] = None
    progressive_levels: Optional[int] = None
    progressive_delta: Optional[float] = None


# Video API protocol models
class VideoResponse(BaseModel):
    id: str
    object: str = "video"
    model: str = "sora-2"
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    size: str = ""
    seconds: str = "4"
    quality: str = "standard"
    url: Optional[str] = None
    remixed_from_video_id: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    num_outputs: Optional[int] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None
    revised_prompt: Optional[str] = None
    action: Optional[Dict[str, Any]] = None


class VideoGenerationsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    enhance_prompt: bool = False
    input_reference: Optional[str] = None
    reference_url: Optional[str] = None
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    task_type: Optional[str] = Field(
        default=None, description="Requested task from the server supported_task_types."
    )
    model: Optional[str] = None
    n: Optional[int] = 1
    num_outputs_per_prompt: Optional[int] = None
    seconds: Optional[int] = 4
    size: Optional[str] = ""
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[Union[int, List[int]]] = None
    generator_device: Optional[str] = "cuda"
    # SGLang extensions
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    true_cfg_scale: Optional[float] = (
        None  # for CFG vs guidance distillation (e.g., QwenImage)
    )
    negative_prompt: Optional[str] = None
    max_sequence_length: Optional[int] = None
    flow_shift: Optional[float] = None
    enable_teacache: Optional[bool] = False
    # Frame interpolation
    enable_frame_interpolation: Optional[bool] = False
    frame_interpolation_exp: Optional[int] = 1  # 1=2×, 2=4×
    frame_interpolation_scale: Optional[float] = 1.0
    frame_interpolation_model_path: Optional[str] = None
    # Upscaling
    enable_upscaling: Optional[bool] = False
    upscaling_model_path: Optional[str] = None
    upscaling_scale: Optional[int] = 4
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    x264_preset: Optional[str] = None
    output_path: Optional[str] = None
    diffusers_kwargs: Optional[Dict[str, Any]] = None  # kwargs for diffusers backend
    # Performance profiling
    perf_dump_path: Optional[str] = None
    profile: Optional[bool] = False
    num_profiled_timesteps: Optional[int] = None
    profile_all_stages: Optional[bool] = False


class VideoListResponse(BaseModel):
    data: List[VideoResponse]
    object: str = "list"


class VideoRemixRequest(BaseModel):
    prompt: str


class RealtimeVideoGenerationsRequest(VideoGenerationsRequest):
    enhance_prompt: Literal[False] = False
    type: Literal["init"]
    # WebSocket does not support multipart/form-data image uploads
    first_frame: Optional[bytes | str] = None
    condition_inputs: Optional[Dict[str, Any]] = None
    max_chunks: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = 42
    guidance_scale: Optional[float] = 1.0
    size: Optional[str] = "832x480"
    profile: Optional[bool] = False
    num_profiled_timesteps: Optional[int] = None
    profile_all_stages: Optional[bool] = False
    realtime_output_format: Optional[Literal["raw", "webp", "jpeg"]] = None
    realtime_preview_max_width: Optional[int] = None
    realtime_output_pacing: Optional[bool] = False
    realtime_causal_sink_size: Optional[int] = None
    realtime_causal_kv_cache_num_frames: Optional[int] = None


class RealtimeEvent(BaseModel):
    type: Literal["event"]
    kind: str
    payload: Any = None
    event_id: Optional[int] = None


# Mesh API protocol models
class MeshResponse(BaseModel):
    id: str
    object: str = "mesh"
    model: str = ""
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    format: str = "glb"
    url: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class MeshGenerationsRequest(BaseModel):
    prompt: str = "generate 3d mesh"
    input_image: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[Union[
```

## sglang/runtime/entrypoints/openai/realtime/__init__.py
```python
# SPDX-License-Identifier: Apache-2.0

```

## sglang/runtime/entrypoints/openai/realtime/adapters/lingbot_world_realtime_adapter.py
```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
    save_realtime_first_frame,
)
from sglang.multimodal_gen.runtime.realtime.control_signals import (
    ControlSignalQueue,
    ParsedControlEventPayload,
    parse_control_event_payload,
)
from sglang.multimodal_gen.runtime.realtime.lingbot_world import (
    LINGBOT_CAMERA_ACTIONS_CONDITION,
    LINGBOT_PROMPT_UPDATED_CONDITION,
)
from sglang.multimodal_gen.runtime.realtime.states import (
    RealtimeCameraControlState,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


LINGBOT_REALTIME_DEFAULT_NUM_INFERENCE_STEPS = 4
LINGBOT_REALTIME_MIN_CONDITION_CHUNKS = 2
COMPOSITE_INPUT_EVENT_KIND = "composite_input"


class LingBotWorldRealtimeState(RealtimeCameraControlState):
    def __init__(self):
        super().__init__(
            min_pulse_items=1,
            script_maxlen=512,
            max_transitions=512,
        )
        self.prompt_queue = ControlSignalQueue(max_events={"prompt": 1})

    def clear(self) -> None:
        super().clear()
        self.prompt_queue.clear()

    def receive_prompt(self, prompt: str, *, event_id: int | None = None) -> None:
        self.prompt_queue.push("prompt", prompt, event_id=event_id)

    def parse_camera_control_event_payload(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> ParsedControlEventPayload:
        return parse_control_event_payload(
            payload,
            event_id=event_id,
            kind="camera_actions",
            normalize_state_payload=self._normalize_state_actions,
            validate_script_payload=LingBotWorldRealtimeAdapter._validate_camera_actions,
        )

    def receive_parsed_camera_control_event_payload(
        self,
        parsed: ParsedControlEventPayload,
        *,
        event_id: int | None,
    ) -> str:
        if parsed.mode == "state":
            transitions = parsed.payload
            self.receive_camera_state_transitions(transitions)
            return f"kind=camera_actions, mode=state, transitions={len(transitions)}"

        camera_actions = parsed.payload
        self.receive_camera_action_script(camera_actions, event_id=event_id)
        return f"kind=camera_actions, mode=script, frames={len(camera_actions)}"

    def receive_camera_control_event_payload(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> str:
        parsed = self.parse_camera_control_event_payload(payload, event_id=event_id)
        return self.receive_parsed_camera_control_event_payload(
            parsed, event_id=event_id
        )

    def sample_prompt(self) -> str:
        prompt = self.prompt_queue.pop_latest("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt event payload must be a string")
        self.latest_sampled_event_id = self.prompt_queue.last_sampled_seq_id("prompt")
        return prompt

    def has_prompt(self) -> bool:
        return self.prompt_queue.has_events("prompt")


class LingBotWorldRealtimeAdapter(BaseRealtimeModelAdapter):
    def create_state(self) -> LingBotWorldRealtimeState:
        return LingBotWorldRealtimeState()

    def _state(self, session: GenerateSession) -> LingBotWorldRealtimeState:
        state = session.adapter_state
        if not isinstance(state, LingBotWorldRealtimeState):
            raise TypeError("LingBot realtime adapter state is not initialized")
        return state

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        condition_inputs = request.condition_inputs or {}
        camera_actions = condition_inputs.get(LINGBOT_CAMERA_ACTIONS_CONDITION)
        if camera_actions is not None:
            state = self._state(session)
            state.receive_camera_action_script(
                self._validate_camera_actions(camera_actions)
            )

        await save_realtime_first_frame(session, request)

    @staticmethod
    def _validate_camera_actions(payload: Any) -> list[list[str]]:
        if not isinstance(payload, list):
            raise ValueError("camera_actions event payload must be list[list[str]]")
        normalized = []
        for frame_actions in payload:
            if not isinstance(frame_actions, list):
                raise ValueError("camera_actions event payload must be list[list[str]]")
            normalized.append(list(frame_actions))
        return normalized

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str:
        state = self._state(session)
        if event.kind == "camera_actions":
            return self._ingest_camera_actions(state, event.payload, event.event_id)
        elif event.kind == "prompt":
            return self._ingest_prompt(state, event.payload, event.event_id)
        elif event.kind == COMPOSITE_INPUT_EVENT_KIND:
            return self._ingest_composite_input(state, event.payload, event.event_id)
        raise ValueError(f"unsupported event kind: {event.kind}")

    def _ingest_camera_actions(
        self,
        state: LingBotWorldRealtimeState,
        payload: Any,
        event_id: int | None,
    ) -> str:
        return state.receive_camera_control_event_payload(
            payload,
            event_id=event_id,
        )

    def _ingest_prompt(
        self,
        state: LingBotWorldRealtimeState,
        payload: Any,
        event_id: int | None,
    ) -> str:
        prompt = self._validate_prompt_payload(payload)
        state.receive_prompt(prompt, event_id=event_id)
        return f"kind=prompt, prompt_len={len(prompt)}"

    @staticmethod
    def _validate_prompt_payload(payload: Any) -> str:
        if not isinstance(payload, str) or not payload:
            raise ValueError("prompt event payload must be a non-empty string")
        return payload

    def _ingest_composite_input(
        self,
        state: LingBotWorldRealtimeState,
        payload: Any,
        event_id: int | None,
    ) -> str:
        if not isinstance(payload, dict):
            raise ValueError("composite_input event payload must be a map")
        input_types = payload.get("input_types")
        if not isinstance(input_types, list) or not input_types:
            raise ValueError(
                "composite_input event payload requires non-empty input_types"
            )

        parsed_inputs = []
        for input_type in input_types:
            if not isinstance(input_type, str) or not input_type:
                raise ValueError(
                    "composite_input input_types must contain non-empty strings"
                )
            if input_type not in payload:
                raise ValueError(f"composite_input event payload requires {input_type}")
            parsed_inputs.append(
                (
                    input_type,
                    self._parse_composite_input_item(
                        state,
                        input_type,
                        payload[input_type],
                        event_id,
                    ),
                )
            )

        input_logs = []
        for input_type, parsed_payload in parsed_inputs:
            input_logs.append(
                self._ingest_parsed_composite_input_item(
                    state,
                    input_type,
                    parsed_payload,
                    event_id,
        
```

## sglang/runtime/entrypoints/openai/realtime/adapters/sana_wm_realtime_adapter.py
```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
    save_realtime_first_frame,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.base import (
    normalize_sana_wm_camera_actions,
    parse_sana_wm_action_string,
    snap_sana_wm_num_frames,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.self_forcing import (
    SanaWMSelfForcingSampler,
)
from sglang.multimodal_gen.runtime.realtime.states import (
    RealtimeCameraControlState,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


SANA_WM_DEFAULT_SIZE = "1280x704"
SANA_WM_DEFAULT_NUM_FRAMES = 1081
SANA_WM_DEFAULT_FPS = 16
SANA_WM_DEFAULT_STEPS = 4
SANA_WM_DEFAULT_GUIDANCE = 1.0
SANA_WM_CONTROL_PULSE_FRAMES = 8


def _normalize_sana_wm_state_actions(actions: list[Any]) -> list[str]:
    return [str(action).lower() for action in actions]


class SanaWMRealtimeAdapterState(RealtimeCameraControlState):
    def __init__(self):
        super().__init__(
            min_pulse_items=SANA_WM_CONTROL_PULSE_FRAMES,
            script_maxlen=2048,
            max_transitions=512,
            normalize_state_actions=_normalize_sana_wm_state_actions,
        )
        self.base_condition_inputs: dict[str, Any] = {}

    def clear(self) -> None:
        super().clear()
        self.base_condition_inputs.clear()

    def receive_camera_control_event_payload(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> str:
        return super().receive_camera_control_event_payload(
            payload,
            event_id=event_id,
            validate_camera_actions=SanaWMRealtimeAdapter._validate_camera_actions,
        )


class SanaWMRealtimeAdapter(BaseRealtimeModelAdapter):
    def create_state(self) -> SanaWMRealtimeAdapterState:
        return SanaWMRealtimeAdapterState()

    def _state(self, session: GenerateSession) -> SanaWMRealtimeAdapterState:
        state = session.adapter_state
        if not isinstance(state, SanaWMRealtimeAdapterState):
            raise TypeError("SANA-WM realtime adapter state is not initialized")
        return state

    @staticmethod
    def _validate_camera_actions(payload: Any) -> list[list[str]]:
        return normalize_sana_wm_camera_actions(
            payload, error_label="camera_actions event payload"
        )

    @staticmethod
    def _raw_frame_count(result: OutputBatch) -> int | None:
        if result.raw_frame_batches is None:
            return None
        return sum(len(frames) for frames in result.raw_frame_batches)

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        request.size = request.size or SANA_WM_DEFAULT_SIZE
        if request.num_frames is not None:
            request.num_frames = int(request.num_frames)
        else:
            # Open-ended session: keep num_frames unset so prepare_next_request
            # samples uniform action chunks (no front-loaded segmentation), and
            # flag the stage explicitly via condition_inputs —
            # build_sampling_params strips None fields, so the per-chunk batch
            # would otherwise carry the SamplingParams default num_frames.
            request.condition_inputs = {
                **(request.condition_inputs or {}),
                "sana_wm_open_ended": True,
            }
        request.fps = int(request.fps or SANA_WM_DEFAULT_FPS)
        request.num_inference_steps = int(
            request.num_inference_steps or SANA_WM_DEFAULT_STEPS
        )
        request.guidance_scale = float(
            request.guidance_scale or SANA_WM_DEFAULT_GUIDANCE
        )
        if request.negative_prompt is None:
            request.negative_prompt = ""
        if request.generator_device is None:
            request.generator_device = "cuda"

        state = self._state(session)
        condition_inputs = dict(request.condition_inputs or {})
        camera_actions = condition_inputs.pop("camera_actions", None)
        action = condition_inputs.pop("action", None)
        if camera_actions is not None and action is not None:
            raise ValueError("pass only one of camera_actions or action")
        if camera_actions is not None:
            state.receive_camera_control_event_payload(camera_actions, event_id=None)
        if action is not None:
            if not isinstance(action, str) or not action:
                raise ValueError("action condition input must be a non-empty string")
            state.receive_camera_action_script(
                parse_sana_wm_action_string(action), event_id=None
            )
        state.base_condition_inputs = condition_inputs

        await save_realtime_first_frame(
            session,
            request,
            required_error="SANA-WM realtime requires first_frame",
            cache_remote_urls=True,
        )

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str:
        state = self._state(session)
        if event.kind == "camera_actions":
            return state.receive_camera_control_event_payload(
                event.payload,
                event_id=event.event_id,
            )
        if event.kind == "action":
            if not isinstance(event.payload, str) or not event.payload:
                raise ValueError("action event payload must be a non-empty string")
            camera_actions = parse_sana_wm_action_string(event.payload)
            state.receive_camera_action_script(camera_actions, event_id=event.event_id)
            return f"kind=action, frames={len(camera_actions)}"
        raise ValueError(f"unsupported event kind: {event.kind}")

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        action_chunk_size = self._action_chunk_size(
            session,
            server_args,
            chunk,
            chunk_size,
        )
        state = self._state(session)
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        condition_inputs = dict(state.base_condition_inputs) if chunk.index == 0 else {}
        camera_actions = state.sample_camera_actions(action_chunk_size)
        if camera_actions is not None:
            condition_inputs["camera_actions"] = camera_actions
        return RealtimeChunkInputs(
            prompt=request.prompt,
            condition_inputs=condition_inputs,
        )

    def build_sampling_params(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
    ):
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        return build_realtime_sampling_params(
            chunk.request_id,
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            chunk_size=chunk_size,
        )

    def _action_ch
```

## sglang/runtime/entrypoints/openai/realtime/generate_session.py
```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    RealtimeSession,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
        BaseRealtimeModelAdapter,
    )


@dataclass(frozen=True, slots=True)
class RealtimeChunkContext:
    session_id: str
    index: int
    request_id: str


class GenerateSession:
    """A realtime generation session"""

    def __init__(self):
        self.id = uuid4().hex
        self.request: RealtimeVideoGenerationsRequest | None = None
        self.input_temp_dir: str | None = None
        self.generate_chunk_cnt = 0
        self.current_chunk: RealtimeChunkContext | None = None
        self.realtime_session = RealtimeSession()
        self.adapter: BaseRealtimeModelAdapter | None = None
        self.adapter_state: Any = None
        self.output_pace_next_send_at: float | None = None
        self.output_pace_last_event_id: int | None = None

    def set_adapter(self, adapter: BaseRealtimeModelAdapter):
        self.adapter = adapter
        self.adapter_state = adapter.create_state()

    def set_request(self, request: RealtimeVideoGenerationsRequest):
        self.request = request

    def dispose(self):
        if self.adapter is not None:
            self.adapter.dispose(self)
        self.request = None
        self.input_temp_dir = None
        self.generate_chunk_cnt = 0
        self.current_chunk = None
        self.adapter = None
        self.adapter_state = None
        self.output_pace_next_send_at = None
        self.output_pace_last_event_id = None
        self.realtime_session.dispose()

    def new_chunk(self) -> RealtimeChunkContext:
        if self.current_chunk is not None:
            raise RuntimeError("previous realtime chunk is still active")
        chunk = RealtimeChunkContext(
            session_id=self.id,
            index=self.generate_chunk_cnt,
            request_id=f"{self.id}_{uuid4().hex}",
        )
        self.current_chunk = chunk
        return chunk

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1
        self.current_chunk = None

    def reached_max_chunks(self) -> bool:
        return (
            self.request is not None
            and self.request.max_chunks is not None
            and self.generate_chunk_cnt >= self.request.max_chunks
        )

```

## sglang/runtime/entrypoints/openai/realtime/realtime_adapter.py
```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
    RealtimeFrameSendStats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    prepare_request,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


@dataclass(slots=True)
class RealtimeChunkInputs:
    """Sampled from realtime control state, consumed by the Req"""

    prompt: str
    condition_inputs: dict[str, Any] = field(default_factory=dict)


async def save_realtime_first_frame(
    session: GenerateSession,
    request: RealtimeVideoGenerationsRequest,
    *,
    required_error: str | None = None,
    cache_remote_urls: bool = False,
) -> None:
    first_frame = request.first_frame
    if first_frame is None:
        if required_error is not None:
            raise ValueError(required_error)
        return

    server_args = get_global_server_args()
    if server_args.input_save_path is not None:
        uploads_dir = server_args.input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        if session.input_temp_dir is None:
            session.input_temp_dir = tempfile.mkdtemp(prefix="sglang_input_")
        uploads_dir = session.input_temp_dir

    if (
        cache_remote_urls
        and isinstance(first_frame, str)
        and first_frame.lower().startswith(("http://", "https://"))
    ):
        suffix = os.path.splitext(first_frame.split("?", 1)[0])[1]
        digest = hashlib.sha256(first_frame.encode("utf-8")).hexdigest()[:16]
        target_path = os.path.join(uploads_dir, f"realtime_ref_{digest}{suffix}")
        if os.path.exists(target_path):
            request.first_frame = target_path
            return
    else:
        target_path = os.path.join(uploads_dir, f"{session.id}_first_frame")

    request.first_frame = await save_image_to_path(first_frame, target_path)


def build_realtime_sampling_params(
    request_id: str,
    *,
    request: RealtimeVideoGenerationsRequest,
    chunk_inputs: RealtimeChunkInputs,
    num_frames: int | None,
    num_inference_steps: int | None,
    chunk_size: int,
):
    return build_sampling_params(
        request_id,
        prompt=chunk_inputs.prompt,
        size=request.size,
        num_frames=num_frames,
        fps=request.fps,
        image_path=request.first_frame,
        output_file_name=request_id,
        save_output=False,
        seed=request.seed,
        generator_device=request.generator_device,
        num_inference_steps=num_inference_steps,
        guidance_scale=request.guidance_scale,
        guidance_scale_2=request.guidance_scale_2,
        negative_prompt=request.negative_prompt,
        enable_teacache=request.enable_teacache,
        enable_frame_interpolation=request.enable_frame_interpolation,
        frame_interpolation_exp=request.frame_interpolation_exp,
        frame_interpolation_scale=request.frame_interpolation_scale,
        frame_interpolation_model_path=request.frame_interpolation_model_path,
        enable_upscaling=request.enable_upscaling,
        upscaling_model_path=request.upscaling_model_path,
        upscaling_scale=request.upscaling_scale,
        diffusers_kwargs=request.diffusers_kwargs,
        profile=request.profile,
        num_profiled_timesteps=request.num_profiled_timesteps,
        profile_all_stages=request.profile_all_stages,
        perf_dump_path=request.perf_dump_path,
        output_path=request.output_path,
        output_compression=request.output_compression,
        output_quality=request.output_quality,
        quality=getattr(request, "quality", None),
        condition_inputs=chunk_inputs.condition_inputs,
        realtime_chunk_size=chunk_size,
    )


class BaseRealtimeModelAdapter:
    def __init__(self):
        self.output_adapter = RawRGBRealtimeOutputAdapter()

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        raise NotImplementedError

    def create_state(self) -> Any:
        """create a state for managing runtime states"""
        raise NotImplementedError

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str:
        """
        Ingest a realtime endpoint event and install it into the model's realtime control queues
        """
        raise NotImplementedError

    async def wait_for_next_chunk(self, session: GenerateSession) -> None:
        del session

    def get_chunk_size(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> int:
        del session, chunk
        arch_config = server_args.pipeline_config.dit_config.arch_config
        return int(getattr(arch_config, "num_frames_per_block", 3))

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        raise NotImplementedError

    def build_sampling_params(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
    ):
        raise NotImplementedError

    def get_realtime_event_id(self, session: GenerateSession) -> int | None:
        del session
        return None

    def prepare_next_request(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> Req:
        chunk_size = self.get_chunk_size(session, server_args, chunk)
        chunk_inputs = self.sample_chunk_inputs(
            session,
            server_args,
            chunk,
            chunk_size,
        )
        sampling_params = self.build_sampling_params(
            session,
            server_args,
            chunk,
            chunk_inputs,
            chunk_size,
        )
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling_params,
        )
        self.apply_realtime_request_fields(
            batch,
            session,
            chunk,
            event_id=self.get_realtime_event_id(session),
        )
        return batch

    def apply_realtime_request_fields(
        self,
        batch: Req,
        session: GenerateSession,
        chunk: RealtimeChunkContext,
        *,
        event_id: int | None,
    ) -> None:
        batch.realtime_session_id = session.id
        batch.return_raw_frames = True
        batch.block_idx = chunk.index
        batch.realtime_event_id = event_id
        if session.request is None:
            return
        batch.realtime_output_format = session.request.realtime_output_format
        batch.realtime_preview_max_width = session.request.realtime_preview_max_width
        batch.realtime_output_pacing = bool(session.request.realtime_output_pacing)
        batch.realtime_causal_sink_size = session.request.realtime_causal_sink_size
        batch.realtime_causal_kv_cache_num_frames = (
            s
```

## sglang/runtime/entrypoints/openai/realtime/realtime_output_adapter.py
```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import msgspec.msgpack
from fastapi import WebSocket
from PIL import Image

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.timer import (
    RealtimeStageTimer,
)
from sglang.multimodal_gen.runtime.realtime.video import (
    JPEG_FRAME_CONTENT_TYPE,
    RAW_RGB_CHANNELS,
    RAW_RGB_CONTENT_TYPE,
    WEBP_FRAME_CONTENT_TYPE,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )


class RealtimeFrameBatchHeader(TypedDict, total=False):
    type: str
    request_id: str
    chunk_index: int
    content_type: str
    num_frames: int
    total_size: int
    format: str
    width: int
    height: int
    channels: int
    bytes_per_frame: int
    raw_size: int
    encoding: str
    delta_reference: str
    payload_lengths: list[int]
    event_id: int
    frame_batch_index: int
    num_frame_batches: int
    is_final_frame_batch: bool


class RealtimeFrameBatchMessage(RealtimeFrameBatchHeader, total=False):
    payload: bytes


class RealtimeFrameSendStats(TypedDict):
    header_pack_ms: float
    header_write_ms: float
    raw_payload_build_ms: float
    raw_write_ms: float
    ws_write_ms: float
    pace_wait_ms: float
    raw_bytes: int
    ws_payload_bytes: int
    num_frames: int
    num_batches: int
    frame_shape: tuple[int, int, int] | None
    content_type: str


def empty_frame_send_stats(content_type: str = "") -> RealtimeFrameSendStats:
    return {
        "header_pack_ms": 0.0,
        "header_write_ms": 0.0,
        "raw_payload_build_ms": 0.0,
        "raw_write_ms": 0.0,
        "ws_write_ms": 0.0,
        "pace_wait_ms": 0.0,
        "raw_bytes": 0,
        "ws_payload_bytes": 0,
        "num_frames": 0,
        "num_batches": 0,
        "frame_shape": None,
        "content_type": content_type,
    }


def _raw_rgb_frame_metadata(batch: Req) -> dict[str, int | str]:
    frame_width = batch.width
    frame_height = batch.height
    if frame_width is None or frame_height is None:
        return {}

    frame_width = int(frame_width)
    frame_height = int(frame_height)
    if batch.enable_upscaling:
        upscaling_scale = int(batch.upscaling_scale or 1)
        frame_width *= upscaling_scale
        frame_height *= upscaling_scale

    return {
        "format": "rgb24",
        "width": frame_width,
        "height": frame_height,
        "channels": RAW_RGB_CHANNELS,
        "bytes_per_frame": frame_width * frame_height * RAW_RGB_CHANNELS,
    }


def _frame_shape_from_metadata(
    metadata: dict[str, int | str] | None,
) -> tuple[int, int, int] | None:
    if not metadata:
        return None
    return (
        int(metadata["height"]),
        int(metadata["width"]),
        int(metadata["channels"]),
    )


RAW_RGB_FRAMES_PER_WS_MESSAGE = 16
ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE = 6
FRAME_BATCH_PACK_OFFLOAD_BYTES = 64 * 1024
WEBP_DEFAULT_QUALITY = 90
JPEG_DEFAULT_QUALITY = 95
JPEG_SUBSAMPLING = 0
RAW_LOSSLESS_OUTPUT_FORMAT = "raw"
ENCODED_PREVIEW_FORMATS = {"webp", "jpeg"}


@dataclass(frozen=True)
class _TransportPayload:
    content_type: str
    payload: bytes
    metadata: dict[str, int | str | bool | list[int]]


def _split_frame_batch(
    frames: list[bytes],
    frames_per_message: int = RAW_RGB_FRAMES_PER_WS_MESSAGE,
) -> list[list[bytes]]:
    if not frames:
        return [frames]
    return [
        frames[i : i + frames_per_message]
        for i in range(0, len(frames), frames_per_message)
    ]


def _encode_rgb_frame_to_webp(
    frame: bytes,
    *,
    width: int,
    height: int,
    quality: int,
    preview_max_width: int | None,
) -> bytes:
    buffer = io.BytesIO()
    image = _resize_preview_image(
        Image.frombytes("RGB", (width, height), frame),
        preview_max_width=preview_max_width,
    )
    image.save(
        buffer,
        format="WEBP",
        quality=quality,
        method=0,
    )
    return buffer.getvalue()


def _encode_rgb_frame_to_jpeg(
    frame: bytes,
    *,
    width: int,
    height: int,
    quality: int,
    preview_max_width: int | None,
) -> bytes:
    buffer = io.BytesIO()
    image = _resize_preview_image(
        Image.frombytes("RGB", (width, height), frame),
        preview_max_width=preview_max_width,
    )
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=JPEG_SUBSAMPLING,
    )
    return buffer.getvalue()


def _preview_dimensions(
    *,
    width: int,
    height: int,
    preview_max_width: int | None,
) -> tuple[int, int]:
    if (
        preview_max_width is None
        or preview_max_width <= 0
        or width <= preview_max_width
    ):
        return width, height
    preview_width = int(preview_max_width)
    preview_height = max(1, round(height * preview_width / width))
    return preview_width, preview_height


def _resize_preview_image(
    image: Image.Image,
    *,
    preview_max_width: int | None,
) -> Image.Image:
    width, height = image.size
    preview_width, preview_height = _preview_dimensions(
        width=width,
        height=height,
        preview_max_width=preview_max_width,
    )
    if (preview_width, preview_height) == image.size:
        return image
    return image.resize((preview_width, preview_height), Image.Resampling.BICUBIC)


def _pack_frame_batch_message(
    header: RealtimeFrameBatchHeader,
    payload: bytes,
) -> bytes:
    message: RealtimeFrameBatchMessage = {
        **header,
        "type": "frame_batch",
        "payload": payload,
    }
    return msgspec.msgpack.encode(message)


def _pack_frame_batch_header(header: RealtimeFrameBatchHeader) -> bytes:
    return msgspec.msgpack.encode(header)


def _build_transport_payload(
    transport_frames: list[bytes],
    *,
    content_type: str,
    metadata: dict[str, int | str],
    output_format: str | None,
    transport_quality: int | None,
    preview_max_width: int | None,
) -> _TransportPayload:
    payload_content_type = content_type
    payload_metadata: dict[str, int | str | bool | list[int]] = {}
    raw_payload = b""

    if (
        output_format in ENCODED_PREVIEW_FORMATS
        and content_type == RAW_RGB_CONTENT_TYPE
        and transport_frames
    ):
        if output_format == "webp":
            encoded_frames = [
                _encode_rgb_frame_to_webp(
                    frame,
                    width=int(metadata["width"]),
                    height=int(metadata["height"]),
                    quality=int(transport_quality or WEBP_DEFAULT_QUALITY),
                    preview_max_width=preview_max_width,
                )
                for frame in transport_frames
            ]
            payload_content_type = WEBP_FRAME_CONTENT_TYPE
        else:
            encoded_frames = [
                _encode_rgb_frame_to_jpeg(
                    frame,
                    width=int(metadata["width"]),
                    height=int(metadata["height"]),
                    quality=int(transport_quality or JPEG_DEFAULT_QUALITY),
                    preview_max_width=preview_max_width,
                )
                for frame in transport_frames
            ]
            payload_content_type = JPEG_FRAME_CONTENT_TYPE
        raw_payload = b"".join(encoded_frames)
        preview_width, preview_height = _preview_dimensions(
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            preview_max_width=preview_max_width,
        )
        payload_metadata = {
            "format": output_format,
            "encoding": output_format,
            "source_width": int(metadata["width"]),
            "source_height": int(metadata["height"]),
     
```

## sglang/runtime/entrypoints/openai/realtime/realtime_video_api.py
```python
# SPDX-License-Identifier: Apache-2.0

import asyncio
import shutil
import time
from typing import TYPE_CHECKING

import msgspec.msgpack
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    ReleaseRealtimeSessionReq,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
    RealtimeChunkContext,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RealtimeFrameSendStats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.timer import (
    RealtimeStageTimer,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])
_ACTIVE_SESSION_IDS: set[str] = set()
_ACTIVE_SESSION_WAIT_SECONDS = 1.0
_ACTIVE_SESSION_WAIT_INTERVAL_SECONDS = 0.1


def _transport_ms(value: float) -> int:
    return max(0, int(value + 0.5))


async def _wait_for_active_session_slot(
    *,
    timeout_s: float = _ACTIVE_SESSION_WAIT_SECONDS,
    interval_s: float = _ACTIVE_SESSION_WAIT_INTERVAL_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while _ACTIVE_SESSION_IDS and time.monotonic() < deadline:
        await asyncio.sleep(interval_s)
    return not _ACTIVE_SESSION_IDS


def _log_realtime_chunk_timing(
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_total_ms: float,
    send_stats: RealtimeFrameSendStats,
) -> None:
    logger.info(
        "realtime chunk timing: session_id=%s request_id=%s "
        "chunk_idx=%s event_id=%s condition_kinds=%s "
        "request_prepare=%.2fms scheduler_forward=%.2fms "
        "output_pace=%.2fms "
        "header_pack=%.2fms "
        "header_write=%.2fms raw_payload_build=%.2fms raw_write=%.2fms "
        "ws_write=%.2fms chunk_total=%.2fms batches=%d frames=%d "
        "frame_shape=%s raw_bytes=%d ws_payload_bytes=%d content_type=%s",
        session.id,
        chunk.request_id,
        batch.block_idx,
        getattr(batch, "realtime_event_id", None),
        sorted(batch.condition_inputs) if batch.condition_inputs else [],
        request_prepare_ms,
        scheduler_forward_ms,
        send_stats["pace_wait_ms"],
        send_stats["header_pack_ms"],
        send_stats["header_write_ms"],
        send_stats["raw_payload_build_ms"],
        send_stats["raw_write_ms"],
        send_stats["ws_write_ms"],
        chunk_total_ms,
        send_stats["num_batches"],
        send_stats["num_frames"],
        send_stats["frame_shape"],
        send_stats["raw_bytes"],
        send_stats["ws_payload_bytes"],
        send_stats["content_type"],
    )


async def _send_realtime_chunk_stats(
    ws: WebSocket,
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_total_ms: float,
    send_stats: RealtimeFrameSendStats,
) -> None:
    await ws.send_bytes(
        msgspec.msgpack.encode(
            {
                "type": "chunk_stats",
                "session_id": session.id,
                "request_id": chunk.request_id,
                "chunk_index": batch.block_idx,
                "event_id": getattr(batch, "realtime_event_id", None),
                "request_prepare_ms": _transport_ms(request_prepare_ms),
                "scheduler_forward_ms": _transport_ms(scheduler_forward_ms),
                "pace_wait_ms": _transport_ms(send_stats["pace_wait_ms"]),
                "header_write_ms": _transport_ms(send_stats["header_write_ms"]),
                "raw_payload_build_ms": _transport_ms(
                    send_stats["raw_payload_build_ms"]
                ),
                "raw_write_ms": _transport_ms(send_stats["raw_write_ms"]),
                "ws_write_ms": _transport_ms(send_stats["ws_write_ms"]),
                "chunk_total_ms": _transport_ms(chunk_total_ms),
                "num_batches": send_stats["num_batches"],
                "num_frames": send_stats["num_frames"],
                "raw_bytes": send_stats["raw_bytes"],
                "ws_payload_bytes": send_stats["ws_payload_bytes"],
                "content_type": send_stats["content_type"],
            }
        )
    )


async def _generate_loop(ws: WebSocket, session: GenerateSession):
    adapter = session.adapter
    if adapter is None:
        raise ValueError("realtime adapter is not initialized")

    pending_send_task = None
    while not session.reached_max_chunks():
        try:
            if pending_send_task is not None and pending_send_task.done():
                await pending_send_task
                pending_send_task = None

            # send to scheduler and generate video chunk
            server_args = get_global_server_args()

            await adapter.wait_for_next_chunk(session)

            timer = RealtimeStageTimer()
            chunk_started = time.perf_counter()

            chunk = session.new_chunk()
            batch = adapter.prepare_next_request(
                session,
                server_args,
                chunk,
            )
            if batch.condition_inputs:
                logger.debug(
                    "consume realtime conditions, session_id=%s, block_idx=%s, kinds=%s",
                    session.id,
                    batch.block_idx,
                    sorted(batch.condition_inputs),
                )
            request_prepare_ms = timer.mark_ms()

            _, result = await process_generation_batch(async_scheduler_client, batch)
            scheduler_forward_ms = timer.mark_ms()

            # finish
            adapter.on_chunk_complete(session, result)
            if pending_send_task is not None:
                await pending_send_task
            if getattr(batch, "realtime_output_pacing", False):
                await _send_output_and_log(
                    ws,
                    session,
                    chunk,
                    batch,
                    result,
                    request_prepare_ms,
                    scheduler_forward_ms,
                    chunk_started,
                )
                pending_send_task = None
            else:
                pending_send_task = asyncio.create_task(
                    _send_output_and_log(
                        ws,
                        session,
                        chunk,
                        batch,
                        result,
                        request_prepare_ms,
                        scheduler_forward_ms,
                        chunk_started,
                    )
                )

        except asyncio.CancelledError:
            if pending_send_task is not None:
                pending_send_task.cancel()
                await _await_realtime_task(pending_send_task)
            logger.info("generation completed, session_id=%s", session.id)
            break
        except WebSocketDisconnect:
            if pending_send_task is not None:
                pending_send_task.cancel()
                await _await_realtime_task(pending_send_task)
            logger.info(
                "client disconnected during generation, session_id=%s", session.id
            )
  
```

## sglang/runtime/entrypoints/openai/realtime/registry.py
```python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
        BaseRealtimeModelAdapter,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


_REALTIME_ADAPTER_REGISTRY: dict[type, type[BaseRealtimeModelAdapter]] = {}
_BUILTIN_ADAPTERS_REGISTERED = False


def register_realtime_model_adapter(
    pipeline_config_cls: type,
    adapter_cls: type[BaseRealtimeModelAdapter],
) -> None:
    _REALTIME_ADAPTER_REGISTRY[pipeline_config_cls] = adapter_cls


def _register_builtin_realtime_model_adapters() -> None:
    global _BUILTIN_ADAPTERS_REGISTERED
    if _BUILTIN_ADAPTERS_REGISTERED:
        return

    from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
        LingBotWorldCausalDMDConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import (
        SanaWMRealtimeConfig,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.lingbot_world_realtime_adapter import (
        LingBotWorldRealtimeAdapter,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.sana_wm_realtime_adapter import (
        SanaWMRealtimeAdapter,
    )

    register_realtime_model_adapter(
        LingBotWorldCausalDMDConfig,
        LingBotWorldRealtimeAdapter,
    )
    register_realtime_model_adapter(
        SanaWMRealtimeConfig,
        SanaWMRealtimeAdapter,
    )
    _BUILTIN_ADAPTERS_REGISTERED = True


def get_realtime_model_adapter(
    server_args: ServerArgs,
) -> BaseRealtimeModelAdapter:
    _register_builtin_realtime_model_adapters()

    pipeline_config = server_args.pipeline_config
    for config_cls in type(pipeline_config).__mro__:
        adapter_cls = _REALTIME_ADAPTER_REGISTRY.get(config_cls)
        if adapter_cls is not None:
            return adapter_cls()

    raise ValueError(
        "Realtime video is not supported for pipeline config "
        f"{type(pipeline_config).__name__}; no realtime adapter is registered."
    )


def has_realtime_model_adapter(server_args: ServerArgs) -> bool:
    """Whether the resolved pipeline config has a registered adapter."""
    _register_builtin_realtime_model_adapters()
    return any(
        config_cls in _REALTIME_ADAPTER_REGISTRY
        for config_cls in type(server_args.pipeline_config).__mro__
    )

```

## sglang/runtime/entrypoints/openai/realtime/timer.py
```python
# SPDX-License-Identifier: Apache-2.0

import time


class RealtimeStageTimer:
    __slots__ = ("_last", "_start")

    def __init__(self):
        now = time.perf_counter()
        self._start = now
        self._last = now

    def mark_ms(self) -> float:
        now = time.perf_counter()
        elapsed_ms = (now - self._last) * 1000.0
        self._last = now
        return elapsed_ms

    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

```

## sglang/runtime/entrypoints/openai/storage.py
```python
import asyncio
import os
from typing import Optional

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CloudStorage:
    def __init__(self):
        self.enabled = os.getenv("SGLANG_CLOUD_STORAGE_TYPE", "").lower() == "s3"
        if not self.enabled:
            return

        try:
            import boto3
        except ImportError:
            logger.error(
                "boto3 is not installed. Please install it with `pip install boto3` to use cloud storage."
            )
            self.enabled = False
            return

        self.bucket_name = os.getenv("SGLANG_S3_BUCKET_NAME")
        if not self.bucket_name:
            self.enabled = False
            return

        endpoint_url = os.getenv("SGLANG_S3_ENDPOINT_URL") or None
        region_name = os.getenv("SGLANG_S3_REGION_NAME") or None

        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("SGLANG_S3_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SGLANG_S3_SECRET_ACCESS_KEY"),
            endpoint_url=endpoint_url,
            region_name=region_name,
        )
        self.endpoint_url = endpoint_url
        self.region_name = region_name

    def is_enabled(self) -> bool:
        return self.enabled

    async def upload_file(self, local_path: str, destination_key: str) -> Optional[str]:
        if not self.is_enabled():
            return None

        def _sync_upload():
            """Synchronous part of the upload to run in a thread."""
            ext = os.path.splitext(local_path)[1].lower()
            content_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".mp4": "video/mp4",
                ".glb": "model/gltf-binary",
                ".obj": "text/plain",
            }.get(ext, "application/octet-stream")

            # Use the client created once in __init__
            self.client.upload_file(
                local_path,
                self.bucket_name,
                destination_key,
                ExtraArgs={"ContentType": content_type},
            )

        try:
            # Offload the blocking I/O call to a thread executor
            await asyncio.get_running_loop().run_in_executor(None, _sync_upload)
        except Exception as e:
            # If upload fails, log the error and return None for fallback
            logger.error(f"Upload failed for {destination_key}: {e}")
            return None

        # Simplified URL generation with a default region
        if self.endpoint_url:
            url = (
                f"{self.endpoint_url.rstrip('/')}/{self.bucket_name}/{destination_key}"
            )
        else:
            region = self.region_name or "us-east-1"
            url = f"https://{self.bucket_name}.s3.{region}.amazonaws.com/{destination_key}"

        logger.info(f"Uploaded {local_path} to {url}")
        return url

    async def upload_and_cleanup(self, file_path: str) -> Optional[str]:
        """Helper to upload a file and delete the local copy if successful."""
        if not self.is_enabled():
            return None

        key = os.path.basename(file_path)
        url = await self.upload_file(file_path, key)

        if url:
            try:
                # pass if removal fails
                os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        return url


# Global instance
cloud_storage = CloudStorage()

```

## sglang/runtime/entrypoints/openai/stores.py
```python
import asyncio
from typing import Any, Dict, Optional


class AsyncDictStore:
    """A small async-safe in-memory key-value store for dict items.

    This encapsulates the usual pattern of a module-level dict guarded by
    an asyncio.Lock and provides simple CRUD methods that are safe to call
    concurrently from FastAPI request handlers and background tasks.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, key: str, value: Dict[str, Any]) -> None:
        async with self._lock:
            self._items[key] = value

    async def update_fields(
        self, key: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            item.update(updates)
            return item

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._items.get(key)

    async def pop(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._items.pop(key, None)

    async def list_page(
        self,
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = "desc",
    ) -> list[Dict[str, Any]]:
        """Return a created-time ordered page, with OpenAI-style cursor semantics."""
        normalized_order = (order or "desc").lower()
        reverse = normalized_order != "asc"
        async with self._lock:
            items = sorted(
                self._items.values(),
                key=lambda item: item.get("created_at", 0),
                reverse=reverse,
            )

        if after is not None:
            try:
                index = next(i for i, item in enumerate(items) if item["id"] == after)
                items = items[index + 1 :]
            except StopIteration:
                return []

        return items[:limit] if limit is not None else items


# Global stores shared by OpenAI entrypoints
# [request_id, dict]
VIDEO_STORE = AsyncDictStore()
IMAGE_STORE = AsyncDictStore()
MESH_STORE = AsyncDictStore()

```

## sglang/runtime/entrypoints/openai/utils.py
```python
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import asyncio
import dataclasses
import inspect
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from functools import cache
from typing import Any, Generator, List, Literal, Optional, Union

import httpx
from fastapi import HTTPException, UploadFile

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    ShutdownReq,
    UnmergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    format_lora_message,
    save_outputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.common import parse_size
from sglang.multimodal_gen.runtime.utils.image_io import save_base64_image_to_path
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    init_logger,
    log_batch_completion,
    log_generation_timer,
)
from sglang.multimodal_gen.runtime.utils.trace_wrapper import trace_req

# re-export LoRA protocol types for backward compatibility
__all__ = [
    "SetLoraReq",
    "MergeLoraWeightsReq",
    "UnmergeLoraWeightsReq",
    "ListLorasReq",
    "ShutdownReq",
    "format_lora_message",
]

logger = init_logger(__name__)

OUTPUT_QUALITY_MAPPER = {"maximum": 100, "high": 90, "medium": 55, "low": 35}
DEFAULT_FPS = 24
DEFAULT_VIDEO_SECONDS = 4


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _parse_size_or_raise(size: str) -> tuple[int, int]:
    width, height = parse_size(size)
    if width is None or height is None or width <= 0 or height <= 0:
        raise _bad_request("size must be formatted as positive WIDTHxHEIGHT")
    return width, height


def _validate_positive_int(kwargs: dict[str, Any], name: str) -> None:
    value = kwargs.get(name)
    if value is not None and int(value) <= 0:
        raise _bad_request(f"{name} must be positive")


def flatten_extra_params(payload: Any) -> dict[str, Any]:
    """Promote vLLM-Omni-style extra_params into regular request fields."""
    if not isinstance(payload, dict):
        return {}

    extra_params = payload.pop("extra_params", None)
    if isinstance(extra_params, str):
        try:
            extra_params = json.loads(extra_params)
        except Exception:
            extra_params = None
    if not isinstance(extra_params, dict):
        if "guardrails" in payload:
            payload.setdefault("use_guardrails", payload["guardrails"])
        return payload

    for key, value in extra_params.items():
        payload.setdefault(key, value)
    if "guardrails" in extra_params:
        payload.setdefault("use_guardrails", extra_params["guardrails"])

    return payload


_REQUEST_EXTRA_CONTAINERS = (
    "extra_body",
    "extra_json",
    "extra_args",
    "extra_params",
)


def _parse_request_extra_container(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    if not isinstance(value, dict):
        return {}
    return flatten_extra_params(dict(value))


def request_extra_value(request: Any, field_name: str) -> Any:
    """Read an extension field while preserving top-level precedence.

    This function only handles transport compatibility. Callers must first use
    the active SamplingParams subclass to decide which model-owned fields are
    valid; transport helpers must not introduce per-model allowlists.
    """

    extra = dict(getattr(request, "model_extra", None) or {})
    direct = {
        key: value
        for key, value in extra.items()
        if key not in _REQUEST_EXTRA_CONTAINERS
    }
    direct = flatten_extra_params(direct)
    if field_name in direct and direct[field_name] is not None:
        return direct[field_name]

    for container_name in _REQUEST_EXTRA_CONTAINERS:
        nested = _parse_request_extra_container(extra.get(container_name))
        if field_name in nested and nested[field_name] is not None:
            return nested[field_name]
    return None


@cache
def get_declared_request_extra_fields(
    sampling_params_cls: type[SamplingParams],
    api: Literal["image", "video"],
) -> frozenset[str]:
    """Return the active model's accepted fields, including transport aliases."""

    if api == "image":
        return sampling_params_cls.image_request_extra_fields()
    return sampling_params_cls.video_request_extra_fields()


@cache
def get_sampling_request_extra_fields(
    sampling_params_cls: type[SamplingParams],
    api: Literal["image", "video"],
) -> frozenset[str]:
    """Return declared extension fields that can initialize SamplingParams.

    A video declaration may also contain transport-only aliases. Those remain
    on the request for the model's lowering hook instead of being passed to the
    dataclass constructor.
    """

    declared = get_declared_request_extra_fields(sampling_params_cls, api)
    init_fields = {
        field.name for field in dataclasses.fields(sampling_params_cls) if field.init
    }
    return declared & init_fields


@contextmanager
def temp_dir_if_disabled(
    configured_path: str | None,
) -> Generator[str, None, None]:
    """Yield *configured_path* when it is set, otherwise create a temporary
    directory that is automatically removed when the context exits."""
    if configured_path is not None:
        os.makedirs(configured_path, exist_ok=True)
        yield configured_path
    else:
        tmp = tempfile.mkdtemp(prefix="sglang_")
        try:
            yield tmp
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def choose_output_image_ext(
    output_format: Optional[str], background: Optional[str]
) -> str:
    fmt = (output_format or "").lower()
    if fmt in {"png", "webp", "jpeg", "jpg"}:
        return "jpg" if fmt == "jpeg" else fmt
    if (background or "auto").lower() == "transparent":
        return "png"
    return "jpg"


def build_sampling_params(request_id: str, **kwargs) -> SamplingParams:
    """Build SamplingParams from request parameters.

    Handles size parsing, output_quality resolution, and None filtering before
    delegating to SamplingParams.from_user_sampling_params_args. Callers pass
    only the parameters they have; None values are stripped automatically so
    that SamplingParams defaults apply.
    """
    server_args = get_global_server_args()

    # pop HTTP-layer params that aren't SamplingParams fields
    output_quality = kwargs.pop("output_quality", None)
    request_data_type = kwargs.pop("request_data_type", None)
    if request_data_type is not None:
        try:
            kwargs["task_type"] = server_args.pipeline_config.resolve_task_type(
                kwargs.get("task_type"),
                data_type=request_data_type,
                has_image=bool(kwargs.get("image_path")),
                has_video=bool(kwargs.get("video_path")),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    has_explicit_compression = kwargs.get("output_compression") is not None

    # parse "WxH" size string if provided
    size = kwargs.pop("size", None)
    if size:
        w, h = _parse_size_or_raise(size)
        # treat None dimensions as unset so parsed size can fill them
        if kwargs.get("width") is None:
            kwargs["width"] = w
        if kwargs.get("height") is None:
            kwargs["height"] = h

    for name in (
        "width",
        "height",
        "num_frames",
        "num_inferenc
```

## sglang/runtime/entrypoints/openai/video_api.py
```python
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import json
import os
import shutil
import tempfile
import time
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.prompt_enhancement import (
    maybe_enhance_prompt,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
    VideoListResponse,
    VideoResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    DEFAULT_FPS,
    DEFAULT_VIDEO_SECONDS,
    add_common_data_to_response,
    build_sampling_params,
    flatten_extra_params,
    get_declared_request_extra_fields,
    get_sampling_request_extra_fields,
    merge_image_input_list,
    process_generation_batch,
    request_extra_value,
    resolve_sampling_params_cls,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.observability.trace import extract_trace_headers

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])

_VIDEO_EXTENSIONS = {
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
_VIDEO_JOB_TASKS: dict[str, asyncio.Task[None]] = {}


def _discard_video_job_task(job_id: str, task: asyncio.Task[None]) -> None:
    if _VIDEO_JOB_TASKS.get(job_id) is task:
        del _VIDEO_JOB_TASKS[job_id]


def _start_video_job(job_id: str, job: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    task = asyncio.create_task(job, name=f"video-job-{job_id}")
    _VIDEO_JOB_TASKS[job_id] = task
    task.add_done_callback(lambda completed: _discard_video_job_task(job_id, completed))
    return task


async def shutdown_video_jobs() -> None:
    tasks = list(_VIDEO_JOB_TASKS.values())
    _VIDEO_JOB_TASKS.clear()
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task


def _extra_value(request: VideoGenerationsRequest, name: str) -> Any:
    return request_extra_value(request, name)


def _request_value(request: VideoGenerationsRequest, name: str) -> Any:
    value = getattr(request, name, None)
    if value is not None:
        return value
    return _extra_value(request, name)


def _video_request_model_kwargs(
    request: VideoGenerationsRequest,
    sampling_params_cls: type[SamplingParams],
) -> dict[str, Any]:
    """Extract fields owned and declared by the active model contract."""

    kwargs = {}
    for field_name in get_sampling_request_extra_fields(sampling_params_cls, "video"):
        value = _extra_value(request, field_name)
        if value is not None:
            kwargs[field_name] = value
    return kwargs


def _parse_form_extra_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


_MULTIPART_EXTRA_FORM_FIELDS = (
    "attention_backend_override",
    "cache_dit_params",
    "cfg_gate_step",
    "enable_cache_dit",
    "quality",
    "skip_softmax_params",
)


def _multipart_extra_form_keys(
    sampling_params_cls: type[SamplingParams],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *VideoGenerationsRequest.model_fields,
                *_MULTIPART_EXTRA_FORM_FIELDS,
                *sorted(
                    get_declared_request_extra_fields(sampling_params_cls, "video")
                ),
            )
        )
    )


def _filter_multipart_declared_fields(
    extra_from_form: Dict[str, Any],
    sampling_params_cls: type[SamplingParams],
) -> Dict[str, Any]:
    declared = set(_multipart_extra_form_keys(sampling_params_cls))
    return {key: value for key, value in extra_from_form.items() if key in declared}


def _merge_multipart_extra_form_fields(
    raw_form: Any,
    extra_from_form: Dict[str, Any],
    sampling_params_cls: type[SamplingParams],
) -> None:
    for key in _multipart_extra_form_keys(sampling_params_cls):
        if key in raw_form:
            extra_from_form[key] = _parse_form_extra_value(raw_form[key])


def _multipart_video_extras(
    raw_form: Any,
    *,
    extra_body: Any,
    extra_params: Any,
    sampling_params_cls: type[SamplingParams],
) -> Dict[str, Any]:
    """Build and validate multipart extras once for request construction."""

    extra_from_form: Dict[str, Any] = {}
    if extra_body:
        try:
            extra_from_form = flatten_extra_params(json.loads(extra_body))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail="extra_body is not valid JSON"
            ) from exc
    if extra_params:
        try:
            extra_from_form.update(
                flatten_extra_params({"extra_params": json.loads(extra_params)})
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail="extra_params is not valid JSON"
            ) from exc
    _merge_multipart_extra_form_fields(
        raw_form,
        extra_from_form,
        sampling_params_cls,
    )
    flatten_extra_params(extra_from_form)
    return _filter_multipart_declared_fields(extra_from_form, sampling_params_cls)


def _is_probably_video_source(source: Any) -> bool:
    content_type = (getattr(source, "content_type", "") or "").lower()
    if content_type.startswith("video/"):
        return True

    if isinstance(source, str):
        if source.lower().startswith("data:video"):
            return True
        source_name = source
    else:
        source_name = getattr(source, "filename", None)

    if not source_name:
        return False
    source_name = str(source_name).split("?", 1)[0].split("#", 1)[0]
    return os.path.splitext(source_name)[1].lower() in _VIDEO_EXTENSIONS


def _resolve_video_path(req: VideoGenerationsRequest) -> str | None:
    video_path = _request_value(req, "video_path") or _request_value(req, "video_url")
    if video_path:
        return str(video_path)

    input_reference = _request_value(req, "input_reference")
    if _is_probably_video_source(input_reference):
        return str(input_reference)

    reference_url = _request_value(req, "reference_url")
    if _is_probably_video_source(reference_url):
        return str(reference_url)

    return None


def _resolve_image_path(
    req: VideoGenerationsRequest, video_path: str | None
) -> str | None:
    image_path = _request_value(req, "input_reference")
    if video_path and image_path == video_path:
        return None
    if _is_probably_video_source(image_path):
        return None
    return image_path


def _build_video_sampling_params(request_id: str, request: VideoGenerationsRequest):
    """Resolve video-specific defaults (fps, seconds → num_frames) then
    delegate to the shared build_sampling_params."""
    server_args = get_global_server_args()
    sampling_params_cls = resolve_sampling_params_cls(server_args)
    seconds = request.seconds if request.seconds is not None else DEFAULT_VIDEO_SECONDS
    fps = request.fps if request.fps is not None e
```

## sglang/test/unit/test_openai_image_api.py
```python
import inspect
import os
from dataclasses import fields

from fastapi import HTTPException
from PIL import Image

from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.ernie_image import (
    ErnieImageSamplingParams,
)
from sglang.multimodal_gen.configs.sample.glmimage import GlmImageSamplingParams
from sglang.multimodal_gen.configs.sample.ideogram import Ideogram4SamplingParams
from sglang.multimodal_gen.configs.sample.longcat_image import (
    LongCatImageSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _build_image_response_kwargs,
    _fallback_image_urls,
    _get_response_resize,
    _image_request_model_kwargs,
    _raise_if_image_variant_not_found,
    _runtime_sampling_quality,
    _select_image_variant_cloud_url,
    _select_image_variant_path,
    edits,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


def test_image_edits_declares_perf_dump_path_form_field():
    assert "perf_dump_path" in inspect.signature(edits).parameters


def test_url_response_returns_one_item_per_output_path():
    paths = ["first.png", "second.png"]

    response = _build_image_response_kwargs(
        paths,
        "url",
        "a lantern",
        "req-123",
        OutputBatch(),
        cloud_urls=["https://cdn.example/first.png", "https://cdn.example/second.png"],
        fallback_urls=_fallback_image_urls("req-123", len(paths), True),
        is_persistent=True,
    )

    assert [item.url for item in response["data"]] == [
        "https://cdn.example/first.png",
        "https://cdn.example/second.png",
    ]
    assert [item.file_path for item in response["data"]] == [
        os.path.abspath("first.png"),
        os.path.abspath("second.png"),
    ]


def test_runtime_sampling_quality_preserves_the_openai_default():
    assert _runtime_sampling_quality(None) is None
    assert _runtime_sampling_quality("auto") is None
    assert _runtime_sampling_quality("lossless") == "lossless"
    assert _runtime_sampling_quality("extra-high") == "extra-high"
    assert _runtime_sampling_quality("high") == "high"


def test_longcat_image_fields_remain_model_specific():
    field_values = {
        "enable_cfg_renorm": False,
        "cfg_renorm_min": 0.25,
        "enable_prompt_rewrite": False,
    }
    request = ImageGenerationsRequest(prompt="a lantern", **field_values)
    base_fields = {field.name for field in fields(SamplingParams)}
    longcat_fields = {field.name for field in fields(LongCatImageSamplingParams)}

    for field_name, value in field_values.items():
        assert field_name not in ImageGenerationsRequest.model_fields
        assert field_name not in base_fields
        assert field_name in longcat_fields
        assert getattr(request, field_name) == value

    assert _image_request_model_kwargs(request, LongCatImageSamplingParams) == (
        field_values
    )
    assert _image_request_model_kwargs(request, SamplingParams) == {}


def test_longcat_image_fields_accept_nested_extra_body():
    request = ImageGenerationsRequest(
        prompt="a lantern",
        enable_prompt_rewrite=True,
        extra_body={
            "enable_prompt_rewrite": False,
            "enable_cfg_renorm": False,
            "cfg_renorm_min": 0.5,
        },
    )

    assert _image_request_model_kwargs(request, LongCatImageSamplingParams) == {
        "enable_prompt_rewrite": True,
        "enable_cfg_renorm": False,
        "cfg_renorm_min": 0.5,
    }


def test_other_image_extensions_remain_model_specific():
    cases = (
        (Cosmos3SamplingParams, "guidance_interval", [400.0, 1000.0]),
        (Cosmos3SamplingParams, "use_guardrails", False),
        (ErnieImageSamplingParams, "use_pe", False),
        (Ideogram4SamplingParams, "preset", "V4_TURBO_12"),
    )
    base_fields = {field.name for field in fields(SamplingParams)}

    for sampling_params_cls, field_name, value in cases:
        request = ImageGenerationsRequest(
            prompt="a lantern",
            extra_body={field_name: value},
        )
        model_fields = {field.name for field in fields(sampling_params_cls)}

        assert field_name not in ImageGenerationsRequest.model_fields
        assert field_name not in base_fields
        assert field_name in model_fields
        assert _image_request_model_kwargs(request, sampling_params_cls) == {
            field_name: value
        }
        assert _image_request_model_kwargs(request, SamplingParams) == {}


def test_cosmos_image_guardrails_alias_is_preserved():
    request = ImageGenerationsRequest(
        prompt="a lantern",
        extra_body={"guardrails": False},
    )

    assert _image_request_model_kwargs(request, Cosmos3SamplingParams) == {
        "use_guardrails": False
    }


def test_image_response_includes_resize_for_every_output():
    response = _build_image_response_kwargs(
        ["first.png", "second.png"],
        "b64_json",
        "a lantern",
        "req-123",
        OutputBatch(),
        b64_list=["first", "second"],
        resize="1280x736",
    )

    assert [item.resize for item in response["data"]] == [
        "1280x736",
        "1280x736",
    ]


def test_response_resize_is_only_populated_for_glm_image():
    glm_sampling = GlmImageSamplingParams(width=1280, height=736)

    assert _get_response_resize(glm_sampling) == "1280x736"
    assert _get_response_resize(SamplingParams(width=1280, height=736)) is None


def test_response_resize_prefers_requested_size_over_generation_canvas():
    glm_sampling = GlmImageSamplingParams(width=1280, height=736)
    glm_sampling.requested_width = 1280
    glm_sampling.requested_height = 720

    assert _get_response_resize(glm_sampling) == "1280x720"


def test_response_resize_uses_actual_generated_image_size(tmp_path):
    output_path = tmp_path / "output.png"
    Image.new("RGB", (1280, 736)).save(output_path)
    glm_sampling = GlmImageSamplingParams(image_path="input.png")

    assert _get_response_resize(glm_sampling, str(output_path)) == "1280x736"


def test_response_resize_prefers_final_output_over_sampling_canvas(tmp_path):
    output_path = tmp_path / "upscaled.png"
    Image.new("RGB", (2560, 1472)).save(output_path)
    glm_sampling = GlmImageSamplingParams(width=1280, height=736)

    assert _get_response_resize(glm_sampling, str(output_path)) == "2560x1472"


def test_url_response_uses_variant_fallback_urls_for_multiple_persistent_outputs():
    paths = ["first.png", "second.png"]

    response = _build_image_response_kwargs(
        paths,
        "url",
        "a lantern",
        "req-123",
        OutputBatch(),
        fallback_urls=_fallback_image_urls("req-123", len(paths), True),
        is_persistent=True,
    )

    assert [item.url for item in response["data"]] == [
        "/v1/images/req-123/content?variant=0",
        "/v1/images/req-123/content?variant=1",
    ]


def test_select_image_variant_path_reads_indexed_file_paths():
    item = {
        "file_path": "first.png",
        "file_paths": ["first.png", "second.png"],
    }

    assert _select_image_variant_path(item, None) == "first.png"
    assert _select_image_variant_path(item, "1") == "second.png"


def test_raise_if_image_variant_not_found_handles_out_of_range_variant():
    item = {
        "file_path": "first.png",
        "file_paths": ["first.png", "second.png"],
    }

    try:
        _raise_if_image_variant_not_found(item, "5")
    except HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "Image variant 5 not found"
    else:
        raise AssertionError("Expected HTTPException")


def test_select_image_variant_path_returns_none_for_cloud_only_variant():
    item = {
        "file_path": No
```

## sglang/test/unit/test_openai_utils.py
```python
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
from types import SimpleNamespace

from starlette.datastructures import UploadFile as StarletteUploadFile

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size_or_raise,
    _save_upload_to_path,
    _validate_positive_int,
    build_sampling_params,
)


def test_save_upload_to_path_accepts_starlette_upload_file(tmp_path):
    upload = StarletteUploadFile(
        io.BytesIO(b"image-bytes"),
        filename="input.png",
    )
    target_path = tmp_path / "input.png"

    saved_path = asyncio.run(_save_upload_to_path(upload, str(target_path)))

    assert saved_path == str(target_path)
    assert target_path.read_bytes() == b"image-bytes"


def test_parse_size_or_raise_accepts_positive_size():
    assert _parse_size_or_raise("512x768") == (512, 768)


def test_parse_size_or_raise_rejects_malformed_size():
    try:
        _parse_size_or_raise("not-a-size")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_parse_size_or_raise_rejects_non_positive_size():
    try:
        _parse_size_or_raise("0x512")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_validate_positive_int_rejects_non_positive_sampling_fields():
    try:
        _validate_positive_int({"num_frames": 0}, "num_frames")
    except Exception as exc:
        assert exc.status_code == 400
        assert "num_frames must be positive" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_build_sampling_params_resolves_size_and_explicit_dimensions(monkeypatch):
    server_args = SimpleNamespace(model_path="zai-org/GLM-Image")
    monkeypatch.setattr(openai_utils, "get_global_server_args", lambda: server_args)
    captured = {}

    def fake_from_user_sampling_params_args(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        SamplingParams,
        "from_user_sampling_params_args",
        fake_from_user_sampling_params_args,
    )

    cases = [
        (
            {"size": "500x500", "width": None, "height": None},
            (500, 500),
        ),
        (
            {"size": "1024x1024", "width": None, "height": 600},
            (1024, 600),
        ),
        (
            {"size": "500x500", "width": None, "height": 600},
            (500, 600),
        ),
        (
            {"size": "500x500", "width": 600, "height": None},
            (600, 500),
        ),
        (
            {"size": "500x500", "width": 600, "height": 700},
            (600, 700),
        ),
    ]

    for request_fields, expected in cases:
        captured.clear()

        build_sampling_params("request-id", **request_fields)

        assert (captured["width"], captured["height"]) == expected

```

## vllm_omni/entrypoints/cli/serve.py
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Omni serve command for vLLM-Omni.

Supports both multi-stage LLM models (e.g., Qwen2.5-Omni) and
diffusion models (e.g., Qwen-Image) through the same CLI interface.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
from types import FrameType
from typing import TYPE_CHECKING, Any, cast

import uvloop
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger

from vllm_omni.diffusion.registry import resolve_native_single_file
from vllm_omni.engine.arg_utils import FRONTEND_ONLY_ID_FILTER, is_frontend_only
from vllm_omni.entrypoints.api_server_shared_dir import (
    create_api_server_shared_dir,
    remove_api_server_shared_dir,
)
from vllm_omni.entrypoints.cli.logo import log_logo
from vllm_omni.entrypoints.openai.api_server import (
    omni_run_server,
    run_omni_api_server_worker_proc,
)
from vllm_omni.entrypoints.utils import parse_stage_overrides, prepare_stage_config_inputs
from vllm_omni.utils.tracking_parser import TrackingArgumentParser, TrackingNamespace

if TYPE_CHECKING:
    from vllm.v1.utils import APIServerProcessManager

    from vllm_omni.engine.stage_runtime import StageEngineLaunch, StageRuntime

logger = init_logger(__name__)

DESCRIPTION = """Launch a local OpenAI-compatible API server to serve Omni models
via HTTP. Supports both multi-stage LLM models and diffusion models.

The server automatically detects the model type:
- LLM models: Served via /v1/chat/completions endpoint
- Diffusion models: Served via /v1/images/generations endpoint

Examples:
  # Start an Omni LLM server
  vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091

  # Start a diffusion model server
  vllm serve Qwen/Qwen-Image --omni --port 8091

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,
--help=OmniConfig)
  Use `--help=all` to show all available flags at once.
"""


def _parse_stage_overrides(value: str) -> dict[str, dict[str, Any]]:
    """Adapt shared stage-override validation to argparse's error type."""
    try:
        parsed = parse_stage_overrides(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed is None:
        raise argparse.ArgumentTypeError("--stage-overrides requires a JSON object")
    return parsed


def _parse_stage_id(value: str) -> int:
    """Parse `--stage-id` as a non-negative int, or `none` (no local stage) as `FRONTEND_ONLY_ID_FILTER`."""
    if value.lower() == "none":
        return FRONTEND_ONLY_ID_FILTER
    msg = f"Stage ID must be a non-negative integer or 'none', got {value!r}"
    try:
        stage_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(msg) from exc
    if stage_id < 0:
        raise argparse.ArgumentTypeError(msg)
    return stage_id


def _nonneg_finite_float(value: str) -> float:
    """Argparse type for finite, non-negative floats (rejects nan/inf)."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(f"must be a finite non-negative number, got {value!r}")
    return parsed


def _json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def _ensure_vllm_platform():
    """Ensure vLLM's current_platform is valid before arg parsing.

    Upstream vLLM's argument parser now instantiates DeviceConfig during
    ``make_arg_parser``, which requires a resolved platform with a non-empty
    ``device_type``.  In some environments (e.g. editable installs with
    broken package metadata), vLLM's own platform auto-detection may fail
    and fall back to ``UnspecifiedPlatform``.  When that happens, use the
    Omni platform (which has its own detection logic) as a drop-in
    replacement so that argument parsing succeeds.
    """
    from vllm import platforms as vllm_platforms

    if vllm_platforms.current_platform.is_unspecified():
        from vllm_omni.platforms import current_omni_platform

        if not current_omni_platform.is_unspecified():
            vllm_platforms.current_platform = current_omni_platform
            logger.debug(
                "Replaced vLLM UnspecifiedPlatform with omni platform %s",
                type(current_omni_platform).__name__,
            )
        else:
            from vllm.platforms.cpu import CpuPlatform

            vllm_platforms.current_platform = CpuPlatform()
            logger.debug(
                "Both vLLM and omni platforms are unspecified, falling back to CpuPlatform for arg parsing",
            )


class OmniServeCommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI."""

    name = "serve"
    # Parser stashed at subparser_init so ``cmd`` can resolve each user-typed
    # flag to its real ``dest`` via the parser's action table.
    _parser: TrackingArgumentParser

    @staticmethod
    def cmd(args: TrackingNamespace) -> None:
        if not os.environ.get("VLLM_DISABLE_LOG_LOGO"):
            os.environ["VLLM_DISABLE_LOG_LOGO"] = "1"
            log_logo()

        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        if getattr(args, "no_guardrails", False):
            existing = getattr(args, "model_config", None)
            model_config = dict(existing) if isinstance(existing, dict) else {}
            model_config["guardrails"] = False
            args.model_config = model_config
            explicit_keys = getattr(args, "explicit_keys", None)
            if explicit_keys is not None:
                # --no-guardrails is a CLI-only alias, not a diffusion engine arg.
                args.explicit_keys = (explicit_keys - {"no_guardrails"}) | {"model_config"}

        if args.headless:
            run_headless(args)
        elif (getattr(args, "api_server_count", None) or 1) > 1:
            run_multi_api_server_omni(args)
        else:
            uvloop.run(omni_run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        # Only the head owns stage-metric collection; API statistics default on.
        if (
            not bool(getattr(args, "headless", False))
            and not bool(getattr(args, "disable_log_stats", False))
            and not bool(getattr(args, "collect_stage_metrics", True))
        ):
            raise ValueError(
                "--no-collect-stage-metrics requires --disable-log-stats when statistics logging is enabled"
            )

        if args.stage_id is not None and (args.omni_master_address is None or args.omni_master_port is None):
            raise ValueError("--stage-id requires both --omni-master-address and --omni-master-port to be set")
        if args.headless and is_frontend_only(args.stage_id):
            raise ValueError("--stage-id none cannot be used with --headless")

        # Require an explicit model under --omni. ``args.model`` always carries
        # vLLM's ModelConfig default (``Qwen/Qwen3-0.6B``), so an omit is silent:
        # the default text LLM is routed into the diffusion stage and startup
        # crashes deep in the diffusion worker with a confusing registry error.
        # Fail fast instead.
        #
        # The model must come from the CLI -- positionally
        # (``vllm serve <model> --omni``) or via ``--model``. A deploy YAML
        # (``--deploy-config``) is NOT a model source: it carries per-stage
        # engine args only, and the checkpoint is always threaded in from
        # ``args.model`` (see
        # ``load_and_resolve_stage_configs``, which takes ``model`` as its first
        # argument). Treating its mere presence as "model provided" let
        # ``vllm serve --omni --deploy-config <cfg>`` slip through with the
        # default model and reproduce the very crash this guard prevents. See
        # https://github.com/vllm-project/vllm-omni/issues/4158.
        if getattr(args, "omni", False):
            explicit_keys = getattr(args, "explicit_keys", None) or frozenset()
            # Resolve the model the user actually supplied on the CLI. A
            # positional ``model_tag`` takes precedence (``cmd`` later copies it
            # onto ``args.model``); otherwise ``--model`` counts only when it was
            # explicitly passed. An empty/whitespace value (e.g. ``--model
            # "$MODEL"`` with ``MODEL`` unset) is treated as "not provided" so it
            # fails here with a clear message instead of the same confusing
            # downstream crash.
            model_tag = getattr(args, "model_tag", None)
            if model_tag is not None:
                explicit_model = model_tag
            elif "model" in explicit_keys:
                explicit_model = getattr(args, "model", None)
            else:
                explicit_model = None
            model_provided = explicit_model is not None and str(explicit_model).strip() != ""
            if not model_provided:
                raise ValueError(
                    "`vllm serve --omni` requires an explicit model. Pass it "
                    "positionally (`vllm serve <model> --omni`) or via `--model`. "
                    "`--deploy-config` carries per-stage engine args only and "
                    "does not supply a model; without an "
                    "explicit model, vLLM's default (Qwen/Qwen3-0.6B) is selected "
                    "and routed into the diffusion stage, which fails with a "
                    "confusing diffusion-registry error."
                )

        # --omni-replica-address is only consulted in run_headless(); reject it
        # on the head so a misconfigured launch fails loudly instead of being
        # silently ignored.
        if getattr(args, "omni_replica_address", None) is not None and not args.headless:
            raise ValueError("--omni-replica-address requires --headless to be set")

        api_server_count = getattr(args, "api_server_count", None)
        if api_server_count is not None and api_server_count < 1 and not args.headless:
            raise ValueError("--api-server-count must be >= 1 unless --headless is set")
        if args.headless and api_server_count is not None and api_server_count > 0:
            raise ValueError("--api-server-count cannot be used with --headless")
        if api_server_count is not None and api_server_count > 1:
            distributed_args = (
                "stage_id",
                "omni_master_address",
                "omni_master_port",
                "omni_replica_address",
            )
            if any(getattr(args, name, None) is not None for name in distributed_args):
                raise ValueError(
                    "--api-server-count > 1 cannot be combined with stage-based "
                    "or distributed stage arguments (--stage-id/--omni-master-* / "
                    "--omni-replica-address)"
                )
            if getattr(args, "omni_dp_size_local", 1) != 1:
                raise ValueError("--api-server-count > 1 requires --omni-dp-size-local=1")
            if getattr(args, "worker_backend", "multi_process") != "multi_process":
                raise ValueError("--api-server-count > 1 requires --worker-backend=multi_process")
            if getattr(args, "enable_fault_tolerance", False):
                raise ValueError("--api-server-count > 1 cannot be combined with --enable-fault-tolerance")
            if getattr(args, "enable_elastic_ep", False):
                raise ValueError("--api-server-count > 1 cannot be combined with --enable-elastic-ep")
            if getattr(args, "enable_sleep_mode", False):
                raise ValueError("--api-server-count > 1 cannot be combined with --enable-sleep-mode")
            from vllm import envs as vllm_envs

            if getattr(vllm_envs, "VLLM_ALLOW_RUNTIME_LORA_UPDATING", False):
                raise ValueError("--api-server-count > 1 cannot be combined with VLLM_ALLOW_RUNTIME_LORA_UPDATING")

        # --omni-dp-size-local is process-local. A value other than 1 only
        # makes sense when this process owns a stage (head or headless).
        omni_dp_size_local = getattr(args, "omni_dp_size_local", None)
        if omni_dp_size_local is not None:
            if omni_dp_size_local < 1:
                raise ValueError(f"--omni-dp-size-local must be >= 1, got {omni_dp_size_local}")
            if omni_dp_size_local != 1 and (args.stage_id is None or is_frontend_only(args.stage_id)):
                raise ValueError("--omni-dp-size-local != 1 requires a numeric --stage-id")

        # vLLM CLI args that omni does not honor: parallelism comes from the
        # per-stage YAML (parallel_config:, enable_expert_parallel:) and the
        # process-local replica count from --omni-dp-size-local. Passing the
        # vLLM equivalents on the command line would silently disagree with
        # those sources of truth, so reject them at parse time.
        if getattr(args, "omni", False):
            explicit_cli_keys: set[str] = getattr(args, "_cli_explicit_keys", set()) or set()
            prohibited_with_omni: dict[str, str] = {
                "data_parallel_size": "--data-parallel-size",
                "data_parallel_size_local": "--data-parallel-size-local",
                "data_parallel_address": "--data-parallel-address",
                "data_parallel_rpc_port": "--data-parallel-rpc-port",
                "data_parallel_start_rank": "--data-parallel-start-rank",
                "data_parallel_backend": "--data-parallel-backend",
                "enable_expert_parallel": "--enable-expert-parallel",
            }
            offenders = sorted(flag for dest, flag in prohibited_with_omni.items() if dest in explicit_cli_keys)
            if offenders:
                raise ValueError(
                    "The following CLI args are not supported under --omni: "
                    f"{', '.join(offenders)}. Configure parallelism through the "
                    "per-stage YAML (`--deploy-config`) "
                    "and replica count via the per-stage `num_replicas` config field "
                    "(single-runtime) or `--omni-dp-size-local` (headless / multi-runtime)."
                )

        lora_backend = getattr(args, "lora_backend", No
```

## vllm_omni/entrypoints/openai/diffusion.py
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI diffusion-stage request helpers.

These helpers are stage-based rather than image-format-specific."""

import json
from http import HTTPStatus
from typing import Any, cast

from fastapi import HTTPException

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.stage_params import (
    build_stage_sampling_params_list,
    get_default_sampling_params_list,
)

MAX_UINT32_SEED = 2**32 - 1


async def _generate_with_async_omni(
    engine_client: AsyncOmni | Any,
    gen_params: Any,
    stage_configs: list[Any],
    **kwargs,
):
    engine_client = cast(AsyncOmni, engine_client)
    result = None
    normalized_stage_configs = list(stage_configs)
    if not normalized_stage_configs:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Stage configs not found. Start server with a multi-stage omni model.",
        )
    sampling_params_list = build_stage_sampling_params_list(
        normalized_stage_configs,
        get_default_sampling_params_list(engine_client),
        diffusion_params=gen_params,
        replace_diffusion_params=True,
    )

    async for output in engine_client.generate(
        sampling_params_list=sampling_params_list,
        **kwargs,
    ):
        result = output

    if result is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="No output generated from multi-stage pipeline.",
        )
    return result


def apply_stage_default_sampling_params(
    default_params_json: str | None,
    sampling_params: Any,
    stage_key: str,
) -> None:
    """
    Update a stage's sampling parameters with vLLM-Omni defaults.

    Args:
        default_params_json: JSON string of stage-keyed default parameters
        sampling_params: The sampling parameters object to update
        stage_key: The stage ID/key in the pipeline
    """
    if default_params_json is not None:
        default_params_dict = json.loads(default_params_json)
        if stage_key in default_params_dict:
            stage_defaults = default_params_dict[stage_key]
            for param_name, param_value in stage_defaults.items():
                if hasattr(sampling_params, param_name):
                    setattr(sampling_params, param_name, param_value)

```

## vllm_omni/entrypoints/openai/protocol/images.py
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
OpenAI-compatible protocol definitions for image generation.

This module provides Pydantic models that follow the OpenAI DALL-E API specification
for text-to-image generation, with vllm-omni specific extensions.
"""

import base64
import io
import uuid
import zipfile
from collections.abc import AsyncIterator
from enum import Enum
from http import HTTPStatus
from typing import Any, Literal

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vllm_omni.entrypoints.openai.image_api_utils import validate_layered_layers

# Bound int request fields to avoid overflow issues.
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

_FILE_RESPONSE_CHUNK_SIZE = 64 * 1024

_IMAGE_FILE_METADATA = {
    "jpg": ("jpg", "image/jpeg"),
    "jpeg": ("jpeg", "image/jpeg"),
    "png": ("png", "image/png"),
    "webp": ("webp", "image/webp"),
}


async def _iter_file_chunks(data: bytes) -> AsyncIterator[memoryview]:
    """Yield fixed-size chunks of an in-memory payload."""
    view = memoryview(data)
    for offset in range(0, len(view), _FILE_RESPONSE_CHUNK_SIZE):
        yield view[offset : offset + _FILE_RESPONSE_CHUNK_SIZE]


class ResponseFormat(str, Enum):
    """Image response format"""

    B64_JSON = "b64_json"
    URL = "url"  # Not implemented in PoC
    FILE = "file"  # file response


class ImageGenerationRequest(BaseModel):
    """
    OpenAI DALL-E compatible image generation request.

    Follows the OpenAI Images API specification with vllm-omni extensions
    for advanced diffusion parameters.
    """

    model_config = ConfigDict(extra="allow")

    # Required fields
    prompt: str = Field(..., description="Text description of the desired image(s)")
    bot_task: str | None = Field(
        None,
        description="Task mode for the model (e.g., 'cot' enables chain-of-thought generation). "
        "Only supported by specific diffusion models.",
    )

    # OpenAI standard fields
    model: str | None = Field(
        default=None,
        description="Model to use (optional, uses server's configured model if omitted)",
    )
    n: int = Field(default=1, ge=1, le=10, description="Number of images to generate")
    size: str | None = Field(
        default=None,
        description="Image dimensions in WIDTHxHEIGHT format (e.g., '1024x1024', uses model defaults if omitted)",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.B64_JSON, description="Format of the returned image")
    user: str | None = Field(default=None, description="User identifier for tracking")
    layers: int | None = Field(
        default=None,
        description="Number of output layers for layered image models. Supported range: 2-10.",
    )

    @field_validator("size")
    @classmethod
    def validate_size(cls, v):
        """Validate size parameter.

        Accepts any string in 'WIDTHxHEIGHT' format (e.g., '1024x1024', '512x768').
        No restrictions on specific dimensions - models can handle arbitrary sizes.
        """
        if v is None:
            return None
        # Validate string format
        if not isinstance(v, str) or "x" not in v:
            raise ValueError("size must be in format 'WIDTHxHEIGHT' (e.g., '1024x1024')")
        return v

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v):
        """Validate response format - only b64_json and file are supported."""
        if v is not None and v not in (ResponseFormat.B64_JSON, ResponseFormat.FILE):
            raise ValueError(f"Only 'b64_json' or 'file' response format is supported, got: {v}")
        return v

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, v):
        """Validate the layers parameter for layered image models."""
        return validate_layered_layers(v)

    # vllm-omni extensions for diffusion control
    negative_prompt: str | None = Field(default=None, description="Text describing what to avoid in the image")
    system_prompt: str | None = Field(
        default=None, description="Custom system prompt. Used when --use_system_prompt is custom"
    )
    use_system_prompt: str | None = Field(
        default=None,
        description="System prompt type. Options: None, dynamic, en_vanilla, "
        "en_recaption, en_think_recaption, en_unified, custom",
    )

    @field_validator("use_system_prompt")
    @classmethod
    def validate_use_system_prompt(cls, v):
        """Validate system prompt type."""
        valid_types = [None, "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "custom"]
        if v not in valid_types:
            raise ValueError(f"Invalid use_system_prompt type: {v}. Must be one of: {valid_types[1:] + [None]}")
        return v

    num_inference_steps: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Number of diffusion sampling steps (uses model defaults if not specified)",
    )
    guidance_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Classifier-free guidance scale (uses model defaults if not specified)",
    )
    true_cfg_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="True CFG scale (model-specific parameter, may be ignored if not supported)",
    )
    flow_shift: float | None = Field(
        default=None, description="Scheduler flow_shift (sigma shift) for flow-matching diffusion models."
    )
    extra_params: dict[str, Any] | None = Field(
        default=None,
        description="Optional model-specific parameters passed directly to the model's extra_args.",
    )
    seed: int | None = Field(default=None, ge=_INT64_MIN, le=_INT64_MAX, description="Random seed for reproducibility")
    generator_device: str | None = Field(
        default=None,
        description="Device for the seeded torch.Generator (e.g. 'cpu', 'cuda'). Defaults to the runner's device.",
    )

    # vllm-omni extension for per-request LoRA.
    # This mirrors the `extra_body.lora` convention in /v1/chat/completions.
    lora: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional LoRA adapter for this request. Expected shape: "
            "{name/path/scale/int_id}. Field names are flexible "
            "(e.g. name|lora_name|adapter, path|lora_path|local_path, "
            "scale|lora_scale, int_id|lora_int_id)."
        ),
    )

    # VAE memory optimizations (set at model init, included for completeness)
    vae_use_slicing: bool | None = Field(default=False, description="Enable VAE slicing")
    vae_use_tiling: bool | None = Field(default=False, description="Enable VAE tiling")

    # Output format for generated images
    output_format: str | None = Field(
        default=None,
        description="Output image format: 'png', 'jpeg', or 'webp'. Defaults to 'png'.",
    )
    output_compression: int = Field(
        default=100,
        ge=0,
        le=100,
        description=(
            "Compression level 0-100. For 'jpeg'/'webp' this is the encoder "
            "quality. For 'png' 100 keeps the fastest, least-compressed "
            "encode and lower values trade encode time for smaller payloads "
            "(100 -> compress_level 0, 1 -> compress_level 9)."
        ),
    )
    return_stage_metrics: bool | None = Field(
        default=None,
        description="Return stage metrics for benchmark clients.",
    )


class ImageData(BaseModel):
    """Single generated image data"""

    b64_json: str | None = Field(default=None, description="Base64-encoded PNG image")
    url: str | None = Field(default=None, description="Image URL (not implemented)")
    revised_prompt: str | None = Field(default=None, description="Revised prompt (OpenAI compatibility, always null)")


class ImageGenerationResponse(BaseModel):
    """
    OpenAI DALL-E compatible image generation response.

    Returns generated images with metadata.
    """

    created: int = Field(..., description="Unix timestamp of when the generation completed")
    data: list[ImageData] = Field(..., description="Array of generated images")
    output_format: str | None = Field(None, description="The output format of the image generation")
    size: str = Field(None, description="The size of the image generated")
    cot_output: str | None = Field(
        None,
        description="Chain-of-thought text output from the AR stage. "
        "Only present for image editing (IT2I) with CoT-enabled models.",
    )
    metrics: dict[str, Any] | None = Field(
        default=None,
        description="Per-request generation metrics.",
    )

    def stream_response(self) -> StreamingResponse:
        if not self.data or not self.data[0].b64_json:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No image data available for file response.",
            )
        extension, media_type = _IMAGE_FILE_METADATA.get(
            (self.output_format or "png").lower(),
            _IMAGE_FILE_METADATA["png"],
        )
        if len(self.data) == 1:
            image_bytes = base64.b64decode(self.data[0].b64_json)
            filename = f"image_{uuid.uuid4().hex[:8]}.{extension}"
            return StreamingResponse(
                _iter_file_chunks(image_bytes),
                media_type=media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Length": str(len(image_bytes)),
                },
            )
        else:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, item in enumerate(self.data):
                    if item.b64_json:
                        zf.writestr(f"image_{idx}.{extension}", base64.b64decode(item.b64_json))
            zip_bytes = zip_buffer.getvalue()
            filename = f"images_{uuid.uuid4().hex[:8]}.zip"
            return StreamingResponse(
                _iter_file_chunks(zip_bytes),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Length": str(len(zip_bytes)),
                },
            )


class ImageEditARDeltaChunk(BaseModel):
    """Streaming chunk carrying a text delta from the image-edit AR stage."""

    object: Literal["image.edit.chunk"] = "image.edit.chunk"
    type: Literal["ar_delta"] = "ar_delta"
    delta: str = Field(..., description="Text delta generated by the AR stage")
    index: int = Field(default=0, description="Completion index for the AR stage output")
    created: int = Field(..., description="Unix timestamp of when the stream was created")
    model: str = Field(..., description="Model used for the image edit request")
    metrics: dict[str, Any] | None = Field(
        default=None,
        description="Optional vLLM-Omni per-stage metrics snapshot for benchmark clients.",
    )


class ImageEditImageChunk(BaseModel):
    """Streaming chunk carrying the final image-edit result."""

    object: Literal["image.edit.chunk"] = "image.edit.chunk"
    type: Literal["image"] = "image"
    data: list[ImageData] = Field(..., description="Array of generated images")
    output_format: str = Field(..., description="The output format of the image generation")
    size: str = Field(..., description="The generated image size")
    created: int = Field(..., description="Unix timestamp of when the stream was created")
    model: str = Field(..., description="Model used for the image edit request")
    metrics: dict[str, Any] | None = Field(
        default=None,
        description="Optional vLLM-Omni per-stage metrics snapshot for benchmark clients.",
    )


class ImageEditStreamError(BaseModel):
    """Streaming error chunk emitted before the terminal [DONE] event."""

    object: Literal["error"] = "error"
    created: int = Field(..., description="Unix timestamp of when the stream was created")
    model: str = Field(..., description="Model used for the image edit request")
    error: dict[str, Any] = Field(..., description="OpenAI-compatible streaming error payload")


ImageEditStreamResponse = ImageEditARDeltaChunk | ImageEditImageChunk | ImageEditStreamError

```

## vllm_omni/diffusion/attention/selector.py
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Diffusion attention backend selector.

This module resolves diffusion attention backends from:
1. per-role AttentionConfig
2. platform default
"""

from __future__ import annotations

import importlib
from functools import cache
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

logger = init_logger(__name__)

# SequenceParallel auto-pad probes mask support before Attention layers exist,
# so it has no real head_dim. Callers must not treat this sentinel as geometry.
HEAD_SIZE_UNKNOWN = -1


def _load_backend_cls(cls_path: str) -> type[AttentionBackend]:
    """Load a backend class from its fully qualified path.

    Args:
        cls_path: Fully qualified class path (e.g.,
            "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend")

    Returns:
        The loaded backend class
    """
    module_path, class_name = cls_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        backend_class = getattr(module, class_name)
        return backend_class
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in module: {e}")


@cache
def _cached_get_backend_cls(
    backend_name: str | None,
    head_size: int,
    allow_trtllm_default: bool = True,
) -> type[AttentionBackend]:
    """Cache backend class resolution by (backend_name, head_size, allow_trtllm_default).

    This ensures platform validation (compute capability checks, package
    availability, etc.) runs only once per unique combination, avoiding
    repeated log messages.
    """
    from vllm_omni.platforms import current_omni_platform

    backend_cls_path = current_omni_platform.get_diffusion_attn_backend_cls(
        selected_backend=backend_name,
        head_size=head_size,
        allow_trtllm_default=allow_trtllm_default,
    )
    return _load_backend_cls(backend_cls_path)


@cache
def _log_backend_resolution(
    role: str,
    role_category: str | None,
    backend_name: str,
    source: str,
) -> None:
    if role_category is not None:
        logger.info(
            "Resolved diffusion attention backend '%s' for role=%r (role_category=%r) via %s",
            backend_name,
            role,
            role_category,
            source,
        )
        return

    logger.info(
        "Resolved diffusion attention backend '%s' for role=%r via %s",
        backend_name,
        role,
        source,
    )


def get_attn_backend_for_role(
    role: str,
    head_size: int,
    attention_config: AttentionConfig | None = None,
    role_category: str | None = None,
    allow_trtllm_default: bool = True,
) -> tuple[type[AttentionBackend], AttentionSpec | None]:
    """
    Get attention backend for a specific attention role.

    Lookup precedence:
      1. attention_config.per_role[role]           — exact match
      2. attention_config.per_role[role_category]   — category fallback
      3. attention_config.default                   — global default
      4. Platform default                           — hardware-specific

    Args:
        role: Attention role string (e.g. "self", "cross", "joint",
              "ltx2.audio_to_video")
        head_size: Head size for attention computation
        attention_config: The AttentionConfig from OmniDiffusionConfig.
            If None, falls back to platform default behavior.
        role_category: Optional category for fallback (e.g. "cross" for
            "ltx2.audio_to_video")

    Returns:
        Tuple of (backend_class, AttentionSpec or None).
        AttentionSpec is None when using platform default without explicit config.
    """
    # 1. Try config from OmniDiffusionConfig
    spec = None
    source = None
    if attention_config is not None:
        spec, source = attention_config.resolve_with_source(
            role=role,
            role_category=role_category,
        )

    if spec is not None:
        backend_cls = _cached_get_backend_cls(spec.backend, head_size)
        _log_backend_resolution(
            role=role,
            role_category=role_category,
            backend_name=backend_cls.get_name(),
            source=source or "attention_config",
        )
        return backend_cls, spec

    # 2. Platform default
    backend_cls = _cached_get_backend_cls(None, head_size, allow_trtllm_default)
    _log_backend_resolution(
        role=role,
        role_category=role_category,
        backend_name=backend_cls.get_name(),
        source="platform default",
    )
    return backend_cls, None


def get_attn_backend_for_capability(
    role: str,
    attention_config: AttentionConfig | None = None,
    role_category: str | None = None,
) -> type[AttentionBackend]:
    """Resolve the backend class for capability queries such as mask support.

    SequenceParallel auto-pad does not know ``head_size``. Explicit
    ``AttentionConfig`` selections are loaded from the registry without
    geometry validation, so CUDNN_ATTN is not rejected for the unknown
    sentinel. Platform defaults still consult the platform with
    ``HEAD_SIZE_UNKNOWN``.
    """
    spec = None
    if attention_config is not None:
        spec, _ = attention_config.resolve_with_source(
            role=role,
            role_category=role_category,
        )
    if spec is not None:
        from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum

        return DiffusionAttentionBackendEnum[spec.backend.upper()].get_class()
    return _cached_get_backend_cls(None, HEAD_SIZE_UNKNOWN)

```

## vllm_omni/diffusion/models/flux/pipeline_flux.py
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import logging
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoConfig, CLIPTextModel, CLIPTokenizer, T5TokenizerFast
from vllm.model_executor.models.transformers.utils import init_on_device_without_buffers
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.flux import FluxTransformer2DModel
from vllm_omni.diffusion.models.flux.flux_pipeline_mixin import FluxPipelineMixin
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.t5_encoder import T5EncoderModel
from vllm_omni.diffusion.models.utils import init_parameters, recursive_replace_linear
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = logging.getLogger(__name__)


def get_flux_post_process_func(
    od_config: OmniDiffusionConfig,
):
    if od_config.output_type == "latent":
        return lambda x: x
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    if not os.path.exists(vae_config_path):
        raise FileNotFoundError(
            f"VAE config not found at {vae_config_path}. "
            "Please ensure the model path contains a valid VAE configuration."
        )
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


class FluxPipeline(
    nn.Module, FluxPipelineMixin, CFGParallelMixin, DiffusionPipelineProfilerMixin, SupportsComponentDiscovery
):
    supports_request_batch = True

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "text_encoder_2"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="text_encoder",
                revision=None,
                prefix="text_encoder.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="text_encoder_2",
                revision=None,
                prefix="text_encoder_2.",
                fall_back_to_pt=True,
            ),
        ]

        self.device = get_local_device()
        model = od_config.model
        # Check if model is a local path
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        # CLIP: meta-init + swap to vLLM linears so quant_config applies; loader
        # streams weights, so no .to() (same as the DiT).
        clip_config = AutoConfig.from_pretrained(model, subfolder="text_encoder", local_files_only=local_files_only)
        with init_on_device_without_buffers("meta"):
            self.text_encoder = CLIPTextModel._from_config(clip_config)
        recursive_replace_linear(self.text_encoder, od_config)
        init_parameters(self.text_encoder, dtype=od_config.dtype, device=self.device)

        # T5 already uses vLLM linears; thread quant_config to quantize its GEMMs.
        t5_config = AutoConfig.from_pretrained(model, subfolder="text_encoder_2", local_files_only=local_files_only)
        self.text_encoder_2 = T5EncoderModel(
            t5_config, prefix="text_encoder_2", quant_config=od_config.quantization_config
        )

        # VAE is precision-sensitive and kept at full precision.
        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
            self.device
        )

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, FluxTransformer2DModel)
        self.transformer = FluxTransformer2DModel(
            **transformer_kwargs, od_config=od_config, quant_config=od_config.quantization_config
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local_files_only
        )

        self.stage = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed.
        # This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        # self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. "
                "Make sure to generate `pooled_prompt_embeds` from the same text encoder "
                "that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. "
                "Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder "
                "that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        dtype: torch.dtype | None = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(self.device))[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str],
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        max_sequence_length: int = 512,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                T
```

## vllm_omni/entrypoints/openai/diffusion_request_utils.py
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Collection, Mapping

from vllm.logger import init_logger

from vllm_omni.inputs.data import (
    DIFFUSION_QUALITY_LEVELS,
    OmniDiffusionSamplingParams,
)

logger = init_logger(__name__)


def _request_mapping(name: str, value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys.")
    return dict(value)


def normalize_diffusion_request_args(
    *,
    root: object | None = None,
    nested: object | None = None,
    explicit_root_args: object | None = None,
    serving_root_fields: Collection[str] = (),
    registered_extra_fields: Collection[str] = (),
    root_field_aliases: Mapping[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate request sources and project diffusion consumer arguments."""
    root_args = _request_mapping("request", root)
    root_args.update(_request_mapping("explicit_root_args", explicit_root_args))
    nested_args = _request_mapping("request.extra_body", nested)

    registered = set(registered_extra_fields)
    aliases = {alias: canonical for alias, canonical in (root_field_aliases or {}).items() if alias not in registered}
    serving_fields = set(serving_root_fields) | aliases.keys()
    declared_fields = registered - serving_fields
    consumer_fields = serving_fields | declared_fields

    root_declared = {key: root_args[key] for key in declared_fields if key in root_args}
    nested_declared = {key: nested_args[key] for key in declared_fields if key in nested_args}
    canonical = _request_mapping("extra_args", root_args.get("extra_args"))
    legacy = _request_mapping("extra_params", root_args.get("extra_params"))
    nested_canonical = _request_mapping("extra_body.extra_args", nested_args.get("extra_args"))
    nested_legacy = _request_mapping("extra_body.extra_params", nested_args.get("extra_params"))

    sources_by_key: dict[str, list[str]] = {}
    sources = (
        ((root_args.keys() & serving_fields) | root_declared.keys(), "request"),
        ((nested_args.keys() & serving_fields) | nested_declared.keys(), "request.extra_body"),
        (canonical, "request.extra_args"),
        (legacy, "request.extra_params"),
        (nested_canonical, "request.extra_body.extra_args"),
        (nested_legacy, "request.extra_body.extra_params"),
    )
    for keys, prefix in sources:
        for key in sorted(keys):
            sources_by_key.setdefault(aliases.get(key, key), []).append(f"{prefix}.{key}")

    conflicts = {key: list(dict.fromkeys(paths)) for key, paths in sources_by_key.items() if len(set(paths)) > 1}
    if conflicts:
        details = "; ".join(f'"{key}": {", ".join(conflicts[key])}' for key in sorted(conflicts))
        raise ValueError(f"Diffusion request parameters were provided more than once: {details}.")

    if root_args.get("extra_params") is not None or nested_args.get("extra_params") is not None:
        logger.warning_once(
            "extra_params is deprecated; use extra_args for model-specific diffusion request parameters."
        )

    # Duplicate request keys were rejected above, so ordering is not precedence.
    normalized_extra_args = {
        **nested_declared,
        **root_declared,
        **legacy,
        **nested_legacy,
        **canonical,
        **nested_canonical,
    }
    normalized_extra_args = {key: value for key, value in normalized_extra_args.items() if value is not None}

    request_args = {
        **{key: nested_args[key] for key in consumer_fields if key in nested_args},
        **{key: root_args[key] for key in consumer_fields if key in root_args},
    }
    for alias, canonical_name in aliases.items():
        if alias in request_args:
            request_args[canonical_name] = request_args.pop(alias)
    quality = request_args.get("quality")
    if quality is not None and quality not in DIFFUSION_QUALITY_LEVELS:
        raise ValueError(f"quality must be one of {list(DIFFUSION_QUALITY_LEVELS)}, got {quality!r}")
    return normalized_extra_args, request_args


def apply_normalized_diffusion_request_extra_args(
    sampling_params: OmniDiffusionSamplingParams,
    normalized_extra_args: Mapping[str, object],
) -> None:
    """Overlay request extras without discarding stage defaults."""
    if normalized_extra_args:
        sampling_params.extra_args = {
            **(sampling_params.extra_args or {}),
            **normalized_extra_args,
        }

```

## vllm_omni/entrypoints/openai/images/__init__.py
```python

```

## vllm_omni/entrypoints/openai/images/helpers.py
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Image endpoint helpers peeled from ``api_server.py``.

PUT HERE:
  - ``/v1/images*`` request/engine helpers: limits, input loading, edit extras,
    result extraction/normalization, and other route-adjacent logic that used
    to live in ``api_server.py``.
  - May import ``image_api_utils`` for encode/validate (helpers → utils OK).

DO NOT PUT HERE:
  - Shared encode/size/layered validation — keep those in ``image_api_utils``.

LONGEVITY:
  - This package ``helpers.py`` is the **longer home** for image endpoint
    logic through the helper (P0.2) + router (P0.3) stages.
  - The later P1.2 image modality PR (#5227) may further split this file (e.g.
    validation / editing / responses) and absorb ``image_api_utils`` into
    this package; until then keep growing here, not in root utils.

See ``images/README.md`` (utils vs helpers, no overlap).
"""

import io
from http import HTTPStatus
from numbers import Integral
from typing import Any

import httpx
import numpy as np
import pybase64 as base64
from fastapi import HTTPException, Request
from PIL import Image
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.app_state import _get_diffusion_od_config

logger = init_logger(__name__)


def _get_max_edit_input_images(raw_request: Request, engine_client: Any) -> int | None:
    od_config = _get_diffusion_od_config(raw_request, engine_client)
    if od_config is None:
        # Preserve the existing compatibility behavior when the diffusion
        # config is not exposed on the serving surface.
        return None

    supports_multimodal_inputs = getattr(od_config, "supports_multimodal_inputs", None)
    if not isinstance(supports_multimodal_inputs, bool):
        # Older serving surfaces and mocked engines may expose a placeholder
        # object instead of a real diffusion config. Treat that as "unknown"
        # so existing single-image flows keep working.
        return None

    if not supports_multimodal_inputs:
        return 1

    max_input_images = getattr(od_config, "max_multimodal_image_inputs", None)
    if max_input_images is None:
        return None
    if isinstance(max_input_images, bool) or not isinstance(max_input_images, Integral):
        return None
    if max_input_images < 1:
        return None
    return int(max_input_images)


def _check_max_generated_image_size(
    app_state_args: Any,
    width: int | None,
    height: int | None,
    resolution: int | None = None,
) -> None:
    """Raise 400 if the requested image size exceeds --max-generated-image-size."""
    max_generated_image_size = getattr(app_state_args, "max_generated_image_size", None)
    # Check max_generated_image_size
    if max_generated_image_size is None:
        return
    if width is not None and height is not None:
        if width * height > max_generated_image_size:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Requested image size {width}x{height} exceeds the maximum allowed "
                f"size of {max_generated_image_size} pixels. You can reduce the requested size "
                f"or increase the server's --max-generated-image-size limit.",
            )
    elif resolution is not None:
        # When resolution is set, the output size is resolution * resolution
        if resolution * resolution > max_generated_image_size:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Requested resolution {resolution} (max {resolution}x{resolution} pixels) "
                f"exceeds the maximum allowed size of {max_generated_image_size} pixels. "
                f"You can reduce the requested size or increase the server's --max-generated-image-size limit.",
            )


def _build_hunyuan_edit_extra_args(
    *,
    bot_task: str | None,
    sys_type: str | None,
    system_prompt: str | None,
) -> dict[str, Any]:
    """Map Hunyuan /v1/images/edits form fields to DiT ``extra_args``."""
    extra_args: dict[str, Any] = {}
    effective_use_system_prompt = sys_type
    if effective_use_system_prompt is None and bot_task is not None:
        from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import resolve_sys_type

        effective_use_system_prompt = resolve_sys_type(bot_task)
    if effective_use_system_prompt is not None:
        extra_args["use_system_prompt"] = effective_use_system_prompt
    if system_prompt is not None:
        extra_args["system_prompt"] = system_prompt
    if bot_task is not None:
        extra_args["bot_task"] = bot_task
    return extra_args


def _update_if_not_none(object: Any, key: str, val: Any) -> None:
    if val is not None:
        setattr(object, key, val)


def _normalize_image(image: Any) -> Any:
    """Normalize a single image output to a PIL-compatible format."""
    if isinstance(image, Image.Image):
        return image
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Unsupported image type: {type(image)}")
    if not np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"Unsupported dtype: {image.dtype}")
    if isinstance(image, np.ndarray):
        while image.ndim > 3:
            image = image[0]
        if image.min() < 0:
            if image.min() < -1.01 or image.max() > 1.01:
                logger.warning(
                    f"Image float range [{image.min():.2f}, {image.max():.2f}] outside expected [-1, 1]. "
                    f"Clipping to [-1, 1] before normalization."
                )
            image = np.clip(image, -1.0, 1.0) * 0.5 + 0.5
        elif image.max() > 1.01:
            logger.warning(
                f"Image float range [{image.min():.2f}, {image.max():.2f}] outside expected [0, 1]. "
                f"Clipping to [0, 1] before normalization."
            )
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(image)
    return image


def _extract_images_from_result(result: Any) -> list[Any]:
    images = []
    if hasattr(result, "images") and result.images:
        images = result.images
    # Handle when generate more than one image
    if images and isinstance(images[0], np.ndarray) and images[0].shape[0] > 1 and images[0].ndim == 5:
        # Unwrap batch: (N, T, H, W, C) -> [img1, img2, ...]
        images = list(images[0])
    # Flatten nested lists (e.g., from layered models like Qwen-Image-Layered).
    # Note: This only flattens one level deep. Deeper nesting is not supported.
    flattened = []
    for img in images:
        if isinstance(img, list):
            flattened.extend(img)
        else:
            flattened.append(img)
    return [_normalize_image(img) for img in flattened]


def _generated_size_str(images: list[Any], fallback: str | None) -> str | None:
    """``"WxH"`` of the first generated image, or ``fallback`` when it is unknown."""
    size = getattr(images[0], "size", None) if images else None
    if isinstance(size, tuple) and len(size) == 2 and all(isinstance(value, Integral) for value in size):
        return f"{size[0]}x{size[1]}"
    return fallback


async def _load_input_images(
    inputs: list[str],
    *,
    normalize_rgb: bool = True,
) -> list[Image.Image]:
    """
    convert to PIL.Image.Image list
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    images: list[Image.Image] = []

    for inp in inputs:
        # 1. URL + base64
        if isinstance(inp, str) and inp.startswith("data:image"):
            try:
                _, b64_data = inp.split(",", 1)
                image_bytes = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(image_bytes))
                images.append(img)
            except Exception as e:
                raise ValueError(f"Invalid base64 image: {e}")

       
```
