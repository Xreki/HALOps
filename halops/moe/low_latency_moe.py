"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from typing import Optional, Sequence, Union

import torch

import halops.halops_kernel as halops_kernel
from halops.gemm.group_wise_quant import group_wise_quant as _group_wise_quant
from halops.moe.low_latency_moe_utils import (
    _interleave_up_for_pair_layout,
    _pack_router_weight,
    _prepare_latent_down_weight,
    _prepare_shared_expert_up_weight,
    _router_gates,
)

__all__ = [
    "low_latency_moe",
    "get_workspace_for_moe",
    "get_workspace_size_for_moe",
    "transform_weights_for_moe",
    "quantize_moe_weights",
]

_QUANT_GROUPS = ("experts", "latent_down", "shared")
_QUANT_MODES = ("w8a8-blockwise", "w8a8-mxfp8", "w4a8-mxfp4")
_DEFAULT_QUANT_MODE = "w8a8-blockwise"
# (experts, latent_down, shared) recipes this build implements; extended as
# the MXFP8 / MXFP4 kernel milestones land.
_SUPPORTED_QUANT_COMBOS = frozenset(
    {
        ("w8a8-blockwise", "w8a8-blockwise", "w8a8-blockwise"),
        ("w8a8-blockwise", "w8a8-mxfp8", "w8a8-blockwise"),
    }
)


def _infer_quant_mode(group: str, weight: torch.Tensor, scale: torch.Tensor) -> str:
    """Infer one weight group's recipe from dtype and shape metadata."""
    if weight.dim() < 1 or scale.dim() < 1:
        raise ValueError(f"{group}: weight and scale need a K dimension.")
    if weight.dtype == torch.float8_e4m3fn:
        reduction, packed = int(weight.shape[-1]), False
        if reduction % 128:
            raise ValueError(
                f"{group}: the reduction size {reduction} must be a multiple " "of 128."
            )
    elif weight.dtype == torch.uint8:
        reduction, packed = 2 * int(weight.shape[-1]), True
        if weight.shape[-1] % 32:
            raise ValueError(
                f"{group}: packed FP4 weights must hold whole 32-element "
                f"blocks, got {weight.shape[-1]} bytes per row."
            )
    else:
        raise ValueError(
            f"{group}: weight dtype must be float8_e4m3fn or uint8 (packed "
            f"FP4), got {weight.dtype}."
        )

    scale_k = int(scale.shape[-1])
    if not packed and scale.dtype == torch.float32 and scale_k == reduction // 128:
        return "w8a8-blockwise"
    if not packed and scale.dtype == torch.uint8 and scale_k == reduction // 32:
        return "w8a8-mxfp8"
    if packed and scale.dtype == torch.uint8 and scale_k == reduction // 32:
        return "w4a8-mxfp4"
    raise ValueError(
        f"{group}: unsupported weight/scale pair: weight {weight.dtype} "
        f"{tuple(weight.shape)} with scale {scale.dtype} "
        f"{tuple(scale.shape)}."
    )


def _normalize_quant_mode(quant_mode: Union[str, dict]) -> dict:
    """Expand a recipe string or a per-group mapping into every group."""
    if isinstance(quant_mode, str):
        modes = dict.fromkeys(_QUANT_GROUPS, quant_mode)
    elif isinstance(quant_mode, dict):
        unknown = set(quant_mode) - set(_QUANT_GROUPS) - {"default"}
        if unknown:
            raise ValueError(
                f"unknown quant_mode group(s) {sorted(unknown)}; expected "
                f"{list(_QUANT_GROUPS)} or 'default'."
            )
        default = quant_mode.get("default", _DEFAULT_QUANT_MODE)
        modes = {group: quant_mode.get(group, default) for group in _QUANT_GROUPS}
    else:
        raise TypeError(
            "quant_mode must be a recipe string or a {group: recipe} dict, "
            f"got {type(quant_mode).__name__}."
        )
    for group, mode in modes.items():
        if mode not in _QUANT_MODES:
            raise ValueError(
                f"{group}: unknown quant_mode {mode!r}; expected one of "
                f"{list(_QUANT_MODES)}."
            )
    return modes


def _encode_quant_mode(modes: dict) -> str:
    """Serialize the per-group recipe in the order the host parses it."""
    return ",".join(f"{group}={modes[group]}" for group in _QUANT_GROUPS)


def _resolve_quant_mode(
    quant_mode: Union[str, dict],
    expert_up_weights: torch.Tensor,
    expert_up_weights_scale: torch.Tensor,
    expert_down_weights: torch.Tensor,
    expert_down_weights_scale: torch.Tensor,
    latent_down_weight: Optional[torch.Tensor],
    latent_down_weight_scale: Optional[torch.Tensor],
    shared_up_weight: torch.Tensor,
    shared_up_weight_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_weight_scale: torch.Tensor,
) -> dict:
    """Cross-check ``quant_mode`` against the operands and the whitelist."""
    modes = _normalize_quant_mode(quant_mode)
    operands = {
        "experts": (
            (expert_up_weights, expert_up_weights_scale),
            (expert_down_weights, expert_down_weights_scale),
        ),
        "latent_down": ((latent_down_weight, latent_down_weight_scale),),
        "shared": (
            (shared_up_weight, shared_up_weight_scale),
            (shared_down_weight, shared_down_weight_scale),
        ),
    }
    for group, pairs in operands.items():
        inferred = {
            _infer_quant_mode(group, weight, scale)
            for weight, scale in pairs
            if weight is not None and scale is not None
        }
        if len(inferred) > 1:
            raise ValueError(
                f"{group}: the operands disagree on the recipe: " f"{sorted(inferred)}."
            )
        if inferred and modes[group] not in inferred:
            (found,) = inferred
            raise ValueError(
                f"{group}: quant_mode={modes[group]!r} does not match the "
                f"operands, which are {found!r}."
            )

    combo = tuple(modes[group] for group in _QUANT_GROUPS)
    if combo not in _SUPPORTED_QUANT_COMBOS:
        supported = ", ".join(
            "(" + ", ".join(f"{g}={m}" for g, m in zip(_QUANT_GROUPS, c)) + ")"
            for c in sorted(_SUPPORTED_QUANT_COMBOS)
        )
        raise NotImplementedError(
            "this MoE build does not implement quant_mode "
            f"{dict(zip(_QUANT_GROUPS, combo))}; supported combinations: "
            f"{supported}."
        )
    return modes


def low_latency_moe(
    input: torch.Tensor,
    router_weight: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_up_weights_scale: torch.Tensor,
    expert_down_weights: torch.Tensor,
    expert_down_weights_scale: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_up_weight_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_weight_scale: torch.Tensor,
    latent_down_weight: Optional[torch.Tensor],
    latent_down_weight_scale: Optional[torch.Tensor],
    workspace: torch.Tensor,
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
    top_k: int,
    num_router_gates: int = 2,
    scoring_func: str = "sigmoid",
    renormalize: bool = True,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    quant_mode: Union[str, dict] = "w8a8-blockwise",
) -> None:
    """MoE forward pass on Blackwell SM100, support latent.

    Runs the router, the latent down-projection, the routed experts' fused
    up/down projections and the shared expert for 1..16 tokens, writing
    ``routed_out`` and ``shared_out`` in place.  The shipped schedule is
    fixed to E=512 experts, hidden size 4096, intermediate size 512 and
    latent size 2048, with block-wise FP8 weights (128x128 blocks).

    Weights are block-wise quantized by the caller; the router, up-projection
    and latent-down operands must first be packed with
    ``transform_weights_for_moe``.  This build runs the latent stage, so
    ``latent_down_weight`` and ``latent_down_weight_scale`` are required.

    Args:
        input: [T, 4096] bfloat16 activations, 1 <= T <= 16.
        router_weight: Packed router weight (bfloat16) from
            ``transform_weights_for_moe``.
        expert_up_weights: [512, 1024, 2048] float8_e4m3fn fused gate/up
            weights, in the pair layout returned by
            ``transform_weights_for_moe``.
        expert_up_weights_scale: [512, 8, 16] float32 block scales of
            ``expert_up_weights``.
        expert_down_weights: [512, 2048, 512] float8_e4m3fn down weights.
        expert_down_weights_scale: [512, 16, 4] float32 block scales of
            ``expert_down_weights``.
        shared_up_weight: [1024, 4096] float8_e4m3fn shared-expert gate/up
            weights, pair layout.
        shared_up_weight_scale: [8, 32] float32 block scales of
            ``shared_up_weight``.
        shared_down_weight: [4096, 512] float8_e4m3fn shared-expert down
            weights.
        shared_down_weight_scale: [32, 4] float32 block scales of
            ``shared_down_weight``.
        latent_down_weight: [2048, 4096] float8_e4m3fn latent
            down-projection weight, transposed by
            ``transform_weights_for_moe``.
        latent_down_weight_scale: [16, 32] float32 block scales of
            ``latent_down_weight``.
        workspace: uint8 workspace from ``get_workspace_for_moe``.
        routed_out: [T, 2048] bfloat16 output of the routed experts, written
            in place.
        shared_out: [T, 4096] bfloat16 output of the shared expert, written
            in place.
        top_k: Experts selected per token, in [1, 10].
        num_router_gates: Router gates merged into ``router_weight``; this
            build requires 2.
        scoring_func: Router scoring, ``"sigmoid"`` or ``"softmax"``.
        renormalize: Renormalize the top-k weights to sum to one.
        activation: Expert activation, ``"silu"`` or ``"situ"``.
        situ_beta: SiTU gate soft cap; <= 0 disables it.  Must be finite.
        situ_linear_beta: SiTU up soft cap; <= 0 disables it.  Must be
            finite.
        quant_mode: Quantization recipe per weight group.  A string applies
            to every group; a dict may set ``"experts"``, ``"latent_down"``
            and ``"shared"`` plus ``"default"`` for the rest.  Recipes are
            cross-checked against the operands; only ``"w8a8-blockwise"``
            is implemented in this build.

    Returns:
        None. Both outputs are written in place.
    """
    modes = _resolve_quant_mode(
        quant_mode,
        expert_up_weights,
        expert_up_weights_scale,
        expert_down_weights,
        expert_down_weights_scale,
        latent_down_weight,
        latent_down_weight_scale,
        shared_up_weight,
        shared_up_weight_scale,
        shared_down_weight,
        shared_down_weight_scale,
    )
    halops_kernel.low_latency_moe(
        input,
        router_weight,
        expert_up_weights,
        expert_up_weights_scale,
        expert_down_weights,
        expert_down_weights_scale,
        shared_up_weight,
        shared_up_weight_scale,
        shared_down_weight,
        shared_down_weight_scale,
        latent_down_weight,
        latent_down_weight_scale,
        workspace,
        routed_out,
        shared_out,
        top_k,
        num_router_gates,
        scoring_func,
        renormalize,
        activation,
        situ_beta,
        situ_linear_beta,
        _encode_quant_mode(modes),
    )


def get_workspace_for_moe(input: torch.Tensor) -> torch.Tensor:
    """Allocate the uint8 workspace of ``low_latency_moe`` for ``input``'s token count.

    Allocate it once, before any CUDA-graph capture, and reuse it across
    calls and graph replays; it is returned zero-initialized.

    Args:
        input: [T, 4096] bfloat16 CUDA tensor; T selects the schedule.

    Returns:
        1-D uint8 CUDA tensor on ``input``'s device.
    """
    return halops_kernel.moe_get_workspace(input)


def get_workspace_size_for_moe(input: Optional[torch.Tensor] = None) -> int:
    """Return the workspace size in bytes for a token count.

    Args:
        input: Optional [T, 4096] bfloat16 CUDA tensor.  Without it, the
            larger of the supported schedules is returned.

    Returns:
        Workspace size in bytes.
    """
    return halops_kernel.moe_get_workspace_size(input)


def transform_weights_for_moe(
    router_weight: Union[torch.Tensor, Sequence[torch.Tensor]],
    expert_up_weights: torch.Tensor,
    shared_up_weight: torch.Tensor,
    latent_down_weight: Optional[torch.Tensor] = None,
    num_tokens: int = 1,
    num_router_gates: int = 2,
) -> dict:
    """Pack the router, up-projection and latent-down weights for ``low_latency_moe``.

    The down-projection weights and all block scales already follow torch's
    ``[out, in]`` layout and are passed to ``low_latency_moe`` as-is.  This repacks the
    operands the kernel reads differently:

    * ``router_weight`` -- the router's SW128 swizzle, one packed blob per
      gate.
    * ``expert_up_weights`` / ``shared_up_weight`` -- the gate/up pair
      layout (pack after quantizing: the scales stay on the raw layout).
    * ``latent_down_weight`` -- transposed to ``[latent, hidden]``.

    ``num_tokens`` selects the router schedule and must match the ``low_latency_moe``
    call.

    Args:
        router_weight: Router weight ``[K, E]`` bfloat16, ``[G, K, E]`` or a
            sequence of G ``[K, E]`` tensors.  A single ``[K, E]`` weight is
            reused for every gate.
        expert_up_weights: [E, 2N, K] float8_e4m3fn routed weights; rows
            [0, N) are gate and [N, 2N) up.
        shared_up_weight: [2N, H] float8_e4m3fn shared-expert weights, same
            gate/up split.
        latent_down_weight: Optional [H, L] float8_e4m3fn latent
            down-projection weight.
        num_tokens: Token count of the ``low_latency_moe`` call, which picks the router
            tile M.
        num_router_gates: Gates to merge; this build requires 2.

    Returns:
        A dict with keys ``"router_weight"``, ``"expert_up_weights"``,
        ``"shared_up_weight"`` and, when given, ``"latent_down_weight"``.
    """
    packed = {}
    packed["router_weight"] = _pack_router_weight(
        _router_gates(router_weight, num_router_gates), num_tokens
    )
    packed["expert_up_weights"] = _interleave_up_for_pair_layout(expert_up_weights)
    packed["shared_up_weight"] = _prepare_shared_expert_up_weight(shared_up_weight)
    if latent_down_weight is not None:
        packed["latent_down_weight"] = _prepare_latent_down_weight(latent_down_weight)
    return packed


def _quantize_blockwise(w: torch.Tensor, block: int = 128):
    """128x128 block-wise FP8 with float32 scales; shape-preserving."""
    *lead, rows, cols = w.shape
    if rows % block or cols % block:
        raise ValueError(
            f"block-wise quantization needs {block}-aligned rows and cols, "
            f"got {tuple(w.shape)}."
        )
    wf = w.float().reshape(*lead, rows // block, block, cols // block, block)
    amax = wf.abs().amax(dim=(-3, -1)).clamp(min=1e-12)
    scales = amax / 448.0
    q = (wf / scales[..., None, :, None]).clamp(-448.0, 448.0)
    return q.reshape(*lead, rows, cols).to(torch.float8_e4m3fn), scales


def _quantize_mxfp8(w: torch.Tensor):
    """OCP MXFP8 (1x32, UE8M0 SfAtom scales); flattens leading dims."""
    flat = w.reshape(-1, w.shape[-1]).contiguous()
    q, scales = _group_wise_quant(flat, mode="w8a8-mxfp8")
    return (
        q.reshape(w.shape),
        scales.reshape(*w.shape[:-1], w.shape[-1] // 32),
    )


def _quantize_expert_up(weight: torch.Tensor, mode: str):
    if mode == "w8a8-blockwise":
        q, scale = _quantize_blockwise(weight)
        return _interleave_up_for_pair_layout(q), scale
    if mode == "w8a8-mxfp8":
        return _quantize_mxfp8(_interleave_up_for_pair_layout(weight))
    raise NotImplementedError(f"quantize_moe_weights does not support {mode!r}.")


def _quantize_shared_up(weight: torch.Tensor, mode: str):
    if mode == "w8a8-blockwise":
        q, scale = _quantize_blockwise(weight)
        return _prepare_shared_expert_up_weight(q), scale
    if mode == "w8a8-mxfp8":
        return _quantize_mxfp8(_prepare_shared_expert_up_weight(weight))
    raise NotImplementedError(f"quantize_moe_weights does not support {mode!r}.")


def _quantize_plain(weight: torch.Tensor, mode: str):
    if mode == "w8a8-blockwise":
        return _quantize_blockwise(weight)
    if mode == "w8a8-mxfp8":
        return _quantize_mxfp8(weight)
    raise NotImplementedError(f"quantize_moe_weights does not support {mode!r}.")


def _quantize_latent(weight: torch.Tensor, mode: str):
    if mode == "w8a8-blockwise":
        q, scale = _quantize_blockwise(weight)
        return q.transpose(0, 1).contiguous(), scale.transpose(0, 1).contiguous()
    if mode == "w8a8-mxfp8":
        return _quantize_mxfp8(weight.transpose(0, 1).contiguous())
    raise NotImplementedError(f"quantize_moe_weights does not support {mode!r}.")


def quantize_moe_weights(
    router_weight: Union[torch.Tensor, Sequence[torch.Tensor]],
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    latent_down_weight: Optional[torch.Tensor] = None,
    num_tokens: int = 1,
    num_router_gates: int = 2,
    quant_mode: Union[str, dict] = "w8a8-blockwise",
) -> dict:
    """Quantize and pack every MoE weight for ``latent_moe``.

    The weights are given in torch layout (bfloat16) and returned already
    quantized and in the orientation/layout the kernel reads: the up weights
    in the gate/up pair layout, the latent weight transposed to
    ``[latent, hidden]``, and the scales oriented for the selected recipe.

    For MX recipes the latent weight is transposed before quantizing: a 1x32
    scale grid cannot be transposed like a 128x128 one.

    Args:
        router_weight: Router weight ``[K, E]`` bfloat16, ``[G, K, E]`` or a
            sequence of G ``[K, E]`` tensors.
        expert_up_weights: [E, 2N, K] bfloat16 routed weights; rows [0, N)
            are gate and [N, 2N) up.
        expert_down_weights: [E, K, N] bfloat16 routed down weights.
        shared_up_weight: [2N, H] bfloat16 shared-expert weights.
        shared_down_weight: [H, N] bfloat16 shared-expert down weights.
        latent_down_weight: Optional [H, L] bfloat16 latent down weight.
        num_tokens: Token count of the ``latent_moe`` call, which picks the
            router tile M.
        num_router_gates: Gates to merge; this build requires 2.
        quant_mode: Recipe per weight group, as for ``latent_moe``.  Only
            ``"w8a8-blockwise"`` and ``"w8a8-mxfp8"`` are implemented.

    Returns:
        A dict with the ``latent_moe`` operand names for the router, both
        expert weights, both shared weights and, when given, the latent
        weight, each paired with its scale.
    """
    modes = _normalize_quant_mode(quant_mode)
    packed = {
        "router_weight": _pack_router_weight(
            _router_gates(router_weight, num_router_gates), num_tokens
        )
    }
    packed["expert_up_weights"], packed["expert_up_weights_scale"] = (
        _quantize_expert_up(expert_up_weights, modes["experts"])
    )
    packed["expert_down_weights"], packed["expert_down_weights_scale"] = (
        _quantize_plain(expert_down_weights, modes["experts"])
    )
    packed["shared_up_weight"], packed["shared_up_weight_scale"] = _quantize_shared_up(
        shared_up_weight, modes["shared"]
    )
    packed["shared_down_weight"], packed["shared_down_weight_scale"] = _quantize_plain(
        shared_down_weight, modes["shared"]
    )
    if latent_down_weight is not None:
        packed["latent_down_weight"], packed["latent_down_weight_scale"] = (
            _quantize_latent(latent_down_weight, modes["latent_down"])
        )
    return packed
