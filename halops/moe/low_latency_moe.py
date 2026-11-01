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
from halops.moe.low_latency_moe_utils import (
    _interleave_up_for_pair_layout,
    _pack_router_weight,
    _prepare_shared_expert_up_weight,
    _router_gates,
)

__all__ = [
    "low_latency_moe",
    "get_workspace_for_moe",
    "get_workspace_size_for_moe",
    "transform_weights_for_moe",
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
    latent size 2048.  The routed and shared experts use block-wise FP8
    weights (128x128 blocks); ``latent_down`` may also be ``w8a8-mxfp8``.

    Weights are quantized by the caller (``group_wise_quant`` for MX
    recipes); the router and up-projection operands are then packed with
    ``transform_weights_for_moe``.  The down-projection and latent-down
    weights are quantized directly in the kernel layout and passed as-is.
    This build runs the latent stage, so ``latent_down_weight`` and
    ``latent_down_weight_scale`` are required.

    Args:
        input (Tensor): Activations, shape [T, 4096], dtype bfloat16, with
            1 <= T <= 16.
        router_weight (Tensor): Packed router weight from
            ``transform_weights_for_moe``, dtype bfloat16. Shape is
            [G * (E // tile_m), K // 64, tile_m, 64], concretely
            [64, 64, 16, 64] for T <= 8 and [32, 64, 32, 64] otherwise.
        expert_up_weights (Tensor): Fused gate/up weights in the pair layout
            returned by ``transform_weights_for_moe``, shape [512, 1024,
            2048], dtype float8_e4m3fn.
        expert_up_weights_scale (Tensor): Block scales of
            ``expert_up_weights``, shape [512, 8, 16], dtype float32.
        expert_down_weights (Tensor): Down weights, shape [512, 2048, 512],
            dtype float8_e4m3fn.
        expert_down_weights_scale (Tensor): Block scales of
            ``expert_down_weights``, shape [512, 16, 4], dtype float32.
        shared_up_weight (Tensor): Shared-expert gate/up weights in the pair
            layout, shape [1024, 4096], dtype float8_e4m3fn.
        shared_up_weight_scale (Tensor): Block scales of
            ``shared_up_weight``, shape [8, 32], dtype float32.
        shared_down_weight (Tensor): Shared-expert down weights, shape
            [4096, 512], dtype float8_e4m3fn.
        shared_down_weight_scale (Tensor): Block scales of
            ``shared_down_weight``, shape [32, 4], dtype float32.
        latent_down_weight (Tensor): Latent down-projection weight in the
            kernel layout, shape [2048, 4096], dtype float8_e4m3fn.
            Quantize it in this layout and pass it directly.
        latent_down_weight_scale (Tensor): Scales of
            ``latent_down_weight``: shape [16, 32] and dtype float32 for
            "w8a8-blockwise", or shape [2048, 128] and dtype uint8 (plain
            row-major UE8M0, from ``group_wise_quant(use_swizzle=False)``)
            for "w8a8-mxfp8".
        workspace (Tensor): 1-D uint8 workspace of
            ``get_workspace_size_for_moe(input)`` bytes from
            ``get_workspace_for_moe``.
        routed_out (Tensor): Routed-expert output written in place, shape
            [T, 2048], dtype bfloat16.
        shared_out (Tensor): Shared-expert output written in place, shape
            [T, 4096], dtype bfloat16.
        top_k (int): Experts selected per token, in [1, 10].
        num_router_gates (int, optional): Router gates merged into
            ``router_weight``; this build requires 2. Default: 2.
        scoring_func (str, optional): Router scoring, either "sigmoid" or
            "softmax". Default: "sigmoid".
        renormalize (bool, optional): Renormalize the top-k weights to sum to
            one. Default: True.
        activation (str, optional): Expert activation, either "silu" or
            "situ". Default: "silu".
        situ_beta (float, optional): SiTU gate soft cap; <= 0 disables it.
            Must be finite. Default: 4.0.
        situ_linear_beta (float, optional): SiTU up soft cap; <= 0 disables
            it. Must be finite. Default: 25.0.
        quant_mode (str or dict, optional): Quantization recipe per weight
            group. A string applies to every group; a dict may set
            "experts", "latent_down" and "shared" plus "default" for the
            rest. Recipes are cross-checked against the operands. On top of
            the all-blockwise recipe this build implements
            "latent_down=w8a8-mxfp8". Default: "w8a8-blockwise".

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
        input (Tensor): Activations, shape [T, 4096], dtype bfloat16; T
            selects the schedule.

    Returns:
        1-D uint8 workspace tensor of ``get_workspace_size_for_moe(input)``
        bytes, on ``input``'s device.
    """
    return halops_kernel.moe_get_workspace(input)


def get_workspace_size_for_moe(input: Optional[torch.Tensor] = None) -> int:
    """Return the workspace size in bytes for a token count.

    Args:
        input (Tensor, optional): Activations, shape [T, 4096], dtype
            bfloat16. Without it, the larger of the supported schedules is
            returned. Default: None.

    Returns:
        Workspace size in bytes.
    """
    return halops_kernel.moe_get_workspace_size(input)


def transform_weights_for_moe(
    router_weight: Union[torch.Tensor, Sequence[torch.Tensor]],
    expert_up_weights: torch.Tensor,
    shared_up_weight: torch.Tensor,
    num_tokens: int = 1,
    num_router_gates: int = 2,
) -> dict:
    """Pack the router and up-projection weights for ``low_latency_moe``.

    The down-projection and latent-down weights, and the scales, already follow
    the layout the kernel reads and are passed to ``low_latency_moe`` as-is.
    This repacks the operands whose layout differs:

    * ``router_weight`` -- the router's SW128 swizzle, one packed blob per
      gate.
    * ``expert_up_weights`` / ``shared_up_weight`` -- the gate/up pair
      layout (pack after quantizing: the scales stay on the raw layout).

    ``num_tokens`` selects the router schedule and must match the ``low_latency_moe``
    call.

    Args:
        router_weight (Tensor): Router weight, shape [K, E], dtype bfloat16;
            shape [G, K, E] or a sequence of G [K, E] tensors is also
            accepted, and a single [K, E] weight is reused for every gate.
        expert_up_weights (Tensor): Routed weights, shape [E, 2N, K], dtype
            float8_e4m3fn; rows [0, N) are gate and [N, 2N) up.
        shared_up_weight (Tensor): Shared-expert weights, shape [2N, H],
            dtype float8_e4m3fn, same gate/up split.
        num_tokens (int, optional): Token count of the ``low_latency_moe``
            call, which picks the router tile M. Default: 1.
        num_router_gates (int, optional): Gates to merge; this build requires
            2. Default: 2.

    Returns:
        A dict with the packed operands:

        * "router_weight" (Tensor): dtype bfloat16, shape
          [G * (E // tile_m), K // 64, tile_m, 64], where tile_m is 16 for
          num_tokens <= 8 and 32 otherwise.
        * "expert_up_weights" (Tensor): dtype float8_e4m3fn, shape
          [E, 2N, K], gate/up interleaved.
        * "shared_up_weight" (Tensor): dtype float8_e4m3fn, shape [2N, H],
          gate/up interleaved.
    """
    packed = {}
    packed["router_weight"] = _pack_router_weight(
        _router_gates(router_weight, num_router_gates), num_tokens
    )
    packed["expert_up_weights"] = _interleave_up_for_pair_layout(expert_up_weights)
    packed["shared_up_weight"] = _prepare_shared_expert_up_weight(shared_up_weight)
    return packed
