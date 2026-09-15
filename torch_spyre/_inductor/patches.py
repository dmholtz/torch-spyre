# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from functools import wraps

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, MutationLayoutSHOULDREMOVE
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.utils import InputType
from torch._inductor.virtualized import V


@contextmanager
def spyre_data_types():
    saved = torch._prims_common._computation_dtype_map
    torch._prims_common._computation_dtype_map = {
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.complex32: torch.complex32,
    }
    try:
        yield
    finally:
        torch._prims_common._computation_dtype_map = saved


@contextmanager
def enable_spyre_context(example_inputs: list[InputType]):
    """
    Context manager that sets up the complete Spyre compilation environment.

    This CM configures PyTorch Inductor to compile graphs for the Spyre device by:
      - Enabling Spyre-specific data type handling
      - Activating Spyre lowerings
      - Configuring Inductor settings optimized for Spyre
      - Setting up custom pre/post compilation passes
      - Disabling incompatible optimizations (e.g., reduction splitting, permute fusion)

    Spyre-specific decompositions are *not* installed by this CM. They are
    threaded into Inductor via ``get_decomp_fn`` (see ``_spyre_inner_compile``,
    which re-binds it to ``get_spyre_decomp_table`` in
    ``torch_spyre._inductor``), which keeps the FX graph cache key picklable
    and avoids mutating PyTorch's global decomposition registry.

    Args:
        example_inputs: List of example inputs to the graph being compiled. Used to
            set real inputs in the virtualized context for shape inference and
            optimization decisions.
    """

    from torch_spyre._inductor.lowering import enable_spyre_lowerings  # your CM

    # Ensure decorators run (custom ops/lowerings modules)
    import torch_spyre._inductor.customops  # noqa: F401
    import torch_spyre._inductor.lowering  # noqa: F401
    from torch_spyre._inductor.choices import SpyreHeuristics
    from torch_spyre._inductor.passes import (
        CustomPreGradPasses,
        CustomPrePasses,
        CustomPostPasses,
        CustomPreFusionPasses,
        CustomPostFusionPasses,
        CustomPreSchedulingPasses,
    )
    from torch_spyre._inductor.propagate_hints import recover_spyre_hints

    # joint_custom_pre_pass fires inside joint_graph_passes() *after*
    # lazy_init() has populated pass_patterns[0] but *before*
    # pass_patterns[0].apply() runs.  That is the first moment the SFDP entries
    # exist and can be wrapped.  We call _patch_sfdp_no_mask_checks() here so
    # the mask-guard extra_checks are in place before any pattern is applied.
    # _patch_sfdp_no_mask_checks() is idempotent (sentinel attribute prevents
    # double-wrapping), so repeated calls are safe.
    from torch._inductor.custom_graph_pass import get_custom_graph_passes

    existing_joint_pre = list(
        get_custom_graph_passes(torch._inductor.config.joint_custom_pre_pass)
    )

    def _ensure_sfdp_patched(_graph):
        _patch_sfdp_no_mask_checks()

    composed_joint_pre_pass = (
        [_ensure_sfdp_patched] + existing_joint_pre
        if existing_joint_pre
        else _ensure_sfdp_patched
    )

    # *) Inductor config tweaks (saved/restored)
    new_config = {
        "split_reductions": False,
        "benchmark_harness": False,
        "pre_grad_custom_pass": CustomPreGradPasses(),
        "post_grad_custom_pre_pass": CustomPrePasses(),
        "post_grad_custom_post_pass": CustomPostPasses(),
        "_pre_fusion_custom_pass": CustomPreFusionPasses(),
        "_post_fusion_custom_pass": CustomPostFusionPasses(),
        "joint_custom_pre_pass": composed_joint_pre_pass,
        # Adding this configuration in so as to avoid the optimization of turning small matmuls into non-matmuls
        # found here: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ir.py#L1580
        "unroll_reductions_threshold": 1,
        # Disable fusing of mm + permute/transpose for now.
        "permute_fusion": False,
        "allow_buffer_reuse": False,  # For now, as buffer reuse does not consider stride_map.
        "reorder_for_locality": False,  # Prevents unhinted ops from being moved into hinted regions.
    }

    from torch._inductor.ir import Loops

    # Force all operations to be realized when LoopLevel IR is initially constructed
    old_loop = Loops.has_large_inner_fn
    Loops.has_large_inner_fn = lambda self, threshold=None: True

    from torch._inductor.fx_passes import joint_graph

    origin_pass = list(joint_graph.pass_patterns)
    # disable mul_softmax_pattern and div_softmax_pattern for now
    joint_graph.pass_patterns.pop()

    old_update_scheduler = GraphLowering._update_scheduler

    _pre_scheduling_pass = CustomPreSchedulingPasses()

    def _spyre_update_scheduler(self: GraphLowering) -> None:
        # Nested compiler contexts may wrap this hook more than once. The
        # graph-mutating pre-scheduling pipeline runs once per GraphLowering.
        if not getattr(self, "_spyre_pre_scheduling_complete", False):
            # recover_spyre_hints runs here (after all post-grad FX passes including
            # decompose_auto_functionalized) rather than in CustomPostPasses.
            # decompose_auto_functionalized replaces auto_functionalized_v2 nodes
            # via make_fx retracing, which creates new FX nodes whose meta["custom"]
            # only contains hints from the innermost scope. Running recovery here
            # ensures the final FX graph (post-decomposition) gets the full hint set.
            gm = self.graph.owning_module
            if gm is not None and "__spyre_dim_hints" in gm.meta:
                recover_spyre_hints(self.graph)
            _pre_scheduling_pass(self)
            setattr(self, "_spyre_pre_scheduling_complete", True)
        old_update_scheduler(self)

    GraphLowering._update_scheduler = _spyre_update_scheduler  # type: ignore[method-assign]

    # coarse_tile.py's nested output-dim + reduction-dim tiling
    # (_propagate_tiled_reduction_op) inserts a copy-out op
    # (_insert_reduction_copy_op) that mutates a pre-loop accumulation buffer
    # (accum_full) so its updated value is visible to the NEXT outer-tile
    # iteration's copy-in. That cross-iteration read has no representation in
    # the single-pass, pre-unroll IR the scheduler's own dead_node_elimination
    # walks, so a copy-out with no other downstream reader looks dead and is
    # removed — even though it is required for correctness. Mark such ops
    # with _coarse_tile_force_live (see _insert_reduction_copy_op) and force
    # SchedulerNode.has_side_effects() to report True for them, mirroring how
    # upstream itself protects effectful FallbackKernels from the same DCE
    # pass (torch/_inductor/lowering.py, effectful op handling).
    old_scheduler_node_has_side_effects = SchedulerNode.has_side_effects

    def _spyre_scheduler_node_has_side_effects(self: SchedulerNode) -> bool:
        if getattr(self.node, "_coarse_tile_force_live", False):
            return True
        # ComputedBuffers with MutationLayoutSHOULDREMOVE write into a
        # pre-existing buffer (e.g. copy_forced dst). The scheduler's own DCE
        # doesn't know about this layout convention and marks them dead when
        # no downstream op reads the output name. Keep them live.
        if isinstance(self.node, ComputedBuffer) and isinstance(
            self.node.layout, MutationLayoutSHOULDREMOVE
        ):
            return True
        return old_scheduler_node_has_side_effects(self)

    SchedulerNode.has_side_effects = _spyre_scheduler_node_has_side_effects  # type: ignore[method-assign]

    with (
        spyre_data_types(),
        enable_spyre_lowerings(),
        V.set_real_inputs(example_inputs),
        V.set_choices_handler(SpyreHeuristics()),
        torch._inductor.config.patch(new_config),
    ):
        try:
            yield
        finally:
            joint_graph.pass_patterns[:] = origin_pass
            Loops.has_large_inner_fn = old_loop
            GraphLowering._update_scheduler = old_update_scheduler  # type: ignore[method-assign]
            SchedulerNode.has_side_effects = old_scheduler_node_has_side_effects  # type: ignore[method-assign]


OBSERVER_HOOKS_KEY = "__spyre_hooks_meta"


def _no_mask_sfdp_has_mask_add(match) -> bool:
    """Return True if any scale node (mul/div) inside the match feeds an
    add.Tensor user that lives *outside* the matched node set.

    SFDP no-mask patterns 1 and 2 describe:
        matmul → (div|mul)(scale) → softmax → matmul
    and hardcode attn_mask=None in their replacement.  When the real graph is:
        matmul → (div|mul)(scale) → add(mask) → softmax → matmul
    the add sits outside the matched node set so filter_nodes cannot detect it.
    We inspect the live .users of the matched scale node directly.
    """
    aten = torch.ops.aten
    matched = set(match.nodes)
    for node in match.nodes:
        if node.target in (aten.mul.Tensor, aten.div.Tensor):
            for user in node.users:
                if user.target == aten.add.Tensor and user not in matched:
                    return True
    return False


def _wrap_sfdp_no_mask_extra_check(original_extra_check):
    """Return a wrapped extra_check that vetoes the match when a mask add is present."""

    def wrapped(match):
        if _no_mask_sfdp_has_mask_add(match):
            return False
        return original_extra_check(match)

    return wrapped


def patch_inductor_fusions():
    import torch._inductor.fx_passes.post_grad

    # disable addmm fusion. The fusion will be undone by the decomposition that is
    # registered in torch-spyre, but the hints are lost in the process
    addmm_fusion_found = False
    for entries in torch._inductor.fx_passes.post_grad.pass_patterns[
        2
    ].patterns.values():
        for entry in entries:
            if (
                entry.extra_check
                == torch._inductor.fx_passes.post_grad.is_valid_addmm_fusion
            ):
                entry.extra_check = lambda x: False
                addmm_fusion_found = True

    assert addmm_fusion_found, (
        "Couldn't find addmm fusion. This patch needs to be reviewed."
    )

    # Wrap the extra_check of SFDP no-mask patterns 1 and 2 so they reject
    # matches where the scale node (mul/div) feeds an add(mask) that is outside
    # the matched node set.  Without this, _sfdp_pattern_2_half_inference (and
    # the pattern-1 div-scale variant) silently match masked manual-attention
    # graphs and replace them with sdpa(attn_mask=None), discarding the mask.
    #
    # Patterns 5, 6, etc. already include attn_mask in their pattern graph and
    # produce the correct sdpa(attn_mask=mask) — they must not be touched.
    #
    # lazy_init() inside joint_graph_passes() populates pass_patterns[0] the
    # first time any joint graph is compiled.  patch_inductor_fusions() is
    # called at import time, before that, so pass_patterns[0] may be empty
    # here.  The wrapping is therefore applied lazily from enable_spyre_context
    # via joint_custom_pre_pass (_ensure_sfdp_patched), which fires after
    # lazy_init() but before pass_patterns[0].apply().
    #
    # See: https://github.com/torch-spyre/torch-spyre/issues/4526

    # Install observer patch
    from torch.fx.passes.graph_transform_observer import GraphTransformObserver

    _original = GraphTransformObserver.apply_graph_pass

    @wraps(GraphTransformObserver.apply_graph_pass)
    def apply_graph_pass(self, pass_fn):
        meta = self.gm.meta.get(OBSERVER_HOOKS_KEY, {})
        self.gm.meta[OBSERVER_HOOKS_KEY] = meta
        meta["pass"] = self.passname
        meta["subsystem"] = self.subsystem
        try:
            return _original(self, pass_fn)
        finally:
            meta.pop("pass", None)
            meta.pop("subsystem", None)

    GraphTransformObserver.apply_graph_pass = apply_graph_pass


def _patch_sfdp_no_mask_checks() -> None:
    """Wrap the extra_check of SFDP no-mask patterns 1 and 2.

    Safe to call multiple times — entries that have already been wrapped are
    detected and skipped via the _spyre_sfdp_patched sentinel attribute.
    Called lazily from enable_spyre_context after lazy_init() has populated
    pass_patterns[0].
    """
    import re

    try:
        from torch._inductor.fx_passes import joint_graph as _jg
    except Exception:
        return  # Inductor internals changed — let tests catch it.

    sfdp_no_mask_re = re.compile(
        r"^_sfdp_pattern_[12](_half)?_(inference|training)$"
    )
    for entries in _jg.pass_patterns[0].patterns.values():
        for entry in entries:
            name = getattr(entry, "pattern_name", "") or ""
            if sfdp_no_mask_re.match(name) and not getattr(
                entry, "_spyre_sfdp_patched", False
            ):
                entry.extra_check = _wrap_sfdp_no_mask_extra_check(
                    entry.extra_check
                )
                entry._spyre_sfdp_patched = True
