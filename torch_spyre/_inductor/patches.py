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

import regex as re
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

    # joint_custom_pre_pass fires inside joint_graph_passes() *after*
    # lazy_init() has populated pass_patterns[0] but *before*
    # pass_patterns[0].apply() runs.  That is the first moment the SFDP entries
    # exist and can be wrapped.  We call _patch_sfdp_no_mask_checks() here so
    # the mask-guard extra_checks are in place before any pattern is applied.
    #
    # wrapped_sfdp_entries accumulates every PatternEntry wrapped during this
    # context manager invocation.  These are reverted in the finally block so
    # that Spyre's guard does not permanently infect pass_patterns[0] and
    # affect subsequent non-Spyre (e.g. CUDA/CPU) compiles in the same process.
    #
    # Any pre-existing joint_custom_pre_pass set on the config before this CM
    # is entered is preserved by composing it after _ensure_sfdp_patched.
    # The value is snapshotted here (at CM-entry time) because
    # torch._inductor.config.patch overwrites it for the duration of the CM;
    # there is no meaningful window in which a caller would set it again inside
    # the CM, so the snapshot captures exactly what the caller intended.
    # The result is always a list for a consistent CustomGraphPassType.
    from torch._inductor.custom_graph_pass import get_custom_graph_passes

    # Ensure decorators run (custom ops/lowerings modules)
    import torch_spyre._inductor.customops  # noqa: F401
    import torch_spyre._inductor.lowering  # noqa: F401
    from torch_spyre._inductor.choices import SpyreHeuristics
    from torch_spyre._inductor.lowering import enable_spyre_lowerings  # your CM
    from torch_spyre._inductor.passes import (
        CustomPostFusionPasses,
        CustomPostPasses,
        CustomPreFusionPasses,
        CustomPreGradPasses,
        CustomPrePasses,
        CustomPreSchedulingPasses,
    )
    from torch_spyre._inductor.propagate_hints import recover_spyre_hints

    wrapped_sfdp_entries: list[object] = []

    def _ensure_sfdp_patched(_graph):
        wrapped_sfdp_entries.extend(_patch_sfdp_no_mask_checks())

    existing_joint_pre = list(
        get_custom_graph_passes(torch._inductor.config.joint_custom_pre_pass)
    )
    composed_joint_pre_pass: list = [_ensure_sfdp_patched] + existing_joint_pre

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
            _unpatch_sfdp_no_mask_checks(wrapped_sfdp_entries)
            joint_graph.pass_patterns[:] = origin_pass
            Loops.has_large_inner_fn = old_loop
            GraphLowering._update_scheduler = old_update_scheduler  # type: ignore[method-assign]
            SchedulerNode.has_side_effects = old_scheduler_node_has_side_effects  # type: ignore[method-assign]


OBSERVER_HOOKS_KEY = "__spyre_hooks_meta"


# Additive-mask targets whose output is consumed by softmax — if a scale node
# inside a no-mask SFDP pattern feeds one of these *outside* the matched set,
# the graph actually has a mask and the pattern must not fire.
_ADD_TARGETS = frozenset({
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.add_.Scalar,
})

# Scale ops present in the no-mask SFDP patterns being guarded.
_SCALE_TARGETS = frozenset({
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
})


def _no_mask_sfdp_has_mask_add(match) -> bool:
    """Return True if any scale node (mul/div) inside the match feeds an
    add user that lives *outside* the matched node set.

    No-mask SFDP patterns (1–4, 11, 12, 28) describe:
        matmul → (div|mul)(scale) [→ optional-cast] → softmax → matmul
    and hardcode attn_mask=None in their replacement.  When the real graph is:
        matmul → (div|mul)(scale) → add(mask) → softmax → matmul
    the add sits outside the matched node set so filter_nodes cannot detect it.
    We inspect the live .users of the matched scale node directly.

    Both add.Tensor (tensor mask) and add.Scalar (scalar bias) are checked.
    In-place variants (add_.*) are included for completeness, though Inductor
    typically functionalises them before pattern matching.
    """
    matched = set(match.nodes)
    for node in match.nodes:
        if node.target in _SCALE_TARGETS:
            for user in node.users:
                if user.target in _ADD_TARGETS and user not in matched:
                    return True
    return False


def _wrap_sfdp_no_mask_extra_check(original_extra_check):
    """Return a wrapped extra_check that vetoes the match when a mask add is present.

    The original callable is stored on the returned function as
    ``_spyre_original_extra_check`` so that ``_unpatch_sfdp_no_mask_checks``
    can restore it when the Spyre compilation context exits.
    """

    def wrapped(match):
        if _no_mask_sfdp_has_mask_add(match):
            return False
        return original_extra_check(match)

    wrapped._spyre_original_extra_check = original_extra_check  # type: ignore[attr-defined]
    return wrapped


def _unpatch_sfdp_no_mask_checks(wrapped_entries: list[object]) -> None:
    """Restore the original extra_check on every entry wrapped in this session.

    Called from the finally block of enable_spyre_context so that Spyre's
    guard does not permanently mutate the global pass_patterns[0] and affect
    subsequent non-Spyre (e.g. CUDA/CPU) compiles in the same process.
    """
    for entry in wrapped_entries:
        original = getattr(
            getattr(entry, "extra_check", None),
            "_spyre_original_extra_check",
            None,
        )
        if original is not None:
            entry.extra_check = original  # type: ignore[attr-defined]
        # Remove the sentinel so a subsequent Spyre compile re-applies the guard.
        try:
            delattr(entry, "_spyre_sfdp_patched")  # type: ignore[attr-defined]
        except AttributeError:
            pass


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


# Regex matching the pattern_name values of all SFDP no-mask entries that need
# the mask-add guard.  Patterns 1, 2, 3, 4, 11, 12, 28 all produce
# sdpa(attn_mask=None) and share the same vulnerable topology:
#   matmul → (div|mul).Tensor(scale) [→ optional cast] → softmax → matmul
# Their registered names follow the scheme built by _get_sfdp_patterns():
#   <base_name>[_half][_bs1]_(inference|training)
# Pattern 28 uses non-contiguous inputs (gn) but batch_size=2 so no _bs1.
_SFDP_NO_MASK_RE = re.compile(
    r"^_sfdp_pattern_(?:1|2|3|4|11|12|28)(_half)?(_bs1)?_(inference|training)$"
)


def _patch_sfdp_no_mask_checks() -> list[object]:
    """Wrap the extra_check of SFDP no-mask patterns 1–4, 11, 12, and 28.

    These patterns all replace:
        matmul → (div|mul)(scale) → [optional cast] → softmax → matmul
    with sdpa(attn_mask=None).  When the real graph has an additive mask between
    the scale node and softmax the pattern fires and silently discards the mask.

    Returns the list of PatternEntry objects newly wrapped in this call.
    Already-wrapped entries are skipped via the _spyre_sfdp_patched sentinel,
    making repeated calls safe.

    Raises AssertionError if neither any new entries were wrapped nor all
    matching entries were already wrapped, which indicates that upstream
    PyTorch has renamed or removed the patterns and the guard needs review.
    Called lazily from enable_spyre_context after lazy_init() has populated
    pass_patterns[0].
    """
    try:
        from torch._inductor.fx_passes import joint_graph as _jg
    except ImportError:
        return []  # Inductor internals changed — let tests catch it.

    newly_wrapped: list[object] = []
    for entries in _jg.pass_patterns[0].patterns.values():
        for entry in entries:
            name = getattr(entry, "pattern_name", "") or ""
            if _SFDP_NO_MASK_RE.match(name) and not getattr(
                entry, "_spyre_sfdp_patched", False
            ):
                entry.extra_check = _wrap_sfdp_no_mask_extra_check(
                    entry.extra_check
                )
                entry._spyre_sfdp_patched = True
                newly_wrapped.append(entry)

    assert newly_wrapped or _all_sfdp_no_mask_already_patched(_jg), (
        "Spyre SFDP mask guard: no entries matching the no-mask pattern regex "
        "were found in pass_patterns[0].  PyTorch may have renamed or removed "
        "these patterns.  Review _patch_sfdp_no_mask_checks() against the "
        "current torch._inductor.fx_passes.fuse_attention module.  "
        "(Issue #4526)"
    )
    return newly_wrapped


def _all_sfdp_no_mask_already_patched(jg: object) -> bool:
    """Return True if every entry matching _SFDP_NO_MASK_RE is already wrapped.

    Distinguishes 'nothing to do (all wrapped)' from 'nothing matched (broken)'
    so the assertion in _patch_sfdp_no_mask_checks can tell them apart.
    """
    found_any = False
    for entries in jg.pass_patterns[0].patterns.values():  # type: ignore[attr-defined]
        for entry in entries:
            name = getattr(entry, "pattern_name", "") or ""
            if _SFDP_NO_MASK_RE.match(name):
                found_any = True
                if not getattr(entry, "_spyre_sfdp_patched", False):
                    return False
    return found_any
