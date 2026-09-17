"""Runtime-only integration for DriveVA/Wan token press experiments.

The adapter deliberately monkey-patches only the active pipeline instance.  It
does not modify the clean DriveVA source tree.  ``VIDEO_INPUT`` is inserted
around ``dit.patchify`` and ``SELF_ATTN_KV`` is inserted at the existing Wan
post-RoPE attention module.
"""

from __future__ import annotations

import types

import torch

from ..core.context import TokenContext
from ..core.domain import build_domain
from ..core.layout import build_driveva_layout
from ..core.plan import validate_protocol
from ..core.persistence import (
    CrossLayerSelectionStore,
    HiddenSequencePersistenceController,
)
from ..core.runtime import InjectionPoint
from .wan_attention import canonicalize_wan_qkv, restore_wan_qkv


class _DriveVAPreDiTController:
    """Shorten the complete residual sequence before DiT block 0."""

    def __init__(self, adapter, runtime, model_name: str):
        self.adapter = adapter
        self.runtime = runtime
        self.model_name = str(model_name)
        self._keep = None
        self._original_length = 0

    @staticmethod
    def _gather(tensor, keep):
        if tensor.shape[0] != keep.shape[0]:
            raise ValueError("batch dimension differs from pre-DiT selection")
        view = keep.view(keep.shape[0], keep.shape[1], *((1,) * (tensor.ndim - 2)))
        return torch.gather(
            tensor,
            1,
            view.expand(keep.shape[0], keep.shape[1], *tensor.shape[2:]),
        )

    def begin_forward(self, x, freqs, t_mod, *, num_blocks: int):
        if self._keep is not None:
            raise RuntimeError("pre-DiT controller was reused before restoring its prior call")
        layout = self.runtime.layout
        if layout is None or int(layout.total_length) != int(x.shape[1]):
            raise RuntimeError(
                "pre-DiT selection requires an exact full-sequence TokenLayout"
            )
        configured, resolved, overridden, domain = self.adapter._resolve_domain_for_device(
            self.runtime, layout, x.device
        )
        metadata = self.adapter._sample_metadata(self.runtime)
        metadata.update(
            self.adapter._domain_metadata(configured, resolved, domain, overridden)
        )
        metadata.update(
            {
                "injection_point": InjectionPoint.BLOCK_INPUT.value,
                "pre_dit": True,
                "model_name": self.model_name,
            }
        )
        scene_tokens = metadata.get("scene_tokens")
        if scene_tokens is None or len(scene_tokens) != x.shape[0]:
            metadata["scene_tokens"] = [self.runtime.current_scene or ""] * x.shape[0]
        ctx = self.adapter.create_context(
            x,
            layout,
            domain,
            scene_token=self.runtime.current_scene or "",
            frame_token=getattr(self.runtime.current_sample, "frame_token", None),
            log_id=getattr(self.runtime.current_sample, "log_id", ""),
            timestamp=getattr(self.runtime.current_sample, "timestamp", None),
            diffusion_rank=self.adapter._sample_value(
                self.runtime, "diffusion_rank", None
            ),
            layer_idx=-1,
            metadata=metadata,
        )
        self.runtime.current_context = ctx
        result = self.runtime.execute_press(ctx)
        mapping = result.mapping
        keep = None if mapping is None else mapping.output_to_input
        if not torch.is_tensor(keep) or keep.ndim != 2:
            raise RuntimeError("pre-DiT operator must return a rectangular token mapping")
        keep = keep.to(x.device).long()
        if result.output.shape[:2] != keep.shape:
            raise RuntimeError("pre-DiT output and mapping lengths disagree")

        self._keep = keep
        self._original_length = int(x.shape[1])
        if freqs.ndim == 3:
            freqs = freqs.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        freqs = self._gather(freqs, keep)
        if t_mod.ndim == 4:
            t_mod = self._gather(t_mod, keep)
        elif t_mod.ndim != 3:
            raise ValueError(f"unsupported t_mod rank for pre-DiT pruning: {t_mod.ndim}")
        result.metadata.update(
            {
                "selection_source_layer": -1,
                "selection_applied_layer": -1,
                "hidden_sequence_length_before": self._original_length,
                "hidden_sequence_length_after": int(keep.shape[1]),
                "hidden_sequence_ratio": float(keep.shape[1] / self._original_length),
                "hidden_sequence_first_compressed_layer": 0,
                "hidden_sequence_last_compressed_layer": int(num_blocks) - 1,
                "hidden_sequence_compressed_layer_count": int(num_blocks),
            }
        )
        return result.output, freqs, t_mod

    def finish_forward(self, x):
        if self._keep is None:
            return x
        restored = x.new_zeros((x.shape[0], self._original_length, x.shape[2]))
        restored.scatter_(1, self._keep.unsqueeze(-1).expand_as(x), x)
        self._keep = None
        self._original_length = 0
        return restored


class DriveVAAdapter:
    def build_layout(self, f, h, w, num_cond_latents, traj_len, traj_prefix_len):
        return build_driveva_layout(f, h, w, num_cond_latents, traj_len, traj_prefix_len)

    def create_context(
        self,
        tokens: torch.Tensor,
        layout,
        domain="last_history",
        *,
        scene_token: str = "",
        frame_token: str | None = None,
        log_id: str = "",
        timestamp: int | None = None,
        timestep: int | None = None,
        diffusion_rank: int | None = None,
        layer_idx: int | None = None,
        q=None,
        k=None,
        v=None,
        metadata: dict | None = None,
    ) -> TokenContext:
        domain_obj = domain if hasattr(domain, "candidate_indices") else build_domain(domain, layout, tokens.device)
        return TokenContext(
            tokens=tokens,
            layout=layout,
            domain=domain_obj,
            scene_token=scene_token,
            frame_token=frame_token,
            log_id=log_id,
            timestamp=timestamp,
            timestep=timestep,
            diffusion_rank=diffusion_rank,
            layer_idx=layer_idx,
            q=q,
            k=k,
            v=v,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _sample_metadata(runtime) -> dict:
        sample = runtime.current_sample
        metadata = getattr(sample, "metadata", {}) if sample is not None else {}
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        # Keep the scene identity inside every runtime event as well as in the
        # outer evaluator record.  This makes cross-scene leakage detectable
        # even when events are inspected independently of the CSV.
        if getattr(runtime, "current_scene", None) is not None:
            metadata.setdefault("scene_token", str(runtime.current_scene))
        return metadata

    @staticmethod
    def _sample_value(runtime, name: str, default=None):
        if name == "diffusion_rank":
            current_rank = getattr(runtime, "current_diffusion_rank", None)
            if current_rank is not None:
                return current_rank
        sample = runtime.current_sample
        value = getattr(sample, name, default) if sample is not None else default
        if value is None:
            value = getattr(runtime, name, default)
        return value

    def _resolve_domain_for_device(self, runtime, layout, device):
        configured, resolved, overridden = runtime.resolve_domain_spec()
        if isinstance(resolved, dict):
            resolved = resolved.get("name", "last_history")
        if hasattr(resolved, "candidate_indices"):
            domain = resolved
            if domain.total_length != layout.total_length:
                raise ValueError("configured TokenDomain length does not match the active layout")
            if domain.candidate_indices.device != torch.device(device):
                domain = build_domain(domain.name, layout, device)
        else:
            domain = build_domain(resolved, layout, device)
        return configured, resolved, overridden, domain

    @staticmethod
    def _domain_metadata(configured, resolved, domain, overridden):
        def label(value):
            if isinstance(value, dict):
                value = value.get("name", "last_history")
            return str(getattr(value, "name", value))

        candidate = domain.candidate_indices
        return {
            "configured_domain": label(configured),
            "resolved_domain": label(resolved),
            "domain_override": bool(overridden),
            "candidate_start": int(candidate.min().item()) if candidate.numel() else None,
            "candidate_end": int(candidate.max().item()) + 1 if candidate.numel() else None,
            "n_candidate": domain.n_candidate,
        }

    def apply_video_input(
        self,
        runtime,
        *,
        video_tokens: torch.Tensor,
        f: int,
        h: int,
        w: int,
        num_cond_latents: int,
        traj_len: int = 0,
        traj_prefix_len: int = 0,
    ) -> torch.Tensor:
        """Apply a causal press before trajectory tokens are concatenated."""

        point = InjectionPoint.parse(runtime.press.injection_point)
        if point is not InjectionPoint.VIDEO_INPUT:
            return video_tokens
        if video_tokens.ndim != 3 or video_tokens.shape[1] != int(f) * int(h) * int(w):
            raise ValueError("VIDEO_INPUT expects [B,F*H*W,C] video tokens")
        full_layout = self.build_layout(
            int(f), int(h), int(w), int(num_cond_latents), int(traj_len), int(traj_prefix_len)
        )
        video_layout = self.build_layout(int(f), int(h), int(w), int(num_cond_latents), 0, 0)
        runtime.layout = full_layout
        configured, resolved, overridden, domain = self._resolve_domain_for_device(
            runtime, video_layout, video_tokens.device
        )
        metadata = self._sample_metadata(runtime)
        metadata.update(self._domain_metadata(configured, resolved, domain, overridden))
        metadata.update({"injection_point": point.value, "video_only_layout": True})
        scene_tokens = metadata.get("scene_tokens")
        if scene_tokens is None or len(scene_tokens) != video_tokens.shape[0]:
            metadata["scene_tokens"] = [runtime.current_scene or ""] * video_tokens.shape[0]
        ctx = self.create_context(
            video_tokens,
            video_layout,
            domain,
            scene_token=runtime.current_scene or "",
            frame_token=getattr(runtime.current_sample, "frame_token", None),
            log_id=getattr(runtime.current_sample, "log_id", ""),
            timestamp=getattr(runtime.current_sample, "timestamp", None),
            diffusion_rank=self._sample_value(runtime, "diffusion_rank", None),
            metadata=metadata,
        )
        runtime.current_context = ctx
        result = runtime.execute_press(ctx)
        if result.output.shape != video_tokens.shape:
            raise RuntimeError("VIDEO_INPUT V1 requires sequence-length-preserving output")
        result.metadata.update(self._domain_metadata(configured, resolved, domain, overridden))
        return result.output

    def install_hooks(self, pipe, runtime) -> None:
        if runtime.press is None:
            return
        validate_protocol(runtime.press, runtime.mode)
        point = InjectionPoint.parse(runtime.press.injection_point)
        if getattr(runtime.press, "name", "") in {"noop", "none", "full"}:
            runtime._driveva_hooks = []
            runtime._driveva_video_hooks = []
            return
        if point is InjectionPoint.VIDEO_INPUT:
            self._install_video_input_hooks(pipe, runtime)
            return
        if point is InjectionPoint.BLOCK_INPUT:
            self._install_block_input_hooks(pipe, runtime)
            return
        if point is InjectionPoint.SELF_ATTN_KV:
            self._install_kv_hooks(pipe, runtime)
            return
        raise NotImplementedError(f"{point.value} is declared but not implemented")

    def _install_block_input_hooks(self, pipe, runtime) -> None:
        hooks = []
        for model_name in ("dit", "dit2"):
            model = getattr(pipe, model_name, None)
            if model is None or not hasattr(model, "blocks"):
                continue
            attribute = "_tokenpress_pre_dit_controller"
            had_controller = hasattr(model, attribute)
            previous_controller = getattr(model, attribute, None)
            setattr(model, attribute, _DriveVAPreDiTController(self, runtime, model_name))
            hooks.append((model, attribute, had_controller, previous_controller))
        if not hooks:
            raise RuntimeError("BLOCK_INPUT hook could not find pipe.dit or pipe.dit2")
        runtime._driveva_pre_dit_hooks = hooks
        runtime._driveva_hooks = []
        runtime._driveva_video_hooks = []
        runtime._driveva_hidden_sequence_hooks = []

        original_model_fn = getattr(pipe, "model_fn", None)
        if original_model_fn is not None:
            def wrapped_model_fn(*args, **kwargs):
                old_state = getattr(runtime, "_model_call_state", None)
                timestep = kwargs.get("timestep")
                if torch.is_tensor(timestep) and timestep.numel():
                    runtime.current_diffusion_rank = int(timestep.reshape(-1)[0].item())
                    runtime.current_timestep = runtime.current_diffusion_rank
                latents = kwargs.get("latents")
                longcat = kwargs.get("longcat_latents")
                traj = kwargs.get("traj_tokens")
                dit = kwargs.get("dit")
                if torch.is_tensor(latents) and latents.ndim == 5 and dit is not None:
                    patch = tuple(int(value) for value in getattr(dit, "patch_size", (1, 2, 2)))
                    if len(patch) != 3 or any(int(value) <= 0 for value in patch):
                        raise RuntimeError(f"invalid DriveVA patch_size={patch}")
                    num_cond = (
                        int(longcat.shape[2])
                        if torch.is_tensor(longcat) and longcat.ndim >= 3
                        else 0
                    )
                    traj_len = (
                        int(traj.shape[1])
                        if torch.is_tensor(traj) and traj.ndim >= 2
                        else 0
                    )
                    runtime.layout = self.build_layout(
                        int(latents.shape[2]) // patch[0],
                        int(latents.shape[3]) // patch[1],
                        int(latents.shape[4]) // patch[2],
                        num_cond,
                        traj_len,
                        int(kwargs.get("traj_prefix_len", 0) or 0),
                    )
                runtime._model_call_state = {}
                try:
                    return original_model_fn(*args, **kwargs)
                finally:
                    runtime._model_call_state = old_state

            pipe.model_fn = wrapped_model_fn
            runtime._driveva_model_fn_hook = (pipe, original_model_fn)

    def _install_video_input_hooks(self, pipe, runtime) -> None:
        hooks = []
        models_found = 0
        for model_name in ("dit", "dit2"):
            model = getattr(pipe, model_name, None)
            if model is None or not hasattr(model, "patchify"):
                continue
            models_found += 1
            original_patchify = model.patchify

            def wrapped_patchify(_model, x, *, _original=original_patchify):
                patched = _original(x)
                if runtime.press is None:
                    return patched
                if not torch.is_tensor(patched) or patched.ndim != 5:
                    raise RuntimeError("DriveVA VIDEO_INPUT hook expects patchify() -> [B,C,F,H,W]")
                batch, channels, frames, height, width = patched.shape
                probe_capture = getattr(runtime, "_probe_capture", None)
                if probe_capture is not None:
                    video_tokens = patched.permute(0, 2, 3, 4, 1).reshape(
                        batch, frames * height * width, channels
                    )
                    if video_tokens.requires_grad:
                        video_tokens.retain_grad()
                    probe_capture["video_tokens"] = video_tokens
                    probe_capture["video_shape"] = (frames, height, width, channels)
                if bool(getattr(runtime, "_probe_passthrough", False)):
                    return patched
                state = getattr(runtime, "_model_call_state", {}) or {}
                num_cond = state.get("num_cond_latents")
                if num_cond is None and runtime.layout is not None:
                    num_cond = runtime.layout.num_cond_latents
                num_cond = int(num_cond or 0)
                traj_len = int(state.get("traj_len", 0) or 0)
                traj_prefix_len = int(state.get("traj_prefix_len", 0) or 0)
                video_tokens = patched.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)
                compressed = runtime.apply_video_input(
                    video_tokens=video_tokens,
                    f=frames,
                    h=height,
                    w=width,
                    num_cond_latents=num_cond,
                    traj_len=traj_len,
                    traj_prefix_len=traj_prefix_len,
                )
                return compressed.reshape(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()

            model.patchify = types.MethodType(wrapped_patchify, model)
            hooks.append((model, original_patchify))

        original_model_fn = getattr(pipe, "model_fn", None)
        if original_model_fn is not None:
            def wrapped_model_fn(*args, **kwargs):
                old_state = getattr(runtime, "_model_call_state", None)
                timestep = kwargs.get("timestep")
                if torch.is_tensor(timestep) and timestep.numel():
                    runtime.current_diffusion_rank = int(timestep.reshape(-1)[0].item())
                    runtime.current_timestep = runtime.current_diffusion_rank
                longcat = kwargs.get("longcat_latents")
                traj = kwargs.get("traj_tokens")
                latents = kwargs.get("latents")
                dit = kwargs.get("dit")
                if torch.is_tensor(latents) and latents.ndim == 5 and dit is not None:
                    patch_size = tuple(int(x) for x in getattr(dit, "patch_size", (1, 2, 2)))
                    if len(patch_size) != 3 or any(x <= 0 for x in patch_size):
                        raise RuntimeError(f"invalid DriveVA patch_size={patch_size}")
                    num_cond = int(longcat.shape[2]) if torch.is_tensor(longcat) and longcat.ndim >= 3 else 0
                    traj_len = int(traj.shape[1]) if torch.is_tensor(traj) and traj.ndim >= 2 else 0
                    traj_prefix_len = int(kwargs.get("traj_prefix_len", 0) or 0)
                    runtime.layout = self.build_layout(
                        int(latents.shape[2]) // patch_size[0],
                        int(latents.shape[3]) // patch_size[1],
                        int(latents.shape[4]) // patch_size[2],
                        num_cond,
                        traj_len,
                        traj_prefix_len,
                    )
                state = {
                    "num_cond_latents": int(longcat.shape[2]) if torch.is_tensor(longcat) and longcat.ndim >= 3 else None,
                    "traj_len": int(traj.shape[1]) if torch.is_tensor(traj) and traj.ndim >= 2 else 0,
                    "traj_prefix_len": int(kwargs.get("traj_prefix_len", 0) or 0),
                }
                runtime._model_call_state = state
                try:
                    return original_model_fn(*args, **kwargs)
                finally:
                    runtime._model_call_state = old_state

            pipe.model_fn = wrapped_model_fn
            runtime._driveva_model_fn_hook = (pipe, original_model_fn)
        if not models_found:
            raise RuntimeError("VIDEO_INPUT hook could not find pipe.dit.patchify or pipe.dit2.patchify")
        runtime._driveva_video_hooks = hooks
        runtime._driveva_hooks = []

    def _install_kv_hooks(self, pipe, runtime) -> None:
        hooks = []
        persistence = getattr(runtime.press, "cross_layer_persistence", None)
        persistence_enabled = bool(getattr(persistence, "enabled", False))
        hidden_sequence_enabled = bool(
            persistence_enabled
            and getattr(persistence, "mode", "kv_only") == "hidden_sequence"
        )
        if persistence_enabled:
            runtime._cross_layer_selection_store = CrossLayerSelectionStore()
        runtime._driveva_hidden_sequence_hooks = []
        for model_name in ("dit", "dit2"):
            model = getattr(pipe, model_name, None)
            if model is None or not hasattr(model, "blocks"):
                continue
            if hidden_sequence_enabled:
                attribute = "_tokenpress_hidden_sequence_controller"
                had_controller = hasattr(model, attribute)
                previous_controller = getattr(model, attribute, None)
                setattr(
                    model,
                    attribute,
                    HiddenSequencePersistenceController(
                        runtime,
                        model_name,
                        persistence,
                    ),
                )
                runtime._driveva_hidden_sequence_hooks.append(
                    (model, attribute, had_controller, previous_controller)
                )
            for layer_idx, block in enumerate(model.blocks):
                attention = getattr(block, "self_attn", None)
                if attention is None or not hasattr(attention, "attn"):
                    continue

                def hook(
                    q,
                    k,
                    v,
                    *,
                    layer_idx=None,
                    _layer_idx=layer_idx,
                    _attention=attention,
                    _model_name=model_name,
                    _model=model,
                ):
                    if runtime.press is None or runtime.layout is None:
                        raise RuntimeError("SELF_ATTN_KV hook requires an active layout and press")
                    scorer = getattr(runtime.press, "scorer", None)
                    requested_layer = getattr(scorer, "layer", None)
                    effective_layer = _layer_idx if layer_idx is None else layer_idx
                    reuse_persistent_selection = False
                    observe_only = False
                    if requested_layer is not None:
                        source_layer = int(requested_layer)
                        current_layer = int(effective_layer)
                        if current_layer < source_layer:
                            observation_layers = getattr(
                                scorer, "observation_layers", lambda: ()
                            )()
                            if current_layer not in observation_layers:
                                return q, k, v
                            observe_only = True
                        if current_layer > source_layer:
                            if not persistence_enabled or not persistence.includes(
                                source_layer, current_layer
                            ):
                                return q, k, v
                            if hidden_sequence_enabled:
                                # The residual stream is already physically
                                # shorter after the source block, so downstream
                                # attention needs no second K/V gather.
                                return q, k, v
                            reuse_persistent_selection = True
                    num_heads = int(getattr(_attention, "num_heads", 0))
                    q_c, q_flat = canonicalize_wan_qkv(q, num_heads)
                    k_c, _ = canonicalize_wan_qkv(k, num_heads)
                    v_c, _ = canonicalize_wan_qkv(v, num_heads)
                    dummy_tokens = q_c.transpose(1, 2).reshape(q_c.shape[0], q_c.shape[2], -1)
                    context_tokens = dummy_tokens
                    if bool(getattr(scorer, "uses_pre_block_hidden", False)):
                        pre_block = getattr(_model, "_tokenpress_pre_block_hidden", None)
                        pre_block_layer = getattr(_model, "_tokenpress_pre_block_layer", None)
                        if pre_block is None or int(pre_block_layer) != int(effective_layer):
                            raise RuntimeError(
                                "learned selector requested pre-block hidden state, but the "
                                f"model did not expose layer {effective_layer}"
                            )
                        if pre_block.shape[:2] != dummy_tokens.shape[:2]:
                            raise RuntimeError(
                                "pre-block hidden shape does not match attention sequence: "
                                f"{tuple(pre_block.shape)} vs {tuple(dummy_tokens.shape)}"
                            )
                        context_tokens = pre_block
                    configured, resolved, overridden, domain = self._resolve_domain_for_device(
                        runtime, runtime.layout, context_tokens.device
                    )
                    metadata = self._sample_metadata(runtime)
                    metadata.update(self._domain_metadata(configured, resolved, domain, overridden))
                    metadata.update(
                        {
                            "injection_point": InjectionPoint.SELF_ATTN_KV.value,
                            "post_rope": True,
                            "model_name": _model_name,
                        }
                    )
                    context = self.create_context(
                        context_tokens,
                        runtime.layout,
                        domain,
                        scene_token=runtime.current_scene or "",
                        frame_token=getattr(runtime.current_sample, "frame_token", None),
                        log_id=getattr(runtime.current_sample, "log_id", ""),
                        timestamp=getattr(runtime.current_sample, "timestamp", None),
                        q=q_c,
                        k=k_c,
                        v=v_c,
                        layer_idx=effective_layer,
                        diffusion_rank=getattr(runtime, "current_diffusion_rank", None)
                        if getattr(runtime, "current_diffusion_rank", None) is not None
                        else getattr(runtime.current_sample, "diffusion_rank", None),
                        metadata=metadata,
                    )
                    runtime.current_context = context
                    if observe_only:
                        observe = getattr(scorer, "observe", None)
                        if observe is None:
                            raise RuntimeError(
                                "scorer declared observation layers without observe(ctx)"
                            )
                        observe(context)
                        return q, k, v
                    if reuse_persistent_selection:
                        record = runtime._cross_layer_selection_store.recall(
                            context,
                            _model_name,
                            int(requested_layer),
                        )
                        result = runtime.execute_persistent_selection(
                            context,
                            record.selection,
                            source_layer=record.source_layer,
                        )
                    else:
                        result = runtime.execute_press(context)
                        persists = bool(
                            persistence is not None
                            and persistence.persists_beyond(int(effective_layer))
                        )
                        if persists:
                            if result.selection is None:
                                raise RuntimeError(
                                    "cross-layer persistence requires a press selection"
                                )
                            runtime._cross_layer_selection_store.remember(
                                context,
                                _model_name,
                                int(effective_layer),
                                result.selection,
                            )
                            result.metadata.update(
                                {
                                    "cross_layer_persistent": True,
                                    "cross_layer_persistence_mode": getattr(
                                        persistence, "mode", "kv_only"
                                    ),
                                    "persistent_selection_reused": False,
                                    "selection_source_layer": int(effective_layer),
                                    "selection_applied_layer": int(effective_layer),
                                }
                            )
                        elif persistence is not None:
                            # Cross-layer persistence is OFF for this press, so the
                            # source-layer selection is applied exactly once and
                            # nothing downstream reuses it.  Record that explicitly
                            # instead of leaving the run indistinguishable from a
                            # persistent one (audit 2026-09-12, BUG-13).
                            result.metadata.update(
                                {
                                    "cross_layer_persistent": False,
                                    "cross_layer_persistence_mode": "one_shot",
                                    "cross_layer_persistence_configured": bool(
                                        persistence_enabled
                                    ),
                                    "persistent_selection_reused": False,
                                    "selection_source_layer": int(effective_layer),
                                    "selection_applied_layer": int(effective_layer),
                                }
                            )
                    # Some physical presses (notably SimilarityMerge) emit
                    # their own metadata instead of going through
                    # ScorerPress.  Keep the resolved candidate range on the
                    # result in both cases so selection-position audits have
                    # one uniform event schema.
                    result.metadata.update(self._domain_metadata(configured, resolved, domain, overridden))
                    aux = result.aux
                    if not aux or "k" not in aux or "v" not in aux:
                        raise RuntimeError(
                            f"{type(runtime.press).__name__} did not emit compressed K/V at SELF_ATTN_KV"
                        )
                    return q, restore_wan_qkv(aux["k"], q_flat), restore_wan_qkv(aux["v"], q_flat)

                attention_module = attention.attn
                original_forward = attention_module.forward

                def wrapped_attention(_module, q, k, v, *, _hook=hook, _original=original_forward):
                    q, k, v = _hook(q, k, v)
                    return _original(q, k, v)

                attention_module.forward = types.MethodType(wrapped_attention, attention_module)
                had_public_hook = hasattr(attention, "tokenpress_hook")
                previous_public_hook = getattr(attention, "tokenpress_hook", None)
                attention.tokenpress_hook = hook
                hooks.append((attention, attention_module, original_forward, had_public_hook, previous_public_hook))
        # SELF_ATTN_KV is reached after the video and trajectory tokens have
        # been concatenated.  Establish the full sequence layout at the
        # official model_fn boundary so the KV hook does not depend on a
        # VIDEO_INPUT hook having run first.
        original_model_fn = getattr(pipe, "model_fn", None)
        if original_model_fn is not None:
            def wrapped_model_fn(*args, **kwargs):
                old_state = getattr(runtime, "_model_call_state", None)
                timestep = kwargs.get("timestep")
                if torch.is_tensor(timestep) and timestep.numel():
                    runtime.current_diffusion_rank = int(timestep.reshape(-1)[0].item())
                    runtime.current_timestep = runtime.current_diffusion_rank
                longcat = kwargs.get("longcat_latents")
                traj = kwargs.get("traj_tokens")
                latents = kwargs.get("latents")
                dit = kwargs.get("dit")
                if torch.is_tensor(latents) and latents.ndim == 5 and dit is not None:
                    patch_size = tuple(int(x) for x in getattr(dit, "patch_size", (1, 2, 2)))
                    if len(patch_size) != 3 or any(x <= 0 for x in patch_size):
                        raise RuntimeError(f"invalid DriveVA patch_size={patch_size}")
                    num_cond = int(longcat.shape[2]) if torch.is_tensor(longcat) and longcat.ndim >= 3 else 0
                    traj_len = int(traj.shape[1]) if torch.is_tensor(traj) and traj.ndim >= 2 else 0
                    traj_prefix_len = int(kwargs.get("traj_prefix_len", 0) or 0)
                    runtime.layout = self.build_layout(
                        int(latents.shape[2]) // patch_size[0],
                        int(latents.shape[3]) // patch_size[1],
                        int(latents.shape[4]) // patch_size[2],
                        num_cond,
                        traj_len,
                        traj_prefix_len,
                    )
                runtime._model_call_state = {
                    "num_cond_latents": int(longcat.shape[2])
                    if torch.is_tensor(longcat) and longcat.ndim >= 3 else None,
                    "traj_len": int(traj.shape[1]) if torch.is_tensor(traj) and traj.ndim >= 2 else 0,
                    "traj_prefix_len": int(kwargs.get("traj_prefix_len", 0) or 0),
                }
                try:
                    return original_model_fn(*args, **kwargs)
                finally:
                    runtime._model_call_state = old_state

            pipe.model_fn = wrapped_model_fn
            runtime._driveva_model_fn_hook = (pipe, original_model_fn)
        if not hooks:
            raise RuntimeError("SELF_ATTN_KV hook could not find any DriveVA self-attention block")
        runtime._driveva_hooks = hooks
        runtime._driveva_video_hooks = []

    def remove_hooks(self, pipe, runtime) -> None:
        for model, original_patchify in getattr(runtime, "_driveva_video_hooks", []):
            model.patchify = original_patchify
        model_fn_hook = getattr(runtime, "_driveva_model_fn_hook", None)
        if model_fn_hook is not None:
            model_fn_hook[0].model_fn = model_fn_hook[1]
        runtime._driveva_video_hooks = []
        runtime._driveva_model_fn_hook = None
        for attention, attention_module, original_forward, had_public_hook, previous_public_hook in getattr(runtime, "_driveva_hooks", []):
            attention_module.forward = original_forward
            if had_public_hook:
                attention.tokenpress_hook = previous_public_hook
            else:
                try:
                    delattr(attention, "tokenpress_hook")
                except AttributeError:
                    pass
        runtime._driveva_hooks = []
        for model, attribute, had_controller, previous_controller in getattr(
            runtime, "_driveva_hidden_sequence_hooks", []
        ):
            if had_controller:
                setattr(model, attribute, previous_controller)
            else:
                try:
                    delattr(model, attribute)
                except AttributeError:
                    pass
        runtime._driveva_hidden_sequence_hooks = []
        for model, attribute, had_controller, previous_controller in getattr(
            runtime, "_driveva_pre_dit_hooks", []
        ):
            if had_controller:
                setattr(model, attribute, previous_controller)
            else:
                try:
                    delattr(model, attribute)
                except AttributeError:
                    pass
        runtime._driveva_pre_dit_hooks = []
        persistence_store = getattr(runtime, "_cross_layer_selection_store", None)
        if persistence_store is not None:
            persistence_store.clear()
        runtime._cross_layer_selection_store = None
