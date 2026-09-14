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
from ..core.runtime import InjectionPoint
from .wan_attention import canonicalize_wan_qkv, restore_wan_qkv


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
        return dict(metadata) if isinstance(metadata, dict) else {}

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
        if point is InjectionPoint.SELF_ATTN_KV:
            self._install_kv_hooks(pipe, runtime)
            return
        raise NotImplementedError(f"{point.value} is declared but not implemented")

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
        for model_name in ("dit", "dit2"):
            model = getattr(pipe, model_name, None)
            if model is None or not hasattr(model, "blocks"):
                continue
            for layer_idx, block in enumerate(model.blocks):
                attention = getattr(block, "self_attn", None)
                if attention is None or not hasattr(attention, "attn"):
                    continue

                def hook(q, k, v, *, layer_idx=None, _layer_idx=layer_idx, _attention=attention):
                    if runtime.press is None or runtime.layout is None:
                        raise RuntimeError("SELF_ATTN_KV hook requires an active layout and press")
                    scorer = getattr(runtime.press, "scorer", None)
                    requested_layer = getattr(scorer, "layer", None)
                    effective_layer = _layer_idx if layer_idx is None else layer_idx
                    if requested_layer is not None and int(requested_layer) != int(effective_layer):
                        return q, k, v
                    num_heads = int(getattr(_attention, "num_heads", 0))
                    q_c, q_flat = canonicalize_wan_qkv(q, num_heads)
                    k_c, _ = canonicalize_wan_qkv(k, num_heads)
                    v_c, _ = canonicalize_wan_qkv(v, num_heads)
                    dummy_tokens = q_c.transpose(1, 2).reshape(q_c.shape[0], q_c.shape[2], -1)
                    configured, resolved, overridden, domain = self._resolve_domain_for_device(
                        runtime, runtime.layout, dummy_tokens.device
                    )
                    metadata = self._sample_metadata(runtime)
                    metadata.update(self._domain_metadata(configured, resolved, domain, overridden))
                    metadata.update(
                        {
                            "injection_point": InjectionPoint.SELF_ATTN_KV.value,
                            "post_rope": True,
                            "model_name": model_name,
                        }
                    )
                    context = self.create_context(
                        dummy_tokens,
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
                    result = runtime.execute_press(context)
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
