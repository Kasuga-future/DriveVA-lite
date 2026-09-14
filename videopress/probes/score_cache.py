"""Disk cache for offline probe scores."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class ScoreKey:
    scene_token: str
    diffusion_rank: int | None
    layer_idx: int | None
    scorer_signature: str

    def filename(self) -> str:
        raw = json.dumps(
            {
                "scene_token": self.scene_token,
                "diffusion_rank": self.diffusion_rank,
                "layer_idx": self.layer_idx,
                "scorer_signature": self.scorer_signature,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest() + ".pt"


class ScoreCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: ScoreKey) -> Path:
        return self.root / key.filename()

    @staticmethod
    def _digest(value: torch.Tensor | None) -> str | None:
        if value is None:
            return None
        tensor = value.detach().cpu().contiguous()
        # Byte-view hashing also supports dtypes such as bfloat16 that NumPy
        # cannot represent directly.
        header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
        return hashlib.sha256(header + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()

    def save(
        self,
        key: ScoreKey,
        scores: torch.Tensor,
        *,
        ranking: torch.Tensor | None = None,
        metadata: dict | None = None,
    ) -> Path:
        path = self.path_for(key)
        torch.save(
            {
                "key": key.__dict__,
                "scores": scores.detach().cpu(),
                "ranking": None if ranking is None else ranking.detach().cpu(),
                "metadata": dict(metadata or {}),
                "score_digest": self._digest(scores),
                "ranking_digest": self._digest(ranking),
            },
            path,
        )
        return path

    def _load_payload(self, key: ScoreKey, map_location: str | torch.device = "cpu") -> dict:
        path = self.path_for(key)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("key") != key.__dict__:
            raise ValueError(f"score cache key mismatch in {path}")
        if payload.get("score_digest") != self._digest(payload.get("scores")):
            raise ValueError(f"score cache digest mismatch in {path}")
        if payload.get("ranking") is not None and payload.get("ranking_digest") != self._digest(payload.get("ranking")):
            raise ValueError(f"ranking cache digest mismatch in {path}")
        return payload

    def load(self, key: ScoreKey, map_location: str | torch.device = "cpu") -> torch.Tensor:
        return self._load_payload(key, map_location)["scores"]

    def load_ranking(self, key: ScoreKey, map_location: str | torch.device = "cpu") -> torch.Tensor | None:
        return self._load_payload(key, map_location).get("ranking")

    def metadata(self, key: ScoreKey) -> dict:
        return dict(self._load_payload(key).get("metadata") or {})

    def digest(self, key: ScoreKey) -> dict:
        payload = self._load_payload(key)
        return {
            "score_digest": payload.get("score_digest"),
            "ranking_digest": payload.get("ranking_digest"),
        }

    def contains(self, key: ScoreKey) -> bool:
        return self.path_for(key).exists()
