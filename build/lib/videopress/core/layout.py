"""DriveVA video/trajectory token layout and index mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TokenRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("TokenRange boundaries must be integers")
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid token range [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.end)

    def contains(self, index: int) -> bool:
        return self.start <= index < self.end


@dataclass(frozen=True)
class TokenLayout:
    total_length: int
    video: TokenRange
    history_video: TokenRange
    future_video: TokenRange
    traj_all: TokenRange
    traj_prefix: TokenRange
    future_action: TokenRange
    num_cond_latents: int
    video_f: int
    video_h: int
    video_w: int
    tokens_per_latent: int

    def __post_init__(self) -> None:
        if self.total_length < 0:
            raise ValueError("total_length must be non-negative")
        for name in (
            "video",
            "history_video",
            "future_video",
            "traj_all",
            "traj_prefix",
            "future_action",
        ):
            current = getattr(self, name)
            if current.end > self.total_length:
                raise ValueError(f"{name} exceeds total_length")
        if self.video.start != 0:
            raise ValueError("video range must start at zero")
        if self.history_video.start != self.video.start:
            raise ValueError("history_video must start at video.start")
        if self.future_video.start != self.history_video.end:
            raise ValueError("history_video and future_video must be contiguous")
        if self.future_video.end != self.video.end:
            raise ValueError("video range must equal history_video + future_video")
        if self.traj_all.start != self.video.end:
            raise ValueError("trajectory tokens must follow video tokens")
        if self.traj_prefix.start != self.traj_all.start:
            raise ValueError("trajectory prefix must start at traj_all.start")
        if self.future_action.start != self.traj_prefix.end:
            raise ValueError("future action must follow trajectory prefix")
        if self.future_action.end != self.traj_all.end:
            raise ValueError("trajectory ranges are not contiguous")
        if self.traj_all.end != self.total_length:
            raise ValueError("traj_all must end at total_length")
        if self.video_f <= 0 or self.video_h <= 0 or self.video_w <= 0:
            raise ValueError("video dimensions must be positive")
        if self.tokens_per_latent != self.video_h * self.video_w:
            raise ValueError("tokens_per_latent must equal video_h * video_w")
        if self.video.length != self.video_f * self.tokens_per_latent:
            raise ValueError("video range does not match video dimensions")
        if not 0 <= self.num_cond_latents <= self.video_f:
            raise ValueError("num_cond_latents must be within video_f")

    @property
    def num_video_tokens(self) -> int:
        return self.video.length

    @property
    def trajectory_length(self) -> int:
        return self.traj_all.length

    def frame_range(self, frame: int) -> TokenRange:
        if frame < 0 or frame >= self.video_f:
            raise IndexError(f"video frame {frame} is outside [0, {self.video_f})")
        start = self.video.start + frame * self.tokens_per_latent
        return TokenRange(start, start + self.tokens_per_latent)

    def to_dict(self) -> dict:
        return {
            "total_length": self.total_length,
            "video": {"start": self.video.start, "end": self.video.end},
            "history_video": {
                "start": self.history_video.start,
                "end": self.history_video.end,
            },
            "future_video": {"start": self.future_video.start, "end": self.future_video.end},
            "traj_all": {"start": self.traj_all.start, "end": self.traj_all.end},
            "traj_prefix": {"start": self.traj_prefix.start, "end": self.traj_prefix.end},
            "future_action": {
                "start": self.future_action.start,
                "end": self.future_action.end,
            },
            "num_cond_latents": self.num_cond_latents,
            "video_f": self.video_f,
            "video_h": self.video_h,
            "video_w": self.video_w,
            "tokens_per_latent": self.tokens_per_latent,
        }


def build_driveva_layout(
    f: int,
    h: int,
    w: int,
    num_cond_latents: int,
    traj_len: int,
    traj_prefix_len: int,
) -> TokenLayout:
    """Build the canonical ``video + trajectory`` DriveVA token layout."""

    for name, value in {
        "f": f,
        "h": h,
        "w": w,
        "num_cond_latents": num_cond_latents,
        "traj_len": traj_len,
        "traj_prefix_len": traj_prefix_len,
    }.items():
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if f <= 0 or h <= 0 or w <= 0:
        raise ValueError("f, h and w must be positive")
    if traj_len < 0 or traj_prefix_len < 0 or traj_prefix_len > traj_len:
        raise ValueError("invalid trajectory lengths")
    if num_cond_latents < 0 or num_cond_latents > f:
        raise ValueError("num_cond_latents must be within the video frame count")

    tokens_per_latent = h * w
    video_len = f * tokens_per_latent
    history_len = num_cond_latents * tokens_per_latent
    traj_start = video_len

    return TokenLayout(
        total_length=video_len + traj_len,
        video=TokenRange(0, video_len),
        history_video=TokenRange(0, history_len),
        future_video=TokenRange(history_len, video_len),
        traj_all=TokenRange(traj_start, traj_start + traj_len),
        traj_prefix=TokenRange(traj_start, traj_start + traj_prefix_len),
        future_action=TokenRange(traj_start + traj_prefix_len, traj_start + traj_len),
        num_cond_latents=num_cond_latents,
        video_f=f,
        video_h=h,
        video_w=w,
        tokens_per_latent=tokens_per_latent,
    )


def get_last_history_range(layout: TokenLayout) -> TokenRange:
    """Return the latest history latent without hard-coded token offsets."""

    if layout.history_video.length < layout.tokens_per_latent:
        raise ValueError("last_history requires at least one complete history latent")
    return TokenRange(
        layout.history_video.end - layout.tokens_per_latent,
        layout.history_video.end,
    )


def decode_video_index(index: int, h: int, w: int) -> tuple[int, int, int]:
    """Map a flattened video index to ``(frame, spatial_y, spatial_x)``."""

    if not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer")
    if h <= 0 or w <= 0:
        raise ValueError("h and w must be positive")
    frame, remainder = divmod(index, h * w)
    y, x = divmod(remainder, w)
    return frame, y, x


def decode_video_index_checked(index: int, layout: TokenLayout) -> tuple[int, int, int]:
    """Decode a global token index after checking that it is a video token.

    Artifact writers must never silently decode trajectory tokens as video
    coordinates.  ``decode_video_index`` remains a small local-index helper;
    this function is the canonical helper for a full DriveVA layout.
    """

    if not isinstance(index, int):
        raise TypeError("index must be an integer")
    if not layout.video.contains(index):
        raise ValueError(
            f"token {index} is not a video token; "
            f"video range=[{layout.video.start}, {layout.video.end})"
        )
    return decode_video_index(index - layout.video.start, layout.video_h, layout.video_w)


def decode_video_indices(indices: Iterable[int], h: int, w: int) -> list[tuple[int, int, int]]:
    return [decode_video_index(int(index), h, w) for index in indices]
