"""Render individual quadruped pose images for the blind-judge loop.

For each seed, writes `seed<N>_left_stand.png` (authored side view) and
`seed<N>_front_stand.png` (front view), each upscaled 12x with nearest-neighbor
scaling so a 20x20 sprite becomes a crisp 240x240 image.

Run:
    uv run python scripts/quadruped_judge_render.py -o .context/dog-judging/round-1 \
        --seeds 1 2 3 4 5 6 7 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from brileta.sprites.quadrupeds import (
    DOG_PRESET,
    QUADRUPED_POSES,
    draw_quadruped_pose,
    roll_quadruped_appearance,
)

# Pose name -> index into QUADRUPED_POSES. left_stand is the authored side view
# (index 6), front_stand is the resting front view (index 0).
_POSE_BY_NAME = {pose.name: i for i, pose in enumerate(QUADRUPED_POSES)}
_RENDER_POSES = ("left_stand", "front_stand")
_SCALE = 12


def _upscale(sprite: np.ndarray, scale: int) -> Image.Image:
    """Nearest-neighbor upscale via np.repeat on both axes (no smoothing)."""
    big = np.repeat(np.repeat(sprite, scale, axis=0), scale, axis=1)
    return Image.fromarray(big, "RGBA")


def render_seeds(seeds: list[int], out_dir: Path, size: int = 20) -> None:
    """Render the side/front stand images for each seed into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        appearance = roll_quadruped_appearance(seed, DOG_PRESET, size)
        for pose_name in _RENDER_POSES:
            pose = QUADRUPED_POSES[_POSE_BY_NAME[pose_name]]
            sprite = draw_quadruped_pose(appearance, pose)
            img = _upscale(sprite, _SCALE)
            img.save(out_dir / f"seed{seed}_{pose_name}.png")
    print(f"Rendered {len(seeds)} seeds x {len(_RENDER_POSES)} poses into {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render individual dog pose images.")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    parser.add_argument("-s", "--size", type=int, default=20)
    args = parser.parse_args()

    render_seeds(args.seeds, Path(args.output), args.size)
