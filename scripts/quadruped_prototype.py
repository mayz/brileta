"""Quadruped sprite preview sheet generator.

Renders N seeds x 12 poses (S, N, W, E) x (stand, walk-A, walk-B) so dog
silhouette quality can be judged at 1x and inspected up close.

Run:
    uv run python scripts/critter_prototype.py -o critter_preview.png
    uv run python scripts/critter_prototype.py -o critter_preview.png --scale 8 -n 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from brileta.sprites.primitives import paste_sprite
from brileta.sprites.quadrupeds import (
    DOG_PRESET,
    QUADRUPED_POSES,
    draw_quadruped_pose,
    roll_quadruped_appearance,
)

COLUMN_LABELS = [
    "Front",
    "F-A",
    "F-B",
    "Back",
    "B-A",
    "B-B",
    "Left",
    "L-A",
    "L-B",
    "Right",
    "R-A",
    "R-B",
]
LABEL_WIDTH = 60
HEADER_HEIGHT = 12


def generate_sheet(
    n: int, base_size: int, base_seed: int, padding: int = 2
) -> tuple[np.ndarray, list[str]]:
    """Return an RGBA sheet of n rows x 12 pose columns, plus row labels."""
    cell = base_size + 6 + padding
    sheet_w = LABEL_WIDTH + len(QUADRUPED_POSES) * cell + padding
    sheet_h = HEADER_HEIGHT + n * cell + padding
    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    sheet[:, :] = (30, 30, 30, 255)

    labels: list[str] = []
    for row in range(n):
        seed = base_seed + row
        appearance = roll_quadruped_appearance(seed, DOG_PRESET, base_size)
        p = appearance.params
        labels.append(f"s{seed} {p.ear_kind.value[:2]}/{p.tail_kind.value[:3]}")
        for col, pose in enumerate(QUADRUPED_POSES):
            sprite = draw_quadruped_pose(appearance, pose)
            sh, sw = sprite.shape[:2]
            x0 = LABEL_WIDTH + col * cell + padding + (cell - padding - sw) // 2
            y0 = HEADER_HEIGHT + row * cell + padding + (cell - padding - sh)
            paste_sprite(sheet, sprite, x0, y0)
    return sheet, labels


def save_sheet(
    sheet: np.ndarray,
    labels: list[str],
    out: Path,
    scale: int,
    base_size: int,
    padding: int = 2,
) -> None:
    """Annotate row/column labels and save the sheet, upscaled by ``scale``."""
    from PIL import Image, ImageDraw
    from PIL.Image import Resampling

    img = Image.fromarray(sheet, "RGBA")
    draw = ImageDraw.Draw(img)
    cell = base_size + 6 + padding
    for col, label in enumerate(COLUMN_LABELS):
        draw.text(
            (LABEL_WIDTH + col * cell + padding, 1), label, fill=(220, 220, 220, 255)
        )
    for row, label in enumerate(labels):
        draw.text((2, HEADER_HEIGHT + row * cell + 2), label, fill=(220, 220, 220, 255))
    if scale > 1:
        img = img.resize(
            (img.width * scale, img.height * scale), resample=Resampling.NEAREST
        )
    img.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview procedural critter sprites.")
    parser.add_argument("-o", "--output", default="critter_preview.png")
    parser.add_argument("-n", "--n-critters", type=int, default=12)
    parser.add_argument("-s", "--size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=6)
    args = parser.parse_args()

    sheet, labels = generate_sheet(args.n_critters, args.size, args.seed)
    out_path = Path(args.output)
    save_sheet(sheet, labels, out_path, args.scale, args.size)
    print(f"Saved {out_path} ({sheet.shape[1]}x{sheet.shape[0]} @ {args.scale}x)")
