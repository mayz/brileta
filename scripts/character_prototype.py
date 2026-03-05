"""Character sprite preview and diagnostic sheet generator.

Run:
    uv run python scripts/character_prototype.py -o character_preview.png
    uv run python scripts/character_prototype.py --poses -o pose_preview.png
    uv run python scripts/character_prototype.py --diagnostic -o character_diagnostic.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from brileta.sprites.characters import (
    CARICATURE_TEMPLATES,
    CLOTHING_PALETTES,
    CLOTHING_STYLE_NAMES,
    HAIR_PALETTES,
    HAIR_STYLE_NAMES,
    PANTS_PALETTES,
    POSE_STAND,
    POSES,
    SKIN_PALETTES,
    CharacterAppearance,
    draw_character_pose,
    generate_character_sprite,
)
from brileta.sprites.primitives import paste_sprite

POSE_DIAGNOSTIC_LABEL_WIDTH = 104
POSE_DIAGNOSTIC_HEADER_HEIGHT = 20
POSE_DIAGNOSTIC_COLUMN_LABELS: list[str] = [
    "Front",
    "Back",
    "Left",
    "Right",
]
POSE_DIAGNOSTIC_HEADER_NOTES: list[str] = []


def _paste_bottom_center(
    sheet: np.ndarray,
    sprite: np.ndarray,
    cell: int,
    padding: int,
    col: int,
    row: int,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> None:
    """Paste one sprite centered in its cell and anchored to the bottom."""
    sh, sw = sprite.shape[:2]
    x0 = x_offset + col * cell + padding + (cell - padding - sw) // 2
    y0 = y_offset + row * cell + padding + (cell - padding - sh)
    paste_sprite(sheet, sprite, x0, y0)


def generate_preview_sheet(
    columns: int = 16,
    rows: int = 10,
    base_size: int = 20,
    padding: int = 2,
    bg_rgba: tuple[int, int, int, int] = (30, 30, 30, 255),
    base_seed: int = 42,
) -> np.ndarray:
    """Generate a grid of standing character sprites."""
    cell = base_size + 8 + padding
    sheet_w = columns * cell + padding
    sheet_h = rows * cell + padding

    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    sheet[:, :] = bg_rgba

    for idx in range(columns * rows):
        col = idx % columns
        row = idx // columns
        sprite = generate_character_sprite(base_seed + idx, base_size)
        _paste_bottom_center(sheet, sprite, cell, padding, col, row)

    return sheet


def _appearance_metadata(
    row: int,
    seed: int,
    appearance: CharacterAppearance,
) -> dict[str, object]:
    """Serialize one rolled appearance row for diagnostic output."""
    return {
        "row": row,
        "seed": seed,
        "build_name": appearance.build_name,
        "body_params": asdict(appearance.body_params),
        "skin_palette_idx": SKIN_PALETTES.index(appearance.skin_pal),
        "hair_palette_idx": HAIR_PALETTES.index(appearance.hair_pal),
        "clothing_palette_idx": CLOTHING_PALETTES.index(appearance.cloth_pal),
        "pants_palette_idx": PANTS_PALETTES.index(appearance.pants_pal),
        "hair_style_idx": appearance.hair_style_idx,
        "hair_style_name": HAIR_STYLE_NAMES[appearance.hair_style_idx],
        "clothing_type_idx": appearance.clothing_style_idx,
        "clothing_type_name": CLOTHING_STYLE_NAMES[appearance.clothing_style_idx],
        "canvas_size": appearance.body_params.canvas_size,
    }


def generate_front_diagnostic_sheet(
    n_characters: int = 5,
    columns: int = 5,
    base_size: int = 20,
    padding: int = 2,
    bg_rgba: tuple[int, int, int, int] = (30, 30, 30, 255),
    base_seed: int = 42,
    start_id: int = 1,
) -> tuple[np.ndarray, list[dict[str, object]], list[tuple[int, int, str]]]:
    """Generate front-facing sprites with stable IDs for systematic review.

    Returns:
        - RGBA sheet image
        - metadata rows (one per sprite)
        - cell labels as (row, col, text) triples for image annotation
    """
    if n_characters <= 0:
        raise ValueError("n_characters must be > 0")
    if columns <= 0:
        raise ValueError("columns must be > 0")

    cell = base_size + 8 + padding
    rows = math.ceil(n_characters / columns)
    sheet_w = columns * cell + padding
    sheet_h = rows * cell + padding
    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    sheet[:, :] = bg_rgba

    metadata: list[dict[str, object]] = []
    cell_labels: list[tuple[int, int, str]] = []

    for idx in range(n_characters):
        col = idx % columns
        row = idx // columns
        seed = base_seed + idx
        sprite_id = start_id + idx

        appearance = CharacterAppearance.from_seed(seed, base_size)
        sprite = draw_character_pose(appearance, POSE_STAND)  # Front stand only.
        _paste_bottom_center(sheet, sprite, cell, padding, col, row)

        row_meta = _appearance_metadata(row, seed, appearance)
        row_meta["id"] = sprite_id
        row_meta["col"] = col
        metadata.append(row_meta)
        cell_labels.append((row, col, str(sprite_id)))

    return sheet, metadata, cell_labels


def generate_pose_sheet(
    n_characters: int = 20,
    base_size: int = 20,
    padding: int = 2,
    bg_rgba: tuple[int, int, int, int] = (30, 30, 30, 255),
    base_seed: int = 42,
    *,
    include_labels: bool = False,
    start_id: int = 1,
) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    """Generate rows of front/back/left/right pose frames for each character."""
    cell = base_size + 8 + padding
    label_width = POSE_DIAGNOSTIC_LABEL_WIDTH if include_labels else 0
    header_height = POSE_DIAGNOSTIC_HEADER_HEIGHT if include_labels else 0
    n_cols = len(POSES)

    sheet_w = label_width + n_cols * cell + padding
    sheet_h = header_height + n_characters * cell + padding
    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    sheet[:, :] = bg_rgba

    labels: list[str] = []
    metadata: list[dict[str, object]] = []

    for row in range(n_characters):
        seed = base_seed + row
        appearance = CharacterAppearance.from_seed(seed, base_size)
        directional_sprites = [draw_character_pose(appearance, pose) for pose in POSES]

        for col, sprite in enumerate(directional_sprites):
            _paste_bottom_center(
                sheet,
                sprite,
                cell,
                padding,
                col,
                row,
                x_offset=label_width,
                y_offset=header_height,
            )

        if include_labels:
            sprite_id = start_id + row
            labels.append(f"{sprite_id} seed {seed}")
            row_meta = _appearance_metadata(row, seed, appearance)
            row_meta["id"] = sprite_id
            metadata.append(row_meta)

    return sheet, labels, metadata


def generate_mixed_sheet(
    columns: int = 16,
    rows: int = 10,
    base_size: int = 20,
    padding: int = 2,
    bg_rgba: tuple[int, int, int, int] = (30, 30, 30, 255),
    base_seed: int = 42,
) -> np.ndarray:
    """Generate a grid with character rows and tree rows for comparison."""
    from brileta.sprites.trees import TreeArchetype, generate_tree_sprite

    cell = base_size + 8 + padding
    sheet_w = columns * cell + padding
    sheet_h = rows * cell + padding

    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    sheet[:, :] = bg_rgba

    tree_rows = 2
    char_rows = rows - tree_rows

    for idx in range(columns * char_rows):
        col = idx % columns
        row = idx // columns
        sprite = generate_character_sprite(base_seed + idx, base_size)
        _paste_bottom_center(sheet, sprite, cell, padding, col, row)

    archetypes = list(TreeArchetype)
    rng = np.random.default_rng(base_seed + 9999)
    for idx in range(columns * tree_rows):
        col = idx % columns
        row = char_rows + idx // columns
        arch = archetypes[int(rng.integers(len(archetypes)))]
        sprite = generate_tree_sprite(int(rng.integers(0, 2**31)), arch, base_size)
        _paste_bottom_center(sheet, sprite, cell, padding, col, row)

    return sheet


def generate_caricature_sheet(
    n_variations: int = 10,
    base_size: int = 20,
    padding: int = 2,
    bg_rgba: tuple[int, int, int, int] = (30, 30, 30, 255),
    base_seed: int = 42,
) -> np.ndarray:
    """Show each caricature body template side by side."""
    cell = base_size + 8 + padding
    sheet_w = len(CARICATURE_TEMPLATES) * cell + padding
    sheet_h = n_variations * cell + padding

    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    sheet[:, :] = bg_rgba

    for row in range(n_variations):
        seed = base_seed + row
        for col, (_name, generate_fn) in enumerate(CARICATURE_TEMPLATES):
            sprite = generate_fn(seed, base_size)
            _paste_bottom_center(sheet, sprite, cell, padding, col, row)

    return sheet


def _save_sheet_with_labels(
    sheet: np.ndarray,
    output_path: Path,
    *,
    scale: int,
    row_labels: list[str] | None = None,
    cell_labels: list[tuple[int, int, str]] | None = None,
    column_labels: list[str] | None = None,
    header_notes: list[str] | None = None,
    label_width: int = 0,
    header_height: int = 0,
    base_size: int = 20,
    padding: int = 2,
) -> tuple[int, int]:
    """Save preview sheet to disk, optionally annotating row labels."""
    from PIL import Image, ImageDraw
    from PIL.Image import Resampling

    img = Image.fromarray(sheet, "RGBA")

    if row_labels:
        draw = ImageDraw.Draw(img)
        cell = base_size + 8 + padding
        for row, label in enumerate(row_labels):
            y = header_height + row * cell + max(1, padding)
            draw.text((4, y), label, fill=(220, 220, 220, 255))

    if column_labels:
        draw = ImageDraw.Draw(img)
        cell = base_size + 8 + padding
        y = max(1, padding)
        for col, label in enumerate(column_labels):
            x = label_width + col * cell + padding + 1
            draw.text((x, y), label, fill=(220, 220, 220, 255))

    if header_notes:
        draw = ImageDraw.Draw(img)
        y = 1
        for note in header_notes:
            draw.text((4, y), note, fill=(200, 200, 200, 255))
            y += 9

    if cell_labels:
        draw = ImageDraw.Draw(img)
        cell = base_size + 8 + padding
        for row, col, label in cell_labels:
            x = label_width + col * cell + padding + 1
            y = header_height + row * cell + padding + 1
            draw.text((x, y), label, fill=(245, 245, 140, 255))

    if scale > 1:
        img = img.resize(
            (img.width * scale, img.height * scale),
            resample=Resampling.NEAREST,
        )

    img.save(output_path)
    return img.width, img.height


def _write_feedback_template(
    output_path: Path,
    metadata: list[dict[str, object]],
) -> Path:
    """Write a CSV template for structured per-sprite feedback."""
    csv_path = output_path.with_name(f"{output_path.stem}_feedback.csv")
    fieldnames = [
        "id",
        "seed",
        "build_name",
        "hair_style_name",
        "clothing_type_name",
        "overall",
        "head_size",
        "face_legibility",
        "hair_legibility",
        "silhouette_readability",
        "notes",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata:
            writer.writerow(
                {
                    "id": row.get("id"),
                    "seed": row.get("seed"),
                    "build_name": row.get("build_name"),
                    "hair_style_name": row.get("hair_style_name"),
                    "clothing_type_name": row.get("clothing_type_name"),
                    "overall": "",
                    "head_size": "",
                    "face_legibility": "",
                    "hair_legibility": "",
                    "silhouette_readability": "",
                    "notes": "",
                }
            )
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate preview PNGs for procedural character sprites."
    )
    parser.add_argument("-o", "--output", default="character_preview.png")
    parser.add_argument("-c", "--columns", type=int, default=16)
    parser.add_argument("-r", "--rows", type=int, default=10)
    parser.add_argument("-s", "--size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=4)

    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--poses", action="store_true")
    parser.add_argument("--n-characters", type=int, default=20)
    parser.add_argument("--caricature", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Write pose sheet plus <output_stem>_params.json metadata.",
    )
    parser.add_argument(
        "--front-diagnostic",
        action="store_true",
        help=(
            "Write numbered front-facing sheet plus params and feedback template "
            "for systematic review."
        ),
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Starting sprite ID used by --front-diagnostic and --diagnostic.",
    )

    args = parser.parse_args()
    output_path = Path(args.output)

    labels: list[str] | None = None
    cell_labels: list[tuple[int, int, str]] | None = None
    column_labels: list[str] | None = None
    header_notes: list[str] | None = None
    metadata: list[dict[str, object]] | None = None
    label_width = 0
    header_height = 0

    if args.front_diagnostic:
        sheet, metadata, cell_labels = generate_front_diagnostic_sheet(
            n_characters=args.n_characters,
            columns=args.columns,
            base_size=args.size,
            base_seed=args.seed,
            start_id=args.start_id,
        )
    elif args.diagnostic:
        sheet, labels, metadata = generate_pose_sheet(
            n_characters=args.n_characters,
            base_size=args.size,
            base_seed=args.seed,
            include_labels=True,
            start_id=args.start_id,
        )
        column_labels = POSE_DIAGNOSTIC_COLUMN_LABELS
        header_notes = POSE_DIAGNOSTIC_HEADER_NOTES
        label_width = POSE_DIAGNOSTIC_LABEL_WIDTH
        header_height = POSE_DIAGNOSTIC_HEADER_HEIGHT
    elif args.caricature:
        sheet = generate_caricature_sheet(
            n_variations=12,
            base_size=args.size,
            base_seed=args.seed,
        )
    elif args.poses:
        sheet, _, _ = generate_pose_sheet(
            n_characters=args.n_characters,
            base_size=args.size,
            base_seed=args.seed,
            include_labels=False,
        )
    elif args.mixed:
        sheet = generate_mixed_sheet(
            columns=args.columns,
            rows=args.rows,
            base_size=args.size,
            base_seed=args.seed,
        )
    else:
        sheet = generate_preview_sheet(
            columns=args.columns,
            rows=args.rows,
            base_size=args.size,
            base_seed=args.seed,
        )

    width, height = _save_sheet_with_labels(
        sheet,
        output_path,
        scale=args.scale,
        row_labels=labels,
        cell_labels=cell_labels,
        column_labels=column_labels,
        header_notes=header_notes,
        label_width=label_width,
        header_height=header_height,
        base_size=args.size,
    )

    if metadata is not None:
        json_path = output_path.with_name(f"{output_path.stem}_params.json")
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved diagnostic metadata to {json_path}")
        if args.front_diagnostic:
            csv_path = _write_feedback_template(output_path, metadata)
            print(f"Saved feedback template to {csv_path}")

    print(f"Saved {width}x{height} preview to {output_path}")
    sys.exit(0)
