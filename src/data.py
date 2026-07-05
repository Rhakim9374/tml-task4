"""Dataset loading and the fixed WM-group -> clean-target mapping.

The forgery task ships:
  * ``clean_targets/``          200 clean images ``1.png`` .. ``200.png``
  * ``watermarked_sources/WM_k`` 8 groups of 25 images, each group carrying the
    SAME hidden watermark message (embedded by one unknown method).

The assignment fixes which watermark goes onto which clean images (25 per group):

    WM_1 -> 1..25   WM_2 -> 26..50  WM_3 -> 51..75   WM_4 -> 76..100
    WM_5 -> 101..125 WM_6 -> 126..150 WM_7 -> 151..175 WM_8 -> 176..200

Empirically each group's source images are at the SAME resolution as its clean
target batch (WM_5: 128x128, WM_7/WM_8: 512x512, the rest 256x256), so no
resizing is needed when transplanting a watermark onto its targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# (group name, first target index, last target index) — inclusive, 1-based.
GROUPS = [
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]

DEFAULT_DATASET = Path("data/Dataset")


@dataclass
class Group:
    name: str
    start: int
    stop: int
    source_paths: list[Path]
    target_paths: list[Path]


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as float32 RGB in [0, 255], shape (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    """Save a float array (values in [0, 255]) as an 8-bit PNG."""
    arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _source_dir(dataset: Path, group: str) -> Path:
    return dataset / "watermarked_sources" / group


def _sorted_sources(source_dir: Path) -> list[Path]:
    # filenames look like src_<n>.png; sort by n so halves are content-diverse.
    return sorted(source_dir.glob("*.png"), key=lambda p: int(p.stem.split("_")[1]))


def iter_groups(dataset: Path = DEFAULT_DATASET) -> list[Group]:
    """Return the 8 groups with their resolved source and target file paths."""
    dataset = Path(dataset)
    targets = dataset / "clean_targets"
    out = []
    for name, start, stop in GROUPS:
        src = _sorted_sources(_source_dir(dataset, name))
        tgt = [targets / f"{i}.png" for i in range(start, stop + 1)]
        out.append(Group(name, start, stop, src, tgt))
    return out


def load_group_sources(group: Group) -> np.ndarray:
    """Stack a group's 25 watermarked sources into (N, H, W, 3) float32."""
    return np.stack([load_rgb(p) for p in group.source_paths])
