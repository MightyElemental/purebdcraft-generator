# dataset.py
"""
Paired source/target dataset for progressive image-to-image training.

Design:
- Scans `source_root` and pairs files with `target_root` by relative path.
- Records each target image's original maximum side length (for stage filtering).
- Use `update_stage(min_target_size)` to restrict pairs to targets >= that size
  without re-scanning the filesystem.
- Robust to unreadable/corrupt images (skips them at scan time).
"""

from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path
import os

from PIL import Image, UnidentifiedImageError
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


def _is_image_file(fname: str) -> bool:
    """Return True if filename looks like an image file by extension."""
    return fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

class PairedSourceTargetDataset(Dataset):
    """
    Pairs <source_root>/<name>/<category>/... with <target_root>/<name>/<category>/...
    and allows filtering by target original size for progressive stages.

    Example layout:
        data/
          source/
            setA/
              blocks/
                file1.png
          target/
            setA/
              blocks/
                file1.png

    Args:
        source_root: path to source root (e.g., ./data/source)
        target_root: path to target root (e.g., ./data/target)
        source_size: size (pixels) to resize source inputs for the model (default 16)
        initial_stage_size: initial minimum target size (e.g., 32 or 128). Use update_stage(...) to change.
        allowed_categories: optional whitelist of category names to include. Categories outside of this list will be put in `other`.
        ignore_categories: optional blacklist of categories to ignore.
    """

    def __init__(
        self,
        source_root: str,
        target_root: str,
        source_size: int = 16,
        initial_stage_size: int = 32,
        allowed_categories: Optional[List[str]] = None,
        ignore_categories: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.source_size = int(source_size)
        self.current_stage_size = int(initial_stage_size)
        self.allowed_categories = set(allowed_categories) if allowed_categories is not None else None
        self.ignore_categories = set(ignore_categories) if ignore_categories is not None else None

        # default transforms: source -> source_size, target -> current_stage_size (target resize updated by update_stage)
        self._make_transforms()

        # internal master list of all discovered pairs (not filtered by stage)
        # entries: (source_path: Path, target_path: Path, category: str, target_max_side: int, rel_path: str)
        self.pairs: List[Tuple[Path, Path, str, int, str]] = []

        # active_pairs is a view filtered according to current_stage_size
        self.active_pairs: List[Tuple[Path, Path, str, int, str]] = []

        # category maps
        self.cat2idx: Dict[str, int] = {}
        self.idx2cat: Dict[int, str] = {}

        # scan filesystem once and build master pairs (skips unreadable files)
        self._scan_pairs()

        # initialize active_pairs based on initial_stage_size
        self.update_stage(self.current_stage_size)

    def _make_transforms(self) -> None:
        """(Re)create transform callables according to sizes."""
        self.transform_source_rgb = transforms.Compose([
            transforms.Resize((self.source_size, self.source_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.transform_source_alpha = transforms.Compose([
            transforms.Resize((self.source_size, self.source_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        # target transforms will use the current_stage_size; recreated in update_stage
        self.transform_target_rgb = transforms.Compose([
            transforms.Resize((self.current_stage_size, self.current_stage_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])
        self.transform_target_alpha = transforms.Compose([
            transforms.Resize((self.current_stage_size, self.current_stage_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _load_rgba_pil(self, path: Path) -> Optional[Tuple[Image.Image, Image.Image]]:
        """
        Try to open an image as RGBA and return (rgb_pil, alpha_pil).
        Returns None if the image cannot be opened/parsed.
        """
        try:
            with Image.open(path) as im:
                im = im.convert("RGBA")
                r, g, b, a = im.split()
                rgb = Image.merge("RGB", (r, g, b))
                alpha = a  # L mode
                return rgb, alpha
        except (UnidentifiedImageError, OSError) as e:
            # unreadable/corrupt image — skip it
            print(f"[WARN] Skipping unreadable image: {path} ({e})")
            return None

    def _get_image_max_side(self, path: Path) -> Optional[int]:
        """
        Return the max(width, height) of the image at `path`, or None if it can't be opened.
        This is used for stage filtering.
        """
        try:
            with Image.open(path) as im:
                w, h = im.size
                return max(w, h)
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] Cannot read image size for: {path} ({e})")
            return None

    def _scan_pairs(self) -> None:
        """
        Walk the source root and build a master list of pairs that have a corresponding
        target file. For each pair we also record the target original size (max side).
        Corrupted/unreadable files are skipped.
        """
        if not self.source_root.exists() or not self.target_root.exists():
            raise RuntimeError(f"Source or target root does not exist: {self.source_root}, {self.target_root}")

        pairs: List[Tuple[Path, Path, str, int, str]] = []

        for src_path in self.source_root.rglob("*"):
            if not src_path.is_file():
                continue
            if not _is_image_file(src_path.name):
                continue
            # relative path from source_root (e.g., <name>/<category>/sub/.../file.png)
            try:
                rel = src_path.relative_to(self.source_root)
            except Exception:
                # some odd path — skip
                continue
            parts = rel.parts  # tuple of path components
            # we require at least: <name>/<category>/<...>/<file>
            if len(parts) < 2:
                # not in expected layout; skip
                continue
            # category is the second part according to your layout: source/<name>/<category>/...
            # when source_root is ./data/source, rel.parts -> (name, category, ...)
            category = parts[1]
            if self.allowed_categories is not None and category not in self.allowed_categories:
                continue
            # build corresponding target path by the same relative path
            tgt_path = self.target_root / rel
            if not tgt_path.exists():
                # no matching target; skip
                continue
            # check target size; if unreadable skip
            tgt_max_side = self._get_image_max_side(tgt_path)
            if tgt_max_side is None:
                # unreadable target -> skip
                continue
            # ensure source is readable as well (we'll open it later, but skip obviously broken files early)
            # We don't read full image here, just a quick open check
            try:
                with Image.open(src_path) as _:
                    pass
            except (UnidentifiedImageError, OSError) as e:
                print(f"[WARN] Skipping unreadable source image: {src_path} ({e})")
                continue

            pairs.append((src_path, tgt_path, category, int(tgt_max_side), str(rel.as_posix())))

        # sort deterministically
        pairs.sort(key=lambda x: x[4])
        self.pairs = pairs

        # build category mapping from discovered categories
        cats = sorted({c for (_, _, c, _, _) in pairs})
        self.cat2idx = {c: i for i, c in enumerate(cats)}
        self.idx2cat = {i: c for c, i in self.cat2idx.items()}

        print(f"loaded {len(self.pairs)} image pairs (pre-filter). categories: {len(self.cat2idx)}")

    def update_stage(self, min_target_size: int) -> None:
        """
        Filter pairs to include only targets whose original max side >= min_target_size.
        Also updates the internal target transforms to resize to the new stage resolution.

        Call this each time you advance to a new progressive training stage.
        """
        self.current_stage_size = int(min_target_size)
        # update target transforms
        self.transform_target_rgb = transforms.Compose([
            transforms.Resize((self.current_stage_size, self.current_stage_size),
                              interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])
        self.transform_target_alpha = transforms.Compose([
            transforms.Resize((self.current_stage_size, self.current_stage_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        self.active_pairs = [p for p in self.pairs if p[3] >= self.current_stage_size]
        print(f"stage {self.current_stage_size}: {len(self.active_pairs)} pairs (filtered)")

    def __len__(self) -> int:
        return len(self.active_pairs)

    def __getitem__(self, index: int):
        """
        Return a dict with:
         - source_rgb: Tensor (3 x Hs x Ws) in -1..1
         - source_alpha: Tensor (1 x Hs x Ws) 0/1
         - target_rgb: Tensor (3 x St x St) in -1..1
         - target_alpha: Tensor (1 x St x St) 0/1
         - label: torch.long scalar (category index)
         - rel_path: str (the relative path used for pairing)
        """
        if len(self.active_pairs) == 0:
            raise IndexError("Dataset has no active pairs for current stage size. Call update_stage(min_target_size) with a smaller value or verify your dataset.")

        src_path, tgt_path, category, tgt_max_side, rel_path = self.active_pairs[index]

        # load source and target RGBA safely
        src_rgba = self._load_rgba_pil(src_path)
        tgt_rgba = self._load_rgba_pil(tgt_path)
        if src_rgba is None or tgt_rgba is None:
            # In the unlikely event the file became unreadable after scan,
            # skip this index by mapping to the next valid index.
            # (This ensures DataLoader workers don't crash.)
            next_index = (index + 1) % len(self.active_pairs)
            return self.__getitem__(next_index)

        src_rgb_pil, src_alpha_pil = src_rgba
        tgt_rgb_pil, tgt_alpha_pil = tgt_rgba

        # apply transforms
        source_rgb_t: Tensor = self.transform_source_rgb(src_rgb_pil)  # 0..1
        source_alpha_t: Tensor = self.transform_source_alpha(src_alpha_pil)  # 0..1 single channel
        target_rgb_t: Tensor = self.transform_target_rgb(tgt_rgb_pil)  # 0..1
        target_alpha_t: Tensor = self.transform_target_alpha(tgt_alpha_pil)  # 0..1

        # binary alpha (dataset spec said alpha is fully transparent or fully solid)
        source_alpha_bin = (source_alpha_t > 0.5).float()
        target_alpha_bin = (target_alpha_t > 0.5).float()

        # normalize RGB to -1..1
        source_rgb_norm = source_rgb_t * 2.0 - 1.0
        target_rgb_norm = target_rgb_t * 2.0 - 1.0

        label_idx = self.cat2idx.get(category, 0)

        return {
            "source_rgb": source_rgb_norm,
            "source_alpha": source_alpha_bin,
            "target_rgb": target_rgb_norm,
            "target_alpha": target_alpha_bin,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "rel_path": rel_path,
        }
