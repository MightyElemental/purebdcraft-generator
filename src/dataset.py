# dataset.py
from typing import Callable, Dict, List, Optional, Tuple
import os
from pathlib import Path
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


def _is_image_file(fname: str) -> bool:
    """Return True if filename looks like an image."""
    return fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))


class PairedSourceTargetDataset(Dataset):
    """
    Dataset that pairs files by relative path between `source_root` and `target_root`.
    Category label is inferred as the first directory in the relative path (e.g. `source/<category>/...`).

    Both source and target are loaded as RGBA (if available). Alpha is converted to a
    binary mask (0.0 or 1.0) and returned separately. Source is expected to be low-res
    (e.g. 16x16) but will be resized if needed.

    Returned dict keys:
      - 'source_rgb' : Tensor (3 x Hs x Ws) in range [-1, 1]
      - 'source_alpha': Tensor (1 x Hs x Ws) with 0.0 or 1.0
      - 'target_rgb' : Tensor (3 x St x St) in range [-1, 1]
      - 'target_alpha': Tensor (1 x St x St) with 0.0 or 1.0
      - 'label' : Tensor scalar long (category index)
      - 'rel_path' : relative path string (useful for saving outputs)
    """

    def __init__(
        self,
        source_root: str,
        target_root: str,
        target_size: int = 128,
        source_size: int = 16,
        allowed_categories: Optional[List[str]] = None,
        transform_source_rgb: Optional[Callable] = None,
        transform_source_alpha: Optional[Callable] = None,
        transform_target_rgb: Optional[Callable] = None,
        transform_target_alpha: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            source_root: Path to folder containing source images (with category subfolders).
            target_root: Path to folder containing target images (mirrored structure).
            target_size: Integer; size for target images (S x S).
            source_size: Integer; size for source images (Hs x Ws). Defaults to 16.
            allowed_categories: Optional list of categories (subfolder names) to include.
            transform_*: optional torchvision transforms applied to respective images (PIL -> Tensor).
        """
        super().__init__()
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.target_size = int(target_size)
        self.source_size = int(source_size)
        self.allowed_categories = set(allowed_categories) if allowed_categories else None

        # default simple transforms if not provided
        self.transform_source_rgb = transform_source_rgb or transforms.Compose([
            transforms.Resize((self.source_size, self.source_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # 0..1
        ])
        self.transform_source_alpha = transform_source_alpha or transforms.Compose([
            transforms.Resize((self.source_size, self.source_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # 0..1 single channel
        ])
        self.transform_target_rgb = transform_target_rgb or transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])
        self.transform_target_alpha = transform_target_alpha or transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Gather pairs by walking source_root and finding matching relative paths in target_root
        self.pairs: List[Tuple[Path, Path, str]] = []
        for root, _, files in os.walk(self.source_root):
            for fname in files:
                if not _is_image_file(fname):
                    continue
                abs_source = Path(root) / fname
                rel_path = abs_source.relative_to(self.source_root)
                # category is the first part of rel_path (must exist)
                parts = rel_path.parts
                if len(parts) < 2:
                    # enforce at least category/filename relative layout
                    continue
                category = parts[0]
                if self.allowed_categories is not None and category not in self.allowed_categories:
                    continue
                abs_target = self.target_root / rel_path
                if abs_target.exists():
                    self.pairs.append((abs_source, abs_target, category))

        # build category mapping
        cats = sorted({cat for _, _, cat in self.pairs})
        self.cat2idx: Dict[str, int] = {c: i for i, c in enumerate(cats)}
        self.idx2cat: Dict[int, str] = {i: c for c, i in self.cat2idx.items()}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_rgba(self, path: Path) -> Tuple[Image.Image, Optional[Image.Image]]:
        """
        Load an image as RGBA. Returns (rgb_pil, alpha_pil) where alpha_pil is single-channel 'L' image.
        If the source has no alpha, alpha_pil will be a white mask.
        """
        img = Image.open(path).convert("RGBA")
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        alpha = a  # 'L' mode, 0..255
        return rgb, alpha

    def __getitem__(self, index: int):
        src_path, tgt_path, category = self.pairs[index]
        rel_path = src_path.relative_to(self.source_root).as_posix()

        # load source RGBA
        src_rgb_pil, src_alpha_pil = self._load_rgba(src_path)
        tgt_rgb_pil, tgt_alpha_pil = self._load_rgba(tgt_path)

        # apply transforms
        src_rgb_t = self.transform_source_rgb(src_rgb_pil)  # 3 x Hs x Ws in 0..1
        src_alpha_t = self.transform_source_alpha(src_alpha_pil)  # 1 x Hs x Ws in 0..1 (float)
        tgt_rgb_t = self.transform_target_rgb(tgt_rgb_pil)  # 3 x St x St
        tgt_alpha_t = self.transform_target_alpha(tgt_alpha_pil)  # 1 x St x St

        # ensure binary alpha (the dataset is binary-transparent)
        src_alpha_bin = (src_alpha_t > 0.5).float()
        tgt_alpha_bin = (tgt_alpha_t > 0.5).float()

        # map RGB from 0..1 to -1..1
        src_rgb_norm = src_rgb_t * 2.0 - 1.0
        tgt_rgb_norm = tgt_rgb_t * 2.0 - 1.0

        label_idx = self.cat2idx[category]

        return {
            "source_rgb": src_rgb_norm,      # Tensor[3,Hs,Ws] in -1..1
            "source_alpha": src_alpha_bin,   # Tensor[1,Hs,Ws] 0/1
            "target_rgb": tgt_rgb_norm,      # Tensor[3,St,St] in -1..1
            "target_alpha": tgt_alpha_bin,   # Tensor[1,St,St] 0/1
            "label": torch.tensor(label_idx, dtype=torch.long),
            "rel_path": rel_path
        }
