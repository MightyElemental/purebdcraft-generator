# visualization_utils.py
from typing import Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: Optional[ImageFont.ImageFont]) -> Tuple[int, int]:
    """Robust text width/height measurement across Pillow versions."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except Exception:
        pass
    if font is not None:
        try:
            return font.getsize(text)
        except Exception:
            pass
    try:
        return draw.textsize(text, font=font)
    except Exception:
        return int(len(text) * 6), 12


def _tensor_to_rgba_pil(rgb: torch.Tensor, alpha: Optional[torch.Tensor]) -> Image.Image:
    """
    Convert tensors to a PIL RGBA image.
    rgb: Tensor (3, H, W) in 0..1 (not -1..1).
    alpha: Tensor (1, H, W) in 0..1 or None (then opaque).
    Returns PIL.Image in RGBA.
    """
    # clamp and convert
    rgb_np = (rgb.clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # H x W x 3
    if alpha is None:
        a_np = np.full((rgb_np.shape[0], rgb_np.shape[1], 1), 255, dtype=np.uint8)
    else:
        a_np = (alpha.clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # H x W x 1
    rgba = np.concatenate([rgb_np, a_np], axis=2)  # H x W x 4
    return Image.fromarray(rgba, mode="RGBA")


def _prepare_image_tensor_for_pil(tensor: torch.Tensor, alpha: Optional[torch.Tensor], target_size: int) -> Image.Image:
    """
    Normalize and upsample tensors to target_size using NEAREST neighbor for both RGB and alpha.
    Accepts rgb in either -1..1 or 0..1 and converts to 0..1.
    """
    # detect range and convert to 0..1 if necessary
    tmax = float(tensor.max().detach().cpu())
    tmin = float(tensor.min().detach().cpu())
    if tmax > 1.1 or tmin < -0.9:
        t = tensor.clamp(0.0, 1.0)
    else:
        if tmax > 1.0 or tmin < -0.5:
            # assume -1..1
            t = (tensor + 1.0) / 2.0
        else:
            t = tensor

    # ensure shape B x 3 x H x W for interpolation; we expect single instance so handle dims
    if t.dim() == 3:
        t_rgb = t.unsqueeze(0)  # 1 x 3 x H x W
    else:
        t_rgb = t

    # UPSAMPLE RGB using NEAREST (preserves pixels)
    t_rgb_up = F.interpolate(t_rgb, size=(target_size, target_size), mode='nearest')[0]

    # Alpha: upsample with NEAREST as well
    if alpha is None:
        alpha_up = None
    else:
        a = alpha
        if a.dim() == 3:
            a = a.unsqueeze(0)
        a_up = F.interpolate(a, size=(target_size, target_size), mode='nearest')[0]
        alpha_up = a_up

    return _tensor_to_rgba_pil(t_rgb_up, alpha_up)


def make_labeled_comparison(
    writer: SummaryWriter,
    dataset,
    stage_size: int,
    src_rgb: torch.Tensor,
    src_alpha: torch.Tensor,
    tgt_rgb: torch.Tensor,
    tgt_alpha: torch.Tensor,
    fake_rgb: torch.Tensor,
    fake_alpha: torch.Tensor,
    labels: torch.Tensor,
    global_step: int,
    cols: int = 6,
    img_px: int = 256,
    tag: str = "comparison_labeled",
) -> None:
    """
    Build a labeled comparison and write to TensorBoard.

    - Creates a grid with `cols` columns (default 6).
    - Each column is 3 rows: source, target, generated, followed by a text label row.
    - All images are scaled to img_px x img_px (default 512) BEFORE adding labels.
    - dataset: used to get idx->category mapping (dataset.idx2cat or dataset.cat2idx)
    - src_rgb, tgt_rgb, fake_rgb: tensors B x 3 x H x W in -1..1 (or 0..1)
    - src_alpha, tgt_alpha, fake_alpha: tensors B x 1 x H x W in 0..1 (or None)
    - labels: B (long)
    """

    device = src_rgb.device
    batch = src_rgb.shape[0]
    cols = min(cols, batch)
    if cols == 0:
        return

    # build idx->category mapping
    idx2cat: Dict[int, str] = {}
    if hasattr(dataset, "idx2cat"):
        idx2cat = getattr(dataset, "idx2cat")
    elif hasattr(dataset, "cat2idx"):
        cat2idx = getattr(dataset, "cat2idx")
        idx2cat = {v: k for k, v in cat2idx.items()}

    # choose a readable font size proportional to the image pixel size
    font_size = max(18, img_px // 16)  # adjust divisor to tune size (smaller -> larger font)
    font = None
    # try common locations for a TTF (DejaVu is commonly available with Pillow)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            # final fallback to default (may be small)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

    # prepare PIL triplet columns
    columns = []
    for i in range(cols):
        # compose PIL images, scaled to img_px
        src_pil = _prepare_image_tensor_for_pil(src_rgb[i], src_alpha[i] if src_alpha is not None else None, img_px)
        tgt_pil = _prepare_image_tensor_for_pil(tgt_rgb[i], tgt_alpha[i] if tgt_alpha is not None else None, img_px)
        fake_pil = _prepare_image_tensor_for_pil(fake_rgb[i].detach(), fake_alpha[i].detach() if fake_alpha is not None else None, img_px)

        # create column: vertical stack (source, target, generated), white background
        column_h = img_px * 3
        label_area_h = max(32, img_px // 16)  # reasonable label area
        total_h = column_h + label_area_h
        col_img = Image.new("RGBA", (img_px, total_h), (255, 255, 255, 255))

        # paste images with no overlay
        col_img.paste(src_pil, (0, 0))
        col_img.paste(tgt_pil, (0, img_px))
        col_img.paste(fake_pil, (0, img_px * 2))

        # draw label in the label area (centered)
        draw = ImageDraw.Draw(col_img)
        label_idx = int(labels[i].item()) if isinstance(labels, torch.Tensor) else int(labels[i])
        label_text = idx2cat.get(label_idx, str(label_idx))
        tw, th = _measure_text(draw, label_text, font)
        text_x = max(0, (img_px - tw) // 2)
        text_y = column_h + (label_area_h - th) // 2
        text_color = (0, 0, 0, 255)
        draw.text((text_x, text_y), label_text, font=font, fill=text_color)

        columns.append(col_img)

    # compose final canvas horizontally
    final_w = img_px * cols
    final_h = columns[0].height
    canvas = Image.new("RGBA", (final_w, final_h), (255, 255, 255, 255))
    for idx, col in enumerate(columns):
        canvas.paste(col, (idx * img_px, 0))

    # convert to tensor CHW in 0..1
    arr = np.array(canvas).astype(np.float32) / 255.0  # H x W x 4
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # 4 x H x W

    writer.add_image(f"{tag}_stage{stage_size}", tensor, global_step)
