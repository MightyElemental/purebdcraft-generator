# visualization_utils.py
from typing import Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

def _scale_to_512(img: Image.Image) -> Image.Image:
    """Scales an image to 512x512 using nearest-neighbor interpolation."""
    return img.resize((128,128), Image.NEAREST)

def _tensor_to_pil_rgba(t: torch.Tensor) -> Image.Image:
    """
    Convert a 4xHxW torch tensor in range [0,1] to a PIL RGBA image.

    Args:
        t: Tensor shape (4, H, W) values in 0..1
    Returns:
        PIL.Image in mode 'RGBA'
    """
    arr = (t.clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # H x W x 4
    img = Image.fromarray(arr, mode="RGBA")
    return _scale_to_512(img)


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: Optional[ImageFont.ImageFont]) -> (int, int):
    """
    Measure text width and height in a way that is robust across Pillow versions.

    Tries the following (in order):
      - draw.textbbox (Pillow >= 8.0)
      - font.getsize
      - draw.textsize (older API)
    Returns (width, height).
    """
    try:
        # modern Pillow: returns (left, top, right, bottom)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return int(w), int(h)
    except Exception:
        pass

    if font is not None:
        try:
            w, h = font.getsize(text)
            return int(w), int(h)
        except Exception:
            pass

    try:
        w, h = draw.textsize(text, font=font)
        return int(w), int(h)
    except Exception:
        # fallback conservative estimate
        return int(len(text) * 6), 12


def _draw_label_on_image(image: Image.Image, label_text: str, font: Optional[ImageFont.ImageFont] = None) -> None:
    """
    Overlay a semi-opaque bar and label text at the top of the image.

    The function mutates the provided PIL image.

    Args:
        image: PIL RGBA image to annotate (mutated in place)
        label_text: text to draw
        font: optional PIL font (fallback to default if None)
    """
    draw = ImageDraw.Draw(image, "RGBA")
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    tw, th = _measure_text(draw, label_text, font)
    pad = 6
    rect_h = th + pad * 2
    # semi-opaque black rectangle across top for legibility
    draw.rectangle([(0, 0), (image.width, rect_h)], fill=(0, 0, 0, 160))
    text_x = 6
    text_y = pad
    # outline strokes for contrast
    outline_color = (0, 0, 0, 255)
    text_color = (255, 255, 255, 255)
    # Draw outline (4 directions)
    draw.text((text_x - 1, text_y), label_text, font=font, fill=outline_color)
    draw.text((text_x + 1, text_y), label_text, font=font, fill=outline_color)
    draw.text((text_x, text_y - 1), label_text, font=font, fill=outline_color)
    draw.text((text_x, text_y + 1), label_text, font=font, fill=outline_color)
    # main text
    draw.text((text_x, text_y), label_text, font=font, fill=text_color)


def _make_triplet_image(
    src_rgba: Image.Image,
    real_rgba: Image.Image,
    fake_rgba: Image.Image,
    label_text: str,
    font: Optional[ImageFont.ImageFont] = None,
) -> Image.Image:
    """
    Create a single horizontal triplet image [src | real | fake] and overlay the label text.

    Args:
        src_rgba, real_rgba, fake_rgba: PIL RGBA images (all same size)
        label_text: category string
        font: optional PIL font
    Returns:
        PIL RGBA image of width = 3 * W and height = H (with label overlay)
    """
    W, H = src_rgba.size
    trip = Image.new("RGBA", (W * 3, H), (0, 0, 0, 0))
    trip.paste(src_rgba, (0, 0))
    trip.paste(real_rgba, (W, 0))
    trip.paste(fake_rgba, (2 * W, 0))
    _draw_label_on_image(trip, label_text, font=font)
    return trip

def make_comparison(
    writer: SummaryWriter,
    stage_size: int,
    src_rgb: torch.Tensor,
    tgt_rgb: torch.Tensor,
    tgt_alpha: torch.Tensor,
    fake_rgb: torch.Tensor,
    fake_alpha: torch.Tensor,
    global_step: int,
    nb: int = 4,
) -> None:
    # log small grid: upsampled source, real rgba, fake rgba (first 4 examples)
    up_src = F.interpolate((src_rgb[:nb] + 1) / 2.0, size=(stage_size, stage_size), mode='nearest')
    real_rgba_grid = torch.cat([ (tgt_rgb[:nb] + 1) / 2.0, tgt_alpha[:nb] ], dim=1)
    fake_rgba_grid = torch.cat([ (fake_rgb[:nb].detach() + 1) / 2.0, fake_alpha[:nb].detach() ], dim=1)
    grid = make_grid(torch.cat([up_src, real_rgba_grid, fake_rgba_grid], dim=0), nrow=nb)
    writer.add_image("comparison", grid, global_step)


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
    nb: int = 4,
    tag: str = "comparison_labeled",
) -> None:
    """
    Build a labeled comparison image and write to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        dataset: dataset instance providing cat2idx or idx2cat mapping (optional)
        stage_size: visualisation target resolution (int)
        src_rgb: B x 3 x Hs x Ws (in -1..1)
        src_alpha: B x 1 x Hs x Ws (0/1)
        tgt_rgb: B x 3 x St x St (in -1..1)
        tgt_alpha: B x 1 x St x St (0/1)
        fake_rgb: B x 3 x St x St (in -1..1)
        fake_alpha: B x 1 x St x St (0..1)
        labels: B long tensor with label indices
        global_step: training step for TB writer
        nb: number of examples to visualize (<= batch_size)
        tag: TB tag name
    """
    device = src_rgb.device
    b = src_rgb.shape[0]
    nb = min(nb, b)

    # upsample source to target resolution (bilinear for rgb, nearest for alpha)
    up_src_rgb = F.interpolate((src_rgb[:nb] + 1) / 2.0, size=(stage_size, stage_size), mode="bilinear", align_corners=False)
    up_src_alpha = F.interpolate(src_alpha[:nb], size=(stage_size, stage_size), mode="nearest")
    up_src_rgba = torch.cat([up_src_rgb, up_src_alpha], dim=1)  # nb x 4 x S x S

    real_rgba = torch.cat([(tgt_rgb[:nb] + 1) / 2.0, tgt_alpha[:nb]], dim=1)
    fake_rgba = torch.cat([(fake_rgb[:nb].detach() + 1) / 2.0, fake_alpha[:nb].detach()], dim=1)

    # build idx->category mapping
    idx2cat: Dict[int, str] = {}
    if hasattr(dataset, "idx2cat"):
        idx2cat = getattr(dataset, "idx2cat")
    elif hasattr(dataset, "cat2idx"):
        cat2idx = getattr(dataset, "cat2idx")
        idx2cat = {v: k for k, v in cat2idx.items()}

    # attempt to load default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pil_triplets = []
    for i in range(nb):
        src_pil = _tensor_to_pil_rgba(up_src_rgba[i])
        real_pil = _tensor_to_pil_rgba(real_rgba[i])
        fake_pil = _tensor_to_pil_rgba(fake_rgba[i])
        label_idx = int(labels[i].item())
        label_text = idx2cat.get(label_idx, str(label_idx))
        trip = _make_triplet_image(src_pil, real_pil, fake_pil, label_text=label_text, font=font)
        pil_triplets.append(trip)

    # compose horizontally
    total_w = pil_triplets[0].width * nb
    total_h = pil_triplets[0].height
    canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    for i, im in enumerate(pil_triplets):
        canvas.paste(im, (i * im.width, 0))

    # convert canvas to tensor CHW in 0..1
    canvas_arr = np.array(canvas).astype(np.float32) / 255.0  # H x W x 4
    canvas_tensor = torch.from_numpy(canvas_arr).permute(2, 0, 1)  # 4 x H x W

    writer.add_image(tag + f"_stage{stage_size}", canvas_tensor, global_step)