# train_infer.py
"""
Main training & inference script.

Usage examples:

# Training:
python train_infer.py --train --source_dir ./data/source --target_dir ./data/target --checkpoint_dir ./checkpoints --epochs_per_stage 2 --device cuda

# Inference:
python train_infer.py --input ./some_input.png --source_dir ./data/source --target_dir ./data/target --checkpoint_dir ./checkpoints --num_samples 3
"""
from typing import Dict, Optional, Tuple, Any
import os
import argparse
from pathlib import Path
import math
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import interpolate
from tensorboardX import SummaryWriter
from PIL import Image
import torchvision.transforms.functional as TF

from dataset import PairedSourceTargetDataset
from models import Generator, PatchDiscriminator, PerceptualLoss
from visualization_utils import make_labeled_comparison

# -------------------------
# Helpers
# -------------------------
def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not os.path.isdir(checkpoint_dir):
        return None
    ckps = sorted(Path(checkpoint_dir).glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return str(ckps[-1]) if ckps else None


def save_ckpt(state: Dict[str, Any], checkpoint_dir: str, name: str = "latest.pt") -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, name))


# -------------------------
# Loss utilities
# -------------------------
bce = BCEWithLogitsLoss()


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    real_loss = bce(real_logits, torch.ones_like(real_logits))
    fake_loss = bce(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real_loss + fake_loss)


def generator_adv_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return bce(fake_logits, torch.ones_like(fake_logits))


# -------------------------
# Training for one stage
# -------------------------
def train_stage(
    *,
    stage_size: int,
    source_dir: str,
    target_dir: str,
    device: torch.device,
    generator: Generator,
    disc: PatchDiscriminator,
    g_opt: optim.Optimizer,
    d_opt: optim.Optimizer,
    batch_size: int = 8,
    epochs: int = 2,
    num_workers: int = 10,
    log_dir: str = "./runs",
    log_steps: int = 50,
    log_images_every: int = 200,
    z_dim: int = 128,
    lambda_adv: float = 1.0,
    lambda_l1: float = 100.0,
    lambda_alpha: float = 10.0,
    lambda_perc: float = 1.0,
    perceptual: Optional[PerceptualLoss] = None,
    checkpoint_dir: str = "./checkpoints",
) -> None:
    """
    Train the model for a given stage size (target resolution).
    The dataset will include only samples whose original target >= stage_size (handled by dataset constructor).
    """
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"stage_{stage_size}"))
    dataset = PairedSourceTargetDataset(source_root=source_dir, target_root=target_dir, target_size=stage_size)
    print(f"loaded {len(dataset)} image pairs")
    if len(dataset) == 0:
        print(f"No samples for stage {stage_size}. Skipping.")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    generator.train(); disc.train()
    if perceptual is not None:
        perceptual.to(device)

    global_step = 0
    for epoch in range(epochs):
        running_g = 0.0
        running_d = 0.0
        for batch_idx, sample in enumerate(loader):
            src_rgb = sample["source_rgb"].to(device)    # B x 3 x Hs x Ws in -1..1
            src_alpha = sample["source_alpha"].to(device)  # B x 1 x Hs x Ws (0/1)
            tgt_rgb = sample["target_rgb"].to(device)    # B x 3 x St x St in -1..1
            tgt_alpha = sample["target_alpha"].to(device)  # B x 1 x St x St
            labels = sample["label"].to(device)          # B

            b = src_rgb.shape[0]
            # build RGBA source to feed generator: concat RGB and alpha (alpha in 0..1)
            src_rgba = torch.cat([src_rgb, src_alpha], dim=1)  # B x 4 x Hs x Ws

            # --------------------
            # Discriminator update
            # --------------------
            disc.zero_grad()
            # produce fake using current generator (no grad for D)
            with torch.no_grad():
                z = torch.randn(b, z_dim, device=device)
                fake_rgb, fake_alpha = generator(src_rgba, labels, z=z)
            # compose real and fake target rgba tensors scaled to 0..1 for discriminator input
            real_rgba = torch.cat([(tgt_rgb + 1) / 2.0, tgt_alpha], dim=1)  # B x 4 x St x St
            fake_rgba = torch.cat([(fake_rgb + 1) / 2.0, fake_alpha], dim=1).detach()
            # upsample source rgba to target size to condition discriminator
            src_rgba_up = interpolate(torch.cat([ (src_rgb + 1) / 2.0, src_alpha ], dim=1), size=(stage_size, stage_size), mode='nearest')
            # concatenate for discriminator: [real_rgba || src_rgba_up] and [fake_rgba || src_rgba_up]
            real_in = torch.cat([real_rgba, src_rgba_up], dim=1)
            fake_in = torch.cat([fake_rgba, src_rgba_up], dim=1)
            real_logits = disc(real_rgba, src_rgba_up, labels)
            fake_logits = disc(fake_rgba, src_rgba_up, labels)
            d_loss = discriminator_loss(real_logits, fake_logits)
            d_loss.backward()
            d_opt.step()

            # --------------------
            # Generator update
            # --------------------
            generator.zero_grad()
            z = torch.randn(b, z_dim, device=device)
            fake_rgb, fake_alpha = generator(src_rgba, labels, z=z)
            fake_rgba = torch.cat([(fake_rgb + 1) / 2.0, fake_alpha], dim=1)
            fake_logits = disc(fake_rgba, src_rgba_up, labels)  # discriminator output on fake
            adv = generator_adv_loss(fake_logits)

            # masked L1 for RGB (only where target alpha == 1)
            mask = tgt_alpha  # B x 1 x St x St
            rgb_l1 = torch.abs(tgt_rgb - fake_rgb) * mask
            l1_loss = rgb_l1.sum() / (mask.sum().clamp_min(1.0) * 3.0)  # normalize by number of foreground pixels*channels

            # alpha loss (BCE)
            alpha_loss = BCEWithLogitsLoss()(fake_alpha, tgt_alpha)

            perc_loss = torch.tensor(0.0, device=device)
            if perceptual is not None:
                # perceptual expects rgb in -1..1
                perc_loss = perceptual(fake_rgb * mask, tgt_rgb * mask)

            g_loss = lambda_adv * adv + lambda_l1 * l1_loss + lambda_alpha * alpha_loss + lambda_perc * perc_loss
            g_loss.backward()
            g_opt.step()

            running_g += g_loss.item()
            running_d += d_loss.item()

            if global_step % log_steps == 0:
                writer.add_scalar("g_loss", g_loss.item(), global_step)
                writer.add_scalar("d_loss", d_loss.item(), global_step)
                writer.add_scalar("l1_loss", l1_loss.item(), global_step)
                writer.add_scalar("alpha_loss", alpha_loss.item(), global_step)
                if perceptual is not None:
                    writer.add_scalar("perceptual", perc_loss.item(), global_step)

            if global_step % log_images_every == 0:
                make_labeled_comparison(
                    writer=writer,
                    dataset=dataset,
                    stage_size=stage_size,
                    src_rgb=src_rgb,
                    src_alpha=src_alpha,
                    tgt_rgb=tgt_rgb,
                    tgt_alpha=tgt_alpha,
                    fake_rgb=fake_rgb,
                    fake_alpha=fake_alpha,
                    labels=labels,
                    global_step=global_step,
                    nb=4,
                )


            global_step += 1

        print(f"Stage {stage_size} epoch {epoch} g_loss={running_g/(batch_idx+1):.4f} d_loss={running_d/(batch_idx+1):.4f}")
        # save checkpoint
        ckpt = {
            "stage_size": stage_size,
            "generator_state": generator.state_dict(),
            "disc_state": disc.state_dict(),
            "g_opt": g_opt.state_dict(),
            "d_opt": d_opt.state_dict(),
            "epoch": epoch,
        }
        save_ckpt(ckpt, checkpoint_dir, name=f"stage_{stage_size}_epoch_{epoch}.pt")
        save_ckpt(ckpt, checkpoint_dir, name="latest.pt")

    writer.close()


# -------------------------
# Inference helpers
# -------------------------
def convert_single(
    generator: Generator,
    source_path: str,
    out_path: str,
    label_idx: int,
    num_samples: int,
    device: torch.device,
    out_size: int,
) -> None:
    """
    Convert a single source image into `num_samples` outputs and write as PNG(s) with alpha channel.
    """
    im = Image.open(source_path).convert("RGBA")
    # resize source to 16x16 for generator input (if not already)
    im16 = im.resize((16, 16), Image.BICUBIC)
    rgb = TF.to_tensor(im16)  # 0..1
    alpha = TF.to_tensor(im16.convert("L"))  # 0..1
    rgb = rgb * 2.0 - 1.0  # -1..1
    alpha = (alpha > 0.5).float()
    src_rgba = torch.cat([rgb, alpha], dim=0).unsqueeze(0).to(device)  # 1 x 4 x 16 x 16
    label_t = torch.tensor([label_idx], dtype=torch.long, device=device)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base, ext = os.path.splitext(out_path)
    for i in range(num_samples):
        z = torch.randn(1, generator.z_dim, device=device)
        with torch.no_grad():
            rgb_out, alpha_out = generator(src_rgba, label_t, z=z)
        rgba = torch.cat([ (rgb_out[0] + 1) / 2.0, alpha_out[0] ], dim=0).clamp(0,1)  # 4 x H x W
        # save as PNG
        out_file = f"{base}_{i}.png" if num_samples > 1 else f"{base}.png"
        save_image(rgba, out_file)
        print("Saved", out_file)


def ensure_out_dir(input_path: str) -> str:
    p = Path(input_path)
    if p.is_dir():
        return str(p.parent / (p.name + "_converted"))
    else:
        return str(p.parent / (p.stem + "_converted"))


# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--source_dir", type=str, default="./source", help="Root folder for source images (with category subfolders)")
    parser.add_argument("--target_dir", type=str, default="./target", help="Root folder for target images (mirrored structure)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs_per_stage", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--n_labels", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--input", type=str, default="", help="Input image or folder for inference")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--perceptual", action="store_true")
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--lambda_alpha", type=float, default=10.0)
    parser.add_argument("--lambda_perc", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)

    # stages (target sizes). Dataset will include only samples >= stage_size (handled by dataset)
    stages = [32, 64, 128, 256, 512]  # start at 32 (16 trivial)
    # build models for final size (largest) and reuse
    generator = Generator(n_labels=args.n_labels, z_dim=args.z_dim, base_ch=64, out_size=stages[-1], in_channels=4).to(device)
    # NOTE: create discriminator once and reuse across stages
    disc = PatchDiscriminator(in_ch=8, base=64, n_labels=args.n_labels).to(device)

    # Try to load discriminator/generator states from latest checkpoint (if present).
    latest = find_latest_checkpoint(args.checkpoint_dir)
    ckpt = None
    if latest:
        ckpt = torch.load(latest, map_location=device)
        # load discriminator state if available
        if 'disc_state' in ckpt:
            try:
                disc.load_state_dict(ckpt['disc_state'], strict=False)
                print("Loaded discriminator state from checkpoint.")
            except Exception as e:
                print("Warning: could not fully load discriminator state:", e)

    if args.train:
        # Discriminator optimizer (single optimizer reused across stages)
        d_opt = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

        perceptual = PerceptualLoss(device) if args.perceptual else None

        # Progressive training: instantiate a generator per stage
        for stage in stages:
            print("Beginning stage", stage)

            # Skip trivial 16 stage (if present) — dataset and generator expect meaningful sizes
            if stage <= 16:
                continue

            # create generator for this stage's output size
            generator = Generator(n_labels=args.n_labels, z_dim=args.z_dim, base_ch=64, out_size=stage, in_channels=4)
            generator = generator.to(device)

            # If we have a checkpoint, attempt to load matching generator params (strict=False).
            if ckpt is not None and 'generator_state' in ckpt:
                try:
                    generator.load_state_dict(ckpt['generator_state'], strict=False)
                    print(f"Partially loaded generator weights into stage-{stage} model (where keys matched).")
                except Exception as e:
                    print("Warning: could not load generator weights into stage model:", e)

            # Generator optimizer (new optimizer for the new generator instance)
            g_opt = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

            # Train this stage (train_stage expects generator and disc objects)
            train_stage(
                stage_size=stage,
                source_dir=args.source_dir,
                target_dir=args.target_dir,
                device=device,
                generator=generator,
                disc=disc,
                g_opt=g_opt,
                d_opt=d_opt,
                batch_size=args.batch_size,
                epochs=args.epochs_per_stage,
                num_workers=args.num_workers,
                log_dir=args.log_dir,
                log_steps=50,
                log_images_every=200,
                z_dim=args.z_dim,
                lambda_adv=args.lambda_adv,
                lambda_l1=args.lambda_l1,
                lambda_alpha=args.lambda_alpha,
                lambda_perc=args.lambda_perc,
                perceptual=perceptual,
                checkpoint_dir=args.checkpoint_dir,
            )

            # After finishing the stage, save the generator for this stage explicitly
            save_ckpt({
                'stage_size': stage,
                'generator_state': generator.state_dict(),
                'disc_state': disc.state_dict(),
            }, args.checkpoint_dir, name=f'stage_{stage}.pt')

        print("Training complete.")
    else:
        # inference
        if not args.input:
            print("No input provided. Use --input <image_or_folder> or --train to train.")
            return
        # build a mapping of categories from dataset (to know label indices)
        ds = PairedSourceTargetDataset(source_root=args.source_dir, target_root=args.target_dir, target_size=512)
        cat2idx = ds.cat2idx
        if os.path.isdir(args.input):
            out_root = ensure_out_dir(args.input)
            for root, _, files in os.walk(args.input):
                rel_root = Path(root).relative_to(args.input)
                for fname in files:
                    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        continue
                    in_path = os.path.join(root, fname)
                    # compute category from rel path same as dataset rule:
                    rel_path = Path(in_path).relative_to(args.input).as_posix()
                    parts = rel_path.split("/")
                    if len(parts) < 2:
                        print("Skipping file not under category subfolder:", in_path)
                        continue
                    category = parts[0]
                    label_idx = cat2idx.get(category, 0)
                    out_dir = Path(out_root) / Path(rel_root).parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    base = Path(fname).stem
                    out_path = str(out_dir / base)
                    convert_single(
                        generator,
                        in_path,
                        out_path,
                        label_idx,
                        args.num_samples,
                        device,
                        out_size=generator.ups[-1].conv.out_channels if generator.ups else 512
                    )
            print("Folder conversion finished.")
        else:
            # single file
            in_path = args.input
            # determine category by parent folder name: expecting <something>/<category>/<file>
            rel = Path(in_path).relative_to(Path(in_path).parents[1]) if len(Path(in_path).parts) >= 2 else None
            # fallback: try to infer category name from the path under source_dir
            try:
                rel_to_source = Path(in_path).relative_to(args.source_dir)
                category = rel_to_source.parts[0]
            except Exception:
                # fallback to parent folder
                category = Path(in_path).parent.name
            ds = PairedSourceTargetDataset(source_root=args.source_dir, target_root=args.target_dir, target_size=512)
            label_idx = ds.cat2idx.get(category, 0)
            out_root = ensure_out_dir(in_path)
            os.makedirs(out_root, exist_ok=True)
            base = Path(in_path).stem
            out_path = os.path.join(out_root, base)
            convert_single(generator, in_path, out_path, label_idx, args.num_samples, device, out_size=512)
            print("Saved outputs to", out_root)


if __name__ == "__main__":
    main()
