#!/usr/bin/env python3
import os
import glob
import math
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# ============================================================
# Dataset
# ============================================================

class AudioDataset(Dataset):
    """
    Loads mono WAV files from a directory, crops/pads to fixed duration,
    and converts to log-mel spectrograms.
    """
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        duration_sec: float = 10.0,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        f_min: float = 0.0,
        f_max: float = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration_sec)

        self.files: List[str] = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No .wav files found under {root_dir}")

        # Transforms to mel-spectrogram then log amplitude
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        self.to_db = AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.files)

    def _load_and_fix_length(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # (channels, time)
        # Convert to mono
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        # Resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        # Crop or pad
        if wav.size(1) > self.target_len:
            wav = wav[:, :self.target_len]
        elif wav.size(1) < self.target_len:
            pad_len = self.target_len - wav.size(1)
            wav = F.pad(wav, (0, pad_len))
        return wav

    def __getitem__(self, idx):
        path = self.files[idx]
        wav = self._load_and_fix_length(path)  # (1, T)
        with torch.no_grad():
            mel = self.mel(wav)  # (1, n_mels, T_frames)
            log_mel = self.to_db(mel + 1e-6)  # (1, n_mels, T_frames)
        # Normalize per-sample (optional; you can also do dataset-wide norm)
        mean = log_mel.mean()
        std = log_mel.std() + 1e-8
        log_mel = (log_mel - mean) / std
        return log_mel  # shape (1, n_mels, T_frames)


# ============================================================
# VAE Model (Conv encoder/decoder on log-mel)
# ============================================================

class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, beta=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder: Conv2d stack, downsample in time & frequency
        # Input shape: (B, 1, n_mels, T_frames)
        enc_channels = [32, 64, 128, 256]
        self.encoder_convs = nn.ModuleList()
        in_ch = in_channels

        # Each block halves freq and time dimensions (approx) using stride (2,2)
        for out_ch in enc_channels:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.encoder_convs.append(block)
            in_ch = out_ch

        # We won't know spatial dims (H, W) until we see data, so we will
        # compute them in a "lazy" way in forward by caching.
        self.enc_out_dim = None  # flattened dimension after conv encoder

        # These linear layers will be initialized lazily once enc_out_dim is known
        self.fc_mu = None
        self.fc_logvar = None

        # Decoder: linear -> conv transpose stack (mirrors encoder)
        self.decoder_input = None
        self.decoder_convs = nn.ModuleList()
        dec_channels = list(reversed(enc_channels))

        for i in range(len(dec_channels) - 1):
            in_ch = dec_channels[i]
            out_ch = dec_channels[i + 1]
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.decoder_convs.append(block)

        # Final layer back to 1 channel
        self.final_conv = nn.ConvTranspose2d(
            dec_channels[-1], 1, kernel_size=4, stride=2, padding=1
        )

    def encode(self, x):
        """
        x: (B, 1, H, W)
        returns: mu, logvar (B, latent_dim)
        """
        batch_size = x.size(0)
        h = x
        for block in self.encoder_convs:
            h = block(h)

        # Flatten
        h = h.view(batch_size, -1)

        # Lazy init of fc layers if needed
        if self.enc_out_dim is None:
            self.enc_out_dim = h.size(1)
            self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim).to(h.device)
            self.fc_logvar = nn.Linear(self.enc_out_dim, self.latent_dim).to(h.device)

            # Decoder input fully-connected: latent -> enc_out_dim
            self.decoder_input = nn.Linear(self.latent_dim, self.enc_out_dim).to(h.device)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, h

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_shape):
        """
        z: (B, latent_dim)
        target_shape: (B, 1, H, W) for final crop (we only need H,W)

        Reverse the encoder operations using ConvTranspose2d stacks.
        """
        B = z.size(0)
        H, W = target_shape[2], target_shape[3]

        h = self.decoder_input(z)
        # Unflatten according to enc_out_dim and conv stack
        # We know enc_out_dim = C * H_enc * W_enc
        # To compute H_enc, W_enc we reuse how many downsamples we did
        num_downsamples = len(self.encoder_convs)
        H_enc = H // (2 ** num_downsamples)
        W_enc = W // (2 ** num_downsamples)
        # Safety: avoid zero
        H_enc = max(1, H_enc)
        W_enc = max(1, W_enc)
        C_enc = self.enc_out_dim // (H_enc * W_enc)

        h = h.view(B, C_enc, H_enc, W_enc)

        # Apply decoder conv-transpose blocks
        for block in self.decoder_convs:
            h = block(h)

        h = self.final_conv(h)

        # h may be slightly larger than target due to stride/padding;
        # crop to match original H, W
        h = h[:, :, :H, :W]
        return h

    def forward(self, x):
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.shape)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Beta-VAE loss: recon + beta * KL
        Recon: L1 loss on log-mel (you can use L2 if preferred).
        """
        recon_loss = F.l1_loss(recon_x, x, reduction="mean")

        # KL divergence between q(z|x) and N(0,I)
        # 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl
        return loss, recon_loss, kl


# ============================================================
# Training Loop
# ============================================================

def train(
    data_dir: str,
    latent_dim: int = 128,
    batch_size: int = 16,
    num_epochs: int = 50,
    lr: float = 1e-4,
    beta: float = 0.1,
    num_workers: int = 4,
    device: str = None,
    sample_rate: int = 16000,
    duration_sec: float = 10.0,
    n_mels: int = 80,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = AudioDataset(
        root_dir=data_dir,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        n_mels=n_mels,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Grab one batch to infer input shape
    x0 = next(iter(dataloader))
    B, C, H, W = x0.shape
    print(f"Example batch shape (log-mel): {x0.shape}")

    model = ConvVAE(in_channels=C, latent_dim=latent_dim, beta=beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0

        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)

            recon, mu, logvar = model(x)
            loss, recon_loss, kl = model.loss_function(recon, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl.item()
            global_step += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                avg_recon = running_recon / 100
                avg_kl = running_kl / 100
                print(
                    f"Epoch [{epoch}/{num_epochs}] "
                    f"Step [{batch_idx+1}/{len(dataloader)}] "
                    f"Loss: {avg_loss:.4f} Recon: {avg_recon:.4f} KL: {avg_kl:.4f}"
                )
                running_loss = running_recon = running_kl = 0.0

        # End of epoch logging
        if len(dataloader) > 0:
            epoch_loss = running_loss / max(1, len(dataloader) % 100)
            print(f"End of Epoch {epoch}: last-window loss = {epoch_loss:.4f}")

        # Save checkpoint occasionally
        ckpt_path = f"vae_latent{latent_dim}_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "latent_dim": latent_dim,
                "beta": beta,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvVAE on short audio (log-mel).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing .wav files.")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Latent dimension of VAE.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Weight on KL term in VAE loss.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=10.0,
                        help="Fixed duration for clips (crop/pad).")
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda', 'cpu', or None to auto-select.")
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        data_dir=args.data_dir,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        beta=args.beta,
        num_workers=args.num_workers,
        device=args.device,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        n_mels=args.n_mels,
    )


if __name__ == "__main__":
    main()
