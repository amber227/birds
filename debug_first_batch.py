#!/usr/bin/env python3
"""
Simulate the first training batch to understand why decoder BatchNorm gets NaN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2d(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act(out)
        return out


class BigConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=256, beta=0.1, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        enc_channels = [64, 128, 256, 512, 512]
        self.encoder_stages = nn.ModuleList()
        prev_ch = in_channels
        for out_ch in enc_channels:
            stage = nn.Sequential(
                nn.Conv2d(prev_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock2d(out_ch, dropout=dropout),
            )
            self.encoder_stages.append(stage)
            prev_ch = out_ch

        self.enc_out_dim = None
        self.enc_C = None
        self.enc_H = None
        self.enc_W = None

        self.fc_mu = None
        self.fc_logvar = None
        self.decoder_input = None

        self.decoder_stages = nn.ModuleList()
        dec_channels = list(reversed(enc_channels))
        for i in range(len(dec_channels) - 1):
            in_ch = dec_channels[i]
            out_ch = dec_channels[i + 1]
            stage = nn.Sequential(
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
                ResBlock2d(out_ch, dropout=dropout),
            )
            self.decoder_stages.append(stage)

        self.final_conv = nn.ConvTranspose2d(
            dec_channels[-1], 1, kernel_size=4, stride=2, padding=1
        )

    def encode(self, x):
        B = x.size(0)
        h = x
        for stage in self.encoder_stages:
            h = stage(h)

        if self.enc_out_dim is None:
            _, C_enc, H_enc, W_enc = h.shape
            self.enc_C = C_enc
            self.enc_H = H_enc
            self.enc_W = W_enc
            self.enc_out_dim = C_enc * H_enc * W_enc
            device = h.device
            print(f"  Creating lazy layers: enc_out_dim={self.enc_out_dim}, latent_dim={self.latent_dim}")
            print(f"  Encoder output shape: C={C_enc}, H={H_enc}, W={W_enc}")
            self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim).to(device)
            self.fc_logvar = nn.Linear(self.enc_out_dim, self.latent_dim).to(device)
            self.decoder_input = nn.Linear(self.latent_dim, self.enc_out_dim).to(device)

        h_flat = h.view(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar, h_flat

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_shape):
        B = z.size(0)
        H, W = target_shape[2], target_shape[3]
        print(f"\n  Decode input z shape: {z.shape}")
        print(f"  Target output shape: H={H}, W={W}")

        h = self.decoder_input(z)
        print(f"  After decoder_input: shape={h.shape}, has_nan={torch.isnan(h).any()}, min={h.min():.6f}, max={h.max():.6f}")

        h = h.view(B, self.enc_C, self.enc_H, self.enc_W)
        print(f"  After reshape: shape={h.shape}")

        for i, stage in enumerate(self.decoder_stages):
            h = stage(h)
            has_nan = torch.isnan(h).any()
            print(f"  After decoder_stage[{i}]: shape={h.shape}, has_nan={has_nan}", end="")
            if not has_nan:
                print(f", min={h.min():.6f}, max={h.max():.6f}")
            else:
                print(" ❌ NaN detected!")
                # Check which layer caused NaN
                return None

        h = self.final_conv(h)
        has_nan = torch.isnan(h).any()
        print(f"  After final_conv: shape={h.shape}, has_nan={has_nan}", end="")
        if not has_nan:
            print(f", min={h.min():.6f}, max={h.max():.6f}")
        else:
            print(" ❌ NaN detected!")

        h = h[:, :, :H, :W]
        print(f"  After crop: shape={h.shape}")
        return h

    def forward(self, x):
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.shape)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.l1_loss(recon_x, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl
        return loss, recon_loss, kl


def main():
    print("=" * 70)
    print("SIMULATING FIRST TRAINING BATCH")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Create model (same parameters as training)
    latent_dim = 256
    beta = 0.2
    dropout = 0.1

    print("Creating model...")
    model = BigConvVAE(
        in_channels=1,
        latent_dim=latent_dim,
        beta=beta,
        dropout=dropout,
    ).to(device)
    model.train()
    print("✓ Model created in training mode\n")

    # Create dummy input (same as dataset: normalized log-mel)
    # Shape: (batch_size, 1, n_mels, T_frames)
    batch_size = 64
    n_mels = 80
    sample_rate = 16000
    duration_sec = 10.0
    hop_length = 256

    target_len = int(sample_rate * duration_sec)
    frames = 1 + target_len // hop_length

    print(f"Creating dummy input:")
    print(f"  Batch size: {batch_size}")
    print(f"  Shape: ({batch_size}, 1, {n_mels}, {frames})")

    # Use randn to simulate normalized log-mel (mean=0, std=1)
    x = torch.randn(batch_size, 1, n_mels, frames, device=device)
    print(f"  Input stats: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std():.6f}\n")

    print("=" * 70)
    print("RUNNING FIRST FORWARD PASS")
    print("=" * 70)

    print("\n[ENCODER]")
    recon, mu, logvar = model(x)

    print("\n[RESULTS]")
    print(f"  Latent mu: min={mu.min():.6f}, max={mu.max():.6f}, mean={mu.mean():.6f}")
    print(f"  Latent logvar: min={logvar.min():.6f}, max={logvar.max():.6f}, mean={logvar.mean():.6f}")

    if recon is not None:
        has_nan = torch.isnan(recon).any()
        print(f"  Reconstruction: shape={recon.shape}, has_nan={has_nan}", end="")
        if not has_nan:
            print(f", min={recon.min():.6f}, max={recon.max():.6f}")

            # Compute loss
            loss, recon_loss, kl = model.loss_function(recon, x, mu, logvar)
            print(f"\n[LOSS]")
            print(f"  Reconstruction loss: {recon_loss.item():.6f}")
            print(f"  KL loss: {kl.item():.6f}")
            print(f"  Total loss: {loss.item():.6f}")

            print("\n✓ First forward pass completed successfully!")
        else:
            print(" ❌")
            print("\n❌ Reconstruction contains NaN!")
    else:
        print("\n❌ Decoder returned None (NaN detected mid-way)")

    # Check BatchNorm running stats
    print("\n=" * 70)
    print("CHECKING BATCHNORM RUNNING STATS AFTER FIRST BATCH")
    print("=" * 70)

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if 'decoder' in name:
                has_nan_mean = torch.isnan(module.running_mean).any()
                has_nan_var = torch.isnan(module.running_var).any()
                print(f"{name}:")
                print(f"  running_mean: has_nan={has_nan_mean}", end="")
                if not has_nan_mean:
                    print(f", range=[{module.running_mean.min():.6f}, {module.running_mean.max():.6f}]")
                else:
                    print(" ❌")
                print(f"  running_var: has_nan={has_nan_var}", end="")
                if not has_nan_var:
                    print(f", range=[{module.running_var.min():.6f}, {module.running_var.max():.6f}]")
                else:
                    print(" ❌")


if __name__ == "__main__":
    main()
