#!/usr/bin/env python3
"""
Comprehensive health check for a trained VAE checkpoint.
Run this after training to verify the model is working correctly.
"""
import argparse
from pathlib import Path
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
        h = self.decoder_input(z)
        h = h.view(B, self.enc_C, self.enc_H, self.enc_W)
        for stage in self.decoder_stages:
            h = stage(h)
        h = self.final_conv(h)
        h = h[:, :, :H, :W]
        return h

    def forward(self, x):
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.shape)
        return recon, mu, logvar


def test_checkpoint(checkpoint_path: Path, device: torch.device):
    print("=" * 70)
    print("CHECKPOINT HEALTH TEST")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}\n")

    # Load checkpoint
    print("[1/7] Loading checkpoint...")
    ckpt = torch.load(str(checkpoint_path), map_location=device)

    epoch = ckpt.get('epoch', 'unknown')
    latent_dim = ckpt.get('latent_dim', 256)
    beta = ckpt.get('beta', 0.2)

    print(f"  Epoch: {epoch}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Beta: {beta}")

    state = ckpt['model_state_dict']
    opt_state = ckpt.get('optimizer_state_dict', {})

    # Test 1: Check for NaN in checkpoint
    print("\n[2/7] Checking for NaN values...")
    nan_count = 0
    for name, param in state.items():
        if torch.isnan(param).any():
            print(f"  ❌ {name}: contains NaN")
            nan_count += 1

    if nan_count == 0:
        print("  ✓ No NaN values in checkpoint")
    else:
        print(f"  ❌ Found NaN in {nan_count} parameters")
        return False

    # Test 2: Check lazy layers exist
    print("\n[3/7] Checking lazy layers...")
    lazy_layers = [k for k in state.keys()
                   if 'fc_mu' in k or 'fc_logvar' in k or 'decoder_input' in k]

    if len(lazy_layers) == 6:
        print(f"  ✓ All 6 lazy layer parameters found")
    else:
        print(f"  ❌ Expected 6 lazy layer parameters, found {len(lazy_layers)}")
        return False

    # Test 3: Check optimizer has all parameters
    print("\n[4/7] Checking optimizer state...")
    num_model_params = sum(1 for k in state.keys()
                          if not ('running_' in k or 'num_batches_tracked' in k))

    param_groups = opt_state.get('param_groups', [])
    if param_groups:
        num_opt_params = len(param_groups[0].get('params', []))
    else:
        num_opt_params = 0

    print(f"  Model parameters: {num_model_params}")
    print(f"  Optimizer parameters: {num_opt_params}")

    if num_model_params == num_opt_params:
        print("  ✓ Optimizer has all model parameters")
    else:
        print(f"  ❌ Mismatch! {num_model_params - num_opt_params} parameters not in optimizer")
        print("  This means some layers were never trained!")
        return False

    # Test 4: Check BatchNorm running stats
    print("\n[5/7] Checking BatchNorm running statistics...")
    decoder_bn_issues = 0
    for name, param in state.items():
        if 'decoder' in name and ('running_mean' in name or 'running_var' in name):
            if torch.isnan(param).any():
                print(f"  ❌ {name}: contains NaN")
                decoder_bn_issues += 1
            elif 'running_var' in name and (param <= 0).any():
                print(f"  ❌ {name}: contains non-positive values")
                decoder_bn_issues += 1

    if decoder_bn_issues == 0:
        print("  ✓ All decoder BatchNorm stats look healthy")
    else:
        print(f"  ❌ Found issues in {decoder_bn_issues} decoder BatchNorm buffers")
        return False

    # Test 5: Load model and run inference
    print("\n[6/7] Loading model and testing inference...")
    model = BigConvVAE(
        in_channels=1,
        latent_dim=latent_dim,
        beta=beta,
        dropout=0.0
    ).to(device)

    # Initialize lazy layers with dummy input
    batch_size = 4
    n_mels = 80
    frames = 626  # 10 sec at 16kHz with hop_length=256

    dummy = torch.randn(batch_size, 1, n_mels, frames, device=device)
    with torch.no_grad():
        _ = model.encode(dummy)

    # Load state dict
    model.load_state_dict(state, strict=True)
    model.eval()

    print("  ✓ Model loaded successfully")

    # Test 6: Run inference and check for NaN
    print("\n[7/7] Running test inference...")
    test_input = torch.randn(batch_size, 1, n_mels, frames, device=device)

    with torch.no_grad():
        try:
            recon, mu, logvar = model(test_input)

            # Check for NaN in outputs
            if torch.isnan(mu).any():
                print("  ❌ Encoder output (mu) contains NaN")
                return False
            if torch.isnan(logvar).any():
                print("  ❌ Encoder output (logvar) contains NaN")
                return False
            if torch.isnan(recon).any():
                print("  ❌ Decoder output (reconstruction) contains NaN")
                return False

            # Check output statistics
            print(f"  Latent mu: min={mu.min():.4f}, max={mu.max():.4f}, mean={mu.mean():.4f}")
            print(f"  Latent logvar: min={logvar.min():.4f}, max={logvar.max():.4f}, mean={logvar.mean():.4f}")
            print(f"  Reconstruction: min={recon.min():.4f}, max={recon.max():.4f}, mean={recon.mean():.4f}")
            print("  ✓ Inference successful, no NaN in outputs")

        except Exception as e:
            print(f"  ❌ Inference failed with error: {e}")
            return False

    # All tests passed
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - CHECKPOINT IS HEALTHY!")
    print("=" * 70)
    print("\nThis checkpoint should work correctly for inference.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test a VAE checkpoint for common issues"
    )
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    success = test_checkpoint(checkpoint_path, device)

    if not success:
        print("\n" + "=" * 70)
        print("❌ CHECKPOINT HAS ISSUES")
        print("=" * 70)
        print("\nThis checkpoint will not work correctly for inference.")
        print("Please retrain with the fixed training script.")
        exit(1)


if __name__ == "__main__":
    main()
