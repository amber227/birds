#!/usr/bin/env python3
"""
Diagnostic script to check VAE model outputs before audio conversion.
This helps identify where the problem occurs in the pipeline.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


# ============================================================
# MODEL DEFINITION (same as reencode_big_vae.py)
# ============================================================

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


# ============================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================

def print_tensor_stats(name: str, tensor: torch.Tensor, check_duplicates: bool = False):
    """Print statistics about a tensor."""
    print(f"\n{name}:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")

    # Check for NaN or Inf
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan:
        print(f"  ⚠️  WARNING: Contains NaN values!")
    if has_inf:
        print(f"  ⚠️  WARNING: Contains Inf values!")

    # Check for constant values
    unique_values = torch.unique(tensor)
    print(f"  Unique values: {len(unique_values)}")
    if len(unique_values) == 1:
        print(f"  ⚠️  WARNING: All values are identical: {unique_values[0].item():.6f}")
    elif len(unique_values) < 10:
        print(f"  ⚠️  WARNING: Very few unique values: {unique_values.tolist()}")

    # Sample some values
    flat = tensor.flatten()
    sample_size = min(20, len(flat))
    sample_indices = torch.linspace(0, len(flat) - 1, sample_size).long()
    sample_values = flat[sample_indices]
    print(f"  Sample values: {sample_values[:10].tolist()}")


def diagnose_file(
    model_path: Path,
    audio_path: Path,
    device: torch.device,
    sample_rate: int = 16000,
    duration_sec: float = 10.0,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    dropout: float = 0.0,
):
    print("=" * 70)
    print("VAE DIAGNOSTIC TOOL")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Audio: {audio_path}")
    print(f"Device: {device}")
    print("=" * 70)

    # Load model
    print("\n[1/6] Loading model...")
    ckpt = torch.load(str(model_path), map_location=device)
    latent_dim = ckpt.get("latent_dim", 256)
    beta = ckpt.get("beta", 0.2)

    model = BigConvVAE(
        in_channels=1,
        latent_dim=latent_dim,
        beta=beta,
        dropout=dropout,
    ).to(device)

    # Initialize lazy layers
    target_len = int(sample_rate * duration_sec)
    frames = 1 + target_len // hop_length
    dummy = torch.zeros(1, 1, n_mels, frames, device=device)
    with torch.no_grad():
        _ = model.encode(dummy)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"  Latent dim: {latent_dim}")
    print(f"  Beta: {beta}")
    print(f"  ✓ Model loaded successfully")

    # Load and preprocess audio
    print("\n[2/6] Loading and preprocessing audio...")
    wav, sr = torchaudio.load(str(audio_path))
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.squeeze(0)

    # Pad or trim
    T = wav.numel()
    if T > target_len:
        wav = wav[:target_len]
    elif T < target_len:
        pad = target_len - T
        wav = torch.cat([wav, torch.zeros(pad, dtype=wav.dtype)], dim=0)

    wav = wav.unsqueeze(0).to(device)  # (1, T)
    print_tensor_stats("Input waveform", wav)

    # Create mel spectrogram
    print("\n[3/6] Computing mel spectrogram...")
    mel = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0.0,
        f_max=None,
        center=True,
        pad_mode="reflect",
        power=2.0,
    ).to(device)
    to_db = AmplitudeToDB(stype="power").to(device)

    with torch.no_grad():
        mel_spec = mel(wav)  # (1, n_mels, T_frames)
        print_tensor_stats("Mel spectrogram (power)", mel_spec)

        log_mel = to_db(mel_spec + 1e-6)  # (1, n_mels, T_frames)
        print_tensor_stats("Log-mel (dB)", log_mel)

        # Normalize (same as training)
        mean = log_mel.mean()
        std = log_mel.std() + 1e-8
        print(f"\n  Normalization stats:")
        print(f"    Mean: {mean.item():.6f}")
        print(f"    Std: {std.item():.6f}")

        log_mel_norm = (log_mel - mean) / std
        print_tensor_stats("Normalized log-mel", log_mel_norm)

        # Run through VAE
        print("\n[4/6] Running VAE encoder...")
        x = log_mel_norm.unsqueeze(0)  # (1, 1, n_mels, T_frames)
        mu, logvar, _ = model.encode(x)
        print_tensor_stats("Latent mu", mu)
        print_tensor_stats("Latent logvar", logvar)

        print("\n[5/6] Running VAE decoder...")
        z = model.reparameterize(mu, logvar)
        print_tensor_stats("Latent z (sampled)", z)

        recon = model.decode(z, x.shape)  # (1, 1, n_mels, T_frames)
        recon = recon.squeeze(0)  # (1, n_mels, T_frames)
        print_tensor_stats("Reconstructed normalized log-mel", recon)

        # Denormalize
        print("\n[6/6] Denormalizing reconstruction...")
        recon_log_mel = recon * std + mean
        print_tensor_stats("Reconstructed log-mel (dB)", recon_log_mel)

        # Compare input vs output
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)

        diff = (log_mel - recon_log_mel).abs()
        print_tensor_stats("Absolute difference (input - output)", diff)

        mse = ((log_mel - recon_log_mel) ** 2).mean()
        print(f"\nMean Squared Error: {mse.item():.6f}")

        # Check if reconstruction looks reasonable
        print("\n" + "=" * 70)
        print("DIAGNOSIS")
        print("=" * 70)

        recon_unique = torch.unique(recon_log_mel)
        if len(recon_unique) == 1:
            print("❌ PROBLEM FOUND: Reconstruction has only one unique value!")
            print(f"   The model is outputting constant: {recon_unique[0].item():.6f}")
        elif len(recon_unique) < 100:
            print(f"⚠️  WARNING: Reconstruction has very few unique values ({len(recon_unique)})")
            print("   The model may not be working correctly.")
        elif torch.isnan(recon_log_mel).any():
            print("❌ PROBLEM FOUND: Reconstruction contains NaN values!")
        elif torch.isinf(recon_log_mel).any():
            print("❌ PROBLEM FOUND: Reconstruction contains Inf values!")
        elif mse.item() < 0.01:
            print("⚠️  WARNING: MSE is very low. Model might be copying input exactly.")
        elif mse.item() > 100:
            print("⚠️  WARNING: MSE is very high. Model reconstruction is very different from input.")
        else:
            print("✓ Reconstruction appears reasonable!")
            print("  The problem is likely in the mel-to-audio conversion step.")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose VAE model outputs before audio conversion"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained VAE checkpoint (.pt)")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to input .wav file to test")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (default: auto)")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_path = Path(args.model)
    audio_path = Path(args.audio)

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return

    diagnose_file(
        model_path=model_path,
        audio_path=audio_path,
        device=device,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
