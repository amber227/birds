#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, InverseMelScale, GriffinLim

# --------------------------
# VAE definition (same as training, fixed reshape)
# --------------------------

class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, beta=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        enc_channels = [32, 64, 128, 256]
        self.encoder_convs = nn.ModuleList()
        in_ch = in_channels
        for out_ch in enc_channels:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.encoder_convs.append(block)
            in_ch = out_ch

        self.enc_out_dim = None
        self.enc_C = None
        self.enc_H = None
        self.enc_W = None

        self.fc_mu = None
        self.fc_logvar = None
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

        self.final_conv = nn.ConvTranspose2d(
            dec_channels[-1], 1, kernel_size=4, stride=2, padding=1
        )

    def encode(self, x):
        B = x.size(0)
        h = x
        for block in self.encoder_convs:
            h = block(h)

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

        for block in self.decoder_convs:
            h = block(h)
        h = self.final_conv(h)

        h = h[:, :, :H, :W]
        return h

    def forward(self, x):
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.shape)
        return recon, mu, logvar


# --------------------------
# Audio + mel utilities
# --------------------------

def load_mono_resampled(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))  # (C, T)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav  # (1, T)


def crop_pad_10s(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    target_len = int(10.0 * sample_rate)
    T = wav.size(1)
    if T > target_len:
        print(f"  Warning: input longer than 10s ({T/sample_rate:.2f}s), truncating to 10s")
        wav = wav[:, :target_len]
    elif T < target_len:
        pad_len = target_len - T
        wav = torch.nn.functional.pad(wav, (0, pad_len))
    return wav


def build_transforms(sample_rate: int, n_fft: int, hop_length: int, n_mels: int, f_min: float, f_max: float):
    mel = MelSpectrogram(
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
    to_db = AmplitudeToDB(stype="power")
    inv_mel = InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
    )
    griffin = GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        power=1.0,
        n_iter=32,
    )
    return mel, to_db, inv_mel, griffin


def logmel_to_waveform(
    log_mel: torch.Tensor,
    inv_mel: InverseMelScale,
    griffin: GriffinLim,
) -> torch.Tensor:
    """
    log_mel: (1, n_mels, T)
    Returns waveform: (1, T_wav)
    """
    # Invert log10 power: log10(power) -> power
    mel_power = 10.0 ** (log_mel / 10.0)
    # mel -> linear magnitude
    spec = inv_mel(mel_power)
    # Griffin-Lim expects magnitude; spec is power, take sqrt
    mag = torch.sqrt(torch.clamp(spec, min=1e-10))
    wav = griffin(mag)  # (1, T)
    return wav


# --------------------------
# Main re-encode function
# --------------------------

def reencode_files(
    ckpt_path: Path,
    input_pattern: str,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float | None = None,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device)
    latent_dim = ckpt.get("latent_dim", 128)

    model = ConvVAE(in_channels=1, latent_dim=latent_dim, beta=ckpt.get("beta", 0.1))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    mel, to_db, inv_mel, griffin = build_transforms(
        sample_rate, n_fft, hop_length, n_mels, f_min, f_max
    )
    mel.to(device)
    to_db.to(device)
    inv_mel.to(device)
    griffin.to(device)

    paths = [Path(p) for p in glob.glob(input_pattern)]
    if not paths:
        print(f"No files match pattern: {input_pattern}")
        return

    print(f"Found {len(paths)} files")

    for path in paths:
        print(f"Processing {path}")
        wav = load_mono_resampled(path, sample_rate)  # (1, T)
        wav = crop_pad_10s(wav, sample_rate)
        wav = wav.to(device)

        with torch.no_grad():
            mel_spec = mel(wav)                 # (1, n_mels, T_frames)
            log_mel = to_db(mel_spec + 1e-6)    # (1, n_mels, T_frames)

            # per-sample normalization (match training)
            mean = log_mel.mean()
            std = log_mel.std() + 1e-8
            log_mel_norm = (log_mel - mean) / std
            x = log_mel_norm.unsqueeze(0)       # (B=1, C=1, H=n_mels, W=T_frames)

            recon_norm, _, _ = model(x)
            recon_norm = recon_norm.squeeze(0)  # (1, n_mels, T_frames)

            # de-normalize using same mean/std
            recon_log_mel = recon_norm * std + mean

            # back to waveform
            recon_wav = logmel_to_waveform(recon_log_mel, inv_mel, griffin)  # (1, T')

        recon_wav = recon_wav.cpu()
        # match original 10s length if needed
        target_len = int(10.0 * sample_rate)
        T = recon_wav.size(1)
        if T > target_len:
            recon_wav = recon_wav[:, :target_len]
        elif T < target_len:
            recon_wav = torch.nn.functional.pad(recon_wav, (0, target_len - T))

        out_path = path.with_name(f"{path.stem}-reencoded{path.suffix}")
        torchaudio.save(str(out_path), recon_wav, sample_rate)
        print(f"  Saved {out_path}")


# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Re-encode audio with trained ConvVAE checkpoint.")
    ap.add_argument("--checkpoint", type=str, required=True, help=".pt checkpoint path")
    ap.add_argument("--input", type=str, required=True,
                    help="File path or glob (e.g. 'data/*.wav')")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=80)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    reencode_files(
        ckpt_path=Path(args.checkpoint),
        input_pattern=args.input,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        device=args.device,
    )


if __name__ == "__main__":
    main())
