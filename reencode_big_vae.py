#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, InverseMelScale, GriffinLim


# ============================================================
# MODEL DEFINITION (same as training script)
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

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.l1_loss(recon_x, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl
        return loss, recon_loss, kl


# ============================================================
# AUDIO PRE/POST PROCESSING
# ============================================================

def load_mono_resampled(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))  # (C, T)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0)  # (T,)


def pad_or_trim(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    T = wav.numel()
    if T > target_len:
        return wav[:target_len]
    if T < target_len:
        pad = target_len - T
        return torch.cat([wav, torch.zeros(pad, dtype=wav.dtype)], dim=0)
    return wav


def prepare_transforms(sample_rate: int, n_mels: int, n_fft: int, hop_length: int):
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
    )
    to_db = AmplitudeToDB(stype="power")
    inv_mel = InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )
    griffin = GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        power=1.0,  # expects magnitude
    )
    return mel, to_db, inv_mel, griffin


def encode_decode_file(
    model: BigConvVAE,
    in_path: Path,
    out_path: Path,
    device: torch.device,
    sample_rate: int,
    duration_sec: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
):
    model.eval()
    target_len = int(sample_rate * duration_sec)

    mel, to_db, inv_mel, griffin = prepare_transforms(sample_rate, n_mels, n_fft, hop_length)

    # ----- load and preprocess -----
    wav = load_mono_resampled(in_path, sample_rate)       # (T,)
    wav = pad_or_trim(wav, target_len)                    # (T,)
    wav = wav.unsqueeze(0)                                # (1, T)

    with torch.no_grad():
        # Mel -> log-mel, normalize (same as dataset)
        mel_spec = mel(wav)                               # (1, n_mels, T_frames)
        log_mel = to_db(mel_spec + 1e-6)

        mean = log_mel.mean()
        std = log_mel.std() + 1e-8
        log_mel_norm = (log_mel - mean) / std             # (1, n_mels, T_frames)

        x = log_mel_norm.unsqueeze(0).to(device)          # (B=1, 1, n_mels, T_frames)

        # ----- VAE forward -----
        recon, mu, logvar = model(x)                      # (1, 1, n_mels, T_frames)
        recon = recon.squeeze(0).cpu()                    # (1, n_mels, T_frames)

        # ----- denormalize log-mel -----
        recon_log_mel = recon * std + mean                # (1, n_mels, T_frames)

        # log-mel (dB) -> mel power
        mel_power = torchaudio.functional.DB_to_amplitude(
            recon_log_mel, ref=1.0, power=2.0
        )                                                 # (1, n_mels, T_frames)

        # mel power -> linear power
        linear_power = inv_mel(mel_power)                 # (1, n_stft, T_frames)

        # power -> magnitude
        magnitude = torch.sqrt(torch.clamp(linear_power, min=1e-10))

        # Griffin-Lim to get waveform
        recon_wav = griffin(magnitude.squeeze(0))         # (T_recon,)

        # pad/trim again to target_len for consistency
        recon_wav = pad_or_trim(recon_wav, target_len)

    recon_wav = recon_wav.unsqueeze(0)                    # (1, T)
    torchaudio.save(str(out_path), recon_wav, sample_rate)
    print(f"Saved reencoded file: {out_path}")


# ============================================================
# MAIN / CLI
# ============================================================

def load_model(model_path: Path, device: torch.device, n_mels: int, duration_sec: float,
               sample_rate: int, n_fft: int, hop_length: int, dropout: float = 0.0) -> BigConvVAE:
    ckpt = torch.load(str(model_path), map_location=device)
    latent_dim = ckpt.get("latent_dim", 256)
    beta = ckpt.get("beta", 0.2)

    model = BigConvVAE(
        in_channels=1,
        latent_dim=latent_dim,
        beta=beta,
        dropout=dropout,
    ).to(device)

    # Build encoder/decoder linear layers with correct spatial dims
    # Use same mel shape that was used in training
    target_len = int(sample_rate * duration_sec)
    # torchaudio MelSpectrogram with center=True gives approx:
    # frames = 1 + T // hop_length
    frames = 1 + target_len // hop_length
    dummy = torch.zeros(1, 1, n_mels, frames, device=device)
    with torch.no_grad():
        _ = model.encode(dummy)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def get_wav_files(path: Path):
    if path.is_file():
        if path.suffix.lower() == ".wav":
            return [path]
        else:
            raise ValueError(f"Input file must be a .wav file, got: {path}")
    elif path.is_dir():
        return sorted([p for p in path.rglob("*.wav")])
    else:
        raise ValueError(f"Input path does not exist: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Encode/decode audio using a trained BigConvVAE and write FILENAME-reencoded.wav"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained VAE checkpoint (.pt)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input .wav file or directory containing .wav files")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (default: auto)")
    # Must match training hyperparameters
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout used when the model was trained (for compatibility)")

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_path = Path(args.model)
    input_path = Path(args.input)

    print(f"Using device: {device}")
    print(f"Loading model from {model_path} ...")
    model = load_model(
        model_path=model_path,
        device=device,
        n_mels=args.n_mels,
        duration_sec=args.duration_sec,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        dropout=args.dropout,
    )
    print("Model loaded.")

    wav_files = get_wav_files(input_path)
    if not wav_files:
        print("No .wav files found.")
        return

    print(f"Found {len(wav_files)} .wav file(s).")

    for wav_path in wav_files:
        out_path = wav_path.with_name(wav_path.stem + "-reencoded.wav")
        encode_decode_file(
            model=model,
            in_path=wav_path,
            out_path=out_path,
            device=device,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )


if __name__ == "__main__":
    main()
