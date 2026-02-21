#!/usr/bin/env python3
import os
import sys
import time
import argparse
import signal
import threading
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# ============================================================
# DENOISE + SEGMENT CORE (segment mode)
# ============================================================

AUDIO_EXTENSIONS = {
    '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma',
    '.opus', '.wav', '.aiff', '.ape', '.ac3', '.amr'
}


def calculate_chunk_volumes(audio: np.ndarray, sr: int, chunk_duration: float) -> Tuple[np.ndarray, int]:
    chunk_size = int(chunk_duration * sr)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive; check chunk_duration and sample rate")

    num_chunks = int(np.ceil(len(audio) / chunk_size))
    padded_length = num_chunks * chunk_size
    padded_audio = np.pad(audio, (0, padded_length - len(audio)), mode='constant')

    chunks = padded_audio.reshape(num_chunks, chunk_size)
    chunk_volumes = np.sqrt(np.mean(chunks ** 2, axis=1))
    return chunk_volumes, chunk_size


def calculate_threshold(chunk_volumes: np.ndarray, threshold_std: float, use_mean: bool) -> float:
    mean_vol = np.mean(chunk_volumes)
    std_vol = np.std(chunk_volumes)
    if use_mean:
        return mean_vol + threshold_std * std_vol
    else:
        return threshold_std * std_vol


def expand_mask(chunk_mask: np.ndarray, padding_chunks: int) -> np.ndarray:
    if padding_chunks == 0:
        return chunk_mask
    expanded_mask = np.copy(chunk_mask)
    above_indices = np.where(chunk_mask)[0]
    for idx in above_indices:
        start = max(0, idx - padding_chunks)
        end = min(len(chunk_mask), idx + padding_chunks + 1)
        expanded_mask[start:end] = True
    return expanded_mask


def find_segments(
    chunk_mask: np.ndarray,
    chunk_size: int,
    chunk_duration: float,
    min_segment_duration: float,
) -> List[Tuple[int, int]]:
    padded = np.pad(chunk_mask, (1, 1), mode='constant', constant_values=0)
    starts = np.where(np.diff(padded.astype(int)) == 1)[0]
    ends = np.where(np.diff(padded.astype(int)) == -1)[0]

    segments = []
    for start_chunk, end_chunk in zip(starts, ends):
        start_sample = start_chunk * chunk_size
        end_sample = end_chunk * chunk_size
        duration = (end_sample - start_sample) / chunk_size * chunk_duration
        if duration >= min_segment_duration:
            segments.append((start_sample, end_sample))
    return segments


@dataclass
class SegmentInfo:
    file_path: Path
    start_sample: int
    end_sample: int
    sr: int


# ============================================================
# STREAMING SEGMENTED DATASET (preprocessing in background)
# ============================================================

class StreamingSegmentedAudioDataset(IterableDataset):
    """
    Streaming dataset that:
      - Starts a background thread to:
          * find audio files
          * denoise+segment them
          * put SegmentInfo objects into a queue
      - __iter__:
          * pulls segments from queue
          * loads just that waveform
          * crops/pads to fixed length
          * converts to log-mel

    Training and preprocessing run in parallel.
    """

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        target_duration_sec: float = 10.0,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        f_min: float = 0.0,
        f_max: float = None,
        threshold_std: float = 0.25,
        use_mean: bool = True,
        chunk_duration: float = 0.05,
        padding_chunks: int = 5,
        min_segment_duration: float = 0.3,
        queue_size: int = 1000,
        min_queue_size: int = 100,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * target_duration_sec)

        # log-mel transforms
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

        # denoise/segment params
        self.threshold_std = threshold_std
        self.use_mean = use_mean
        self.chunk_duration = chunk_duration
        self.padding_chunks = padding_chunks
        self.min_segment_duration = min_segment_duration

        # streaming setup
        self.queue_size = queue_size
        self.min_queue_size = min_queue_size
        self.segment_queue: queue.Queue[SegmentInfo] = queue.Queue(maxsize=queue_size)
        self.processing_complete = threading.Event()
        self.stop_processing = threading.Event()

        # stats
        self.total_segments_found = 0
        self.total_files_processed = 0
        self.total_files = 0

        self._start_processing()

    def _find_audio_files(self) -> List[Path]:
        audio_files: List[Path] = []
        for root, dirs, files in os.walk(self.root_dir):
            for fname in files:
                p = Path(root) / fname
                if p.suffix.lower() in AUDIO_EXTENSIONS:
                    audio_files.append(p)
        return sorted(audio_files)

    def _load_mono_resampled(self, path: Path) -> Tuple[np.ndarray, int]:
        wav, sr = torchaudio.load(str(path))  # (C, T)
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            sr = self.sample_rate
        audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
        return audio, sr

    def _process_files_worker(self):
        print(f"Starting background processing of audio files in {self.root_dir}")
        files = self._find_audio_files()
        if not files:
            print(f"No audio files found under {self.root_dir}")
            self.processing_complete.set()
            return

        self.total_files = len(files)
        print(f"Found {self.total_files} audio files to process")

        for i, path in enumerate(files, 1):
            if self.stop_processing.is_set():
                break
            try:
                audio, sr = self._load_mono_resampled(path)
                if len(audio) == 0:
                    continue

                chunk_volumes, chunk_size = calculate_chunk_volumes(
                    audio, sr, self.chunk_duration
                )
                if len(chunk_volumes) == 0:
                    continue

                threshold = calculate_threshold(
                    chunk_volumes, self.threshold_std, self.use_mean
                )
                chunk_mask = chunk_volumes >= threshold
                chunk_mask = expand_mask(chunk_mask, self.padding_chunks)

                segs = find_segments(
                    chunk_mask,
                    chunk_size,
                    self.chunk_duration,
                    self.min_segment_duration,
                )

                file_segments = 0
                for start, end in segs:
                    if self.stop_processing.is_set():
                        break
                    end = min(end, len(audio))
                    if end <= start:
                        continue
                    seg_info = SegmentInfo(
                        file_path=path,
                        start_sample=int(start),
                        end_sample=int(end),
                        sr=sr,
                    )
                    # block if queue is full
                    while not self.stop_processing.is_set():
                        try:
                            self.segment_queue.put(seg_info, timeout=1.0)
                            file_segments += 1
                            self.total_segments_found += 1
                            break
                        except queue.Full:
                            continue

                self.total_files_processed += 1
                if i % 10 == 0 or i == len(files):
                    print(
                        f"  Processed {i}/{len(files)} files, "
                        f"segments found: {self.total_segments_found}, "
                        f"queue size: {self.segment_queue.qsize()}"
                    )
            except Exception as e:
                print(f"Error processing {path}: {e}")

        print(f"Background processing complete. Total segments found: {self.total_segments_found}")
        self.processing_complete.set()

    def _start_processing(self):
        self.processing_thread = threading.Thread(
            target=self._process_files_worker, daemon=True
        )
        self.processing_thread.start()

        # Wait for some initial segments
        print("Waiting for initial segments to be processed...")
        while (
            self.segment_queue.qsize() < self.min_queue_size
            and not self.processing_complete.is_set()
        ):
            time.sleep(0.1)

        if self.segment_queue.qsize() > 0:
            print(f"Ready to start training with {self.segment_queue.qsize()} segments queued")
        elif self.processing_complete.is_set():
            if self.total_segments_found == 0:
                raise RuntimeError(f"No segments found under {self.root_dir} with current settings")
            print(f"Processing complete with {self.total_segments_found} total segments")

    def _extract_segment_waveform(self, seg: SegmentInfo) -> torch.Tensor:
        audio, sr = self._load_mono_resampled(seg.file_path)
        assert sr == self.sample_rate
        start = min(seg.start_sample, len(audio))
        end = min(seg.end_sample, len(audio))
        segment = audio[start:end]
        if len(segment) == 0:
            segment = np.zeros(1, dtype=np.float32)
        if len(segment) > self.target_len:
            segment = segment[: self.target_len]
        elif len(segment) < self.target_len:
            pad_len = self.target_len - len(segment)
            segment = np.pad(segment, (0, pad_len), mode="constant")
        return torch.from_numpy(segment).unsqueeze(0)  # (1, T)

    def __iter__(self):
        segments_yielded = 0
        while True:
            try:
                seg = self.segment_queue.get(timeout=1.0)
                wav = self._extract_segment_waveform(seg)  # (1, T)
                with torch.no_grad():
                    mel = self.mel(wav)  # (1, n_mels, T_frames)
                    log_mel = self.to_db(mel + 1e-6)
                mean = log_mel.mean()
                std = log_mel.std() + 1e-8
                log_mel = (log_mel - mean) / std
                segments_yielded += 1
                yield log_mel  # (1, n_mels, T_frames)
            except queue.Empty:
                if self.processing_complete.is_set() and self.segment_queue.empty():
                    print(f"Finished iterating over all segments. Total yielded: {segments_yielded}")
                    break
                continue

    def stop(self):
        self.stop_processing.set()
        if hasattr(self, "processing_thread"):
            self.processing_thread.join(timeout=5.0)

    def __del__(self):
        self.stop()


# ============================================================
# BIGGER VAE (DEEPER, MORE CHANNELS, RESBLOCKS)
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
# TRAINING WITH TIMING + INTERRUPT MENU
# ============================================================

def train(
    data_dir: str,
    latent_dim: int = 256,
    batch_size: int = 64,
    num_epochs: int = 1,
    lr: float = 2e-4,
    beta: float = 0.2,
    dropout: float = 0.1,
    device: Optional[str] = None,
    sample_rate: int = 16000,
    duration_sec: float = 10.0,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    threshold_std: float = 0.25,
    use_mean: bool = True,
    chunk_duration: float = 0.05,
    padding_chunks: int = 5,
    min_segment_duration: float = 0.3,
    log_every: int = 100,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = StreamingSegmentedAudioDataset(
        root_dir=data_dir,
        sample_rate=sample_rate,
        target_duration_sec=duration_sec,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        threshold_std=threshold_std,
        use_mean=use_mean,
        chunk_duration=chunk_duration,
        padding_chunks=padding_chunks,
        min_segment_duration=min_segment_duration,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,   # IterableDataset cannot be shuffled by DataLoader
        num_workers=0,   # important: dataset already uses threads
        pin_memory=True,
        drop_last=True,
    )

    # Warm-up: get an example batch
    print("Getting example batch to determine input shape...")
    example = next(iter(dataloader))
    print(f"Example batch shape (log-mel): {example.shape}")

    model = BigConvVAE(
        in_channels=1,
        latent_dim=latent_dim,
        beta=beta,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    interrupted = False

    def get_unique_checkpoint_path(base_name: str) -> str:
        if not os.path.exists(base_name):
            return base_name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem, *ext = base_name.rsplit(".", 1)
        if ext:
            base = f"{stem}_{ts}.{ext[0]}"
        else:
            base = f"{base_name}_{ts}"
        path = base
        counter = 1
        while os.path.exists(path):
            if ext:
                path = f"{stem}_{ts}_{counter}.{ext[0]}"
            else:
                path = f"{base_name}_{ts}_{counter}"
            counter += 1
        return path

    def save_checkpoint(epoch: int, reason: str = "interrupted") -> str:
        base = f"big_vae_latent{latent_dim}_epoch{epoch}_{reason}.pt"
        ckpt_path = get_unique_checkpoint_path(base)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "latent_dim": latent_dim,
                "beta": beta,
                "interrupted": True,
            },
            ckpt_path,
        )
        return ckpt_path

    def handle_interrupt_menu(epoch: int):
        print("\n" + "=" * 50)
        print("TRAINING INTERRUPTED")
        print("=" * 50)
        print("Choose an option:")
        print("1. Cancel run and save checkpoint")
        print("2. Continue run but archive current checkpoint")
        print("3. Continue run without doing anything")
        print("4. Cancel run without saving")
        print("=" * 50)
        while True:
            try:
                choice = input("Enter your choice (1–4): ").strip()
                if choice == "1":
                    ck = save_checkpoint(epoch, "cancelled")
                    print(f"Saved checkpoint: {ck}")
                    print("Exiting...")
                    dataset.stop()
                    sys.exit(0)
                elif choice == "2":
                    ck = save_checkpoint(epoch, "archived")
                    print(f"Archived checkpoint: {ck}")
                    print("Continuing training...")
                    return True
                elif choice == "3":
                    print("Continuing training without saving...")
                    return True
                elif choice == "4":
                    print("Exiting without saving...")
                    dataset.stop()
                    sys.exit(0)
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except (EOFError, KeyboardInterrupt):
                print("\nForced exit...")
                dataset.stop()
                sys.exit(0)

    def sigint_handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, sigint_handler)

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_start = time.perf_counter()

        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        running_batches = 0

        load_time_accum = 0.0
        compute_time_accum = 0.0
        timing_batches = 0

        prev_time = time.perf_counter()

        for batch_idx, x in enumerate(dataloader):
            if interrupted:
                interrupted = False
                _ = handle_interrupt_menu(epoch)

            batch_start = time.perf_counter()
            data_loading_time = batch_start - prev_time

            x = x.to(device, non_blocking=True)

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            compute_start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                recon, mu, logvar = model(x)
                loss, recon_loss, kl = model.loss_function(recon, x, mu, logvar)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            compute_end = time.perf_counter()
            compute_time = compute_end - compute_start

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl.item()
            running_batches += 1

            load_time_accum += data_loading_time
            compute_time_accum += compute_time
            timing_batches += 1

            global_step += 1
            prev_time = compute_end

            if (batch_idx + 1) % log_every == 0:
                avg_loss = running_loss / running_batches
                avg_recon = running_recon / running_batches
                avg_kl = running_kl / running_batches

                avg_load = load_time_accum / max(1, timing_batches)
                avg_compute = compute_time_accum / max(1, timing_batches)
                ratio = avg_compute / (avg_load + 1e-8)

                print(
                    f"Epoch [{epoch}/{num_epochs}] "
                    f"Step {batch_idx+1} "
                    f"Loss: {avg_loss:.4f} Recon: {avg_recon:.4f} KL: {avg_kl:.4f} | "
                    f"Avg load: {avg_load*1000:.1f} ms, "
                    f"Avg compute: {avg_compute*1000:.1f} ms "
                    f"(compute/load ratio: {ratio:.2f})"
                )

                running_loss = running_recon = running_kl = 0.0
                running_batches = 0
                load_time_accum = compute_time_accum = 0.0
                timing_batches = 0

        epoch_end = time.perf_counter()
        print(f"Epoch {epoch} finished in {(epoch_end-epoch_start)/60:.2f} min")

        base = f"big_vae_latent{latent_dim}_epoch{epoch}.pt"
        ckpt_path = base if not os.path.exists(base) else get_unique_checkpoint_path(base)
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

    dataset.stop()


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train a bigger ConvVAE on streaming denoised+segmented audio.")
    p.add_argument("--data_dir", type=str, required=True)

    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--device", type=str, default=None)

    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration_sec", type=float, default=10.0)
    p.add_argument("--n_mels", type=int, default=80)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)

    p.add_argument("--threshold_std", type=float, default=0.25)
    p.add_argument("--no_mean", action="store_true")
    p.add_argument("--chunk_duration", type=float, default=0.05)
    p.add_argument("--padding_chunks", type=int, default=5)
    p.add_argument("--min_segment_duration", type=float, default=0.3)

    p.add_argument("--log_every", type=int, default=100)

    return p.parse_args()


def main():
    args = parse_args()
    use_mean = not args.no_mean

    train(
        data_dir=args.data_dir,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        beta=args.beta,
        dropout=args.dropout,
        device=args.device,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        threshold_std=args.threshold_std,
        use_mean=use_mean,
        chunk_duration=args.chunk_duration,
        padding_chunks=args.padding_chunks,
        min_segment_duration=args.min_segment_duration,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
