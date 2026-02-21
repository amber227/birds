#!/usr/bin/env python3
import os
import glob
import math
import argparse
import signal
import sys
import threading
import queue
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

import warnings

# Option A: silence all torchaudio UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# ============================================================
# Denoise + segmentation core (ported from SimpleAudioDenoiser)
# ============================================================

AUDIO_EXTENSIONS = {
    '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma',
    '.opus', '.wav', '.aiff', '.ape', '.ac3', '.amr'
}


def calculate_chunk_volumes(audio: np.ndarray, sr: int, chunk_duration: float) -> Tuple[np.ndarray, int]:
    """
    Calculate RMS volume for each chunk of audio.
    Returns (chunk_volumes, chunk_size_samples).
    """
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
        threshold = mean_vol + (threshold_std * std_vol)
    else:
        threshold = threshold_std * std_vol
    return threshold


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
    min_segment_duration: float
) -> List[Tuple[int, int]]:
    """
    Returns list of (start_sample, end_sample) for segments of consecutive
    chunks above threshold, filtered by min_segment_duration (in seconds).
    """
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
# Dataset: denoise+segment on-the-fly, then log-mel
# ============================================================

class StreamingSegmentedAudioDataset(IterableDataset):
    """
    Streaming dataset that processes audio files in parallel with training.
    Uses a background thread to continuously find and queue segments.
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
        # denoise/segment params
        threshold_std: float = 0.25,
        use_mean: bool = True,
        chunk_duration: float = 0.05,
        padding_chunks: int = 5,
        min_segment_duration: float = 0.3,
        # streaming params
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
        self.segment_queue = queue.Queue(maxsize=queue_size)
        self.processing_complete = threading.Event()
        self.stop_processing = threading.Event()
        
        # stats
        self.total_segments_found = 0
        self.total_files_processed = 0
        self.total_files = 0
        
        # start background processing
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
        """Background worker that processes files and adds segments to queue"""
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
                for (start, end) in segs:
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
                    
                    # Block if queue is full
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
                    print(f"  Processed {i}/{len(files)} files, "
                          f"segments found: {self.total_segments_found}, "
                          f"queue size: {self.segment_queue.qsize()}")
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")

        print(f"Background processing complete. Total segments found: {self.total_segments_found}")
        self.processing_complete.set()

    def _start_processing(self):
        """Start the background processing thread"""
        self.processing_thread = threading.Thread(target=self._process_files_worker, daemon=True)
        self.processing_thread.start()
        
        # Wait for initial segments to be available
        print("Waiting for initial segments to be processed...")
        while (self.segment_queue.qsize() < self.min_queue_size and 
               not self.processing_complete.is_set()):
            time.sleep(0.1)
        
        if self.segment_queue.qsize() > 0:
            print(f"Ready to start training with {self.segment_queue.qsize()} segments queued")
        elif self.processing_complete.is_set():
            if self.total_segments_found == 0:
                raise RuntimeError(f"No segments found under {self.root_dir} with current settings")
            print(f"Processing complete with {self.total_segments_found} total segments")

    def _extract_segment_waveform(self, seg: SegmentInfo) -> torch.Tensor:
        # reload audio for this file (we don't keep full audio in RAM)
        audio, sr = self._load_mono_resampled(seg.file_path)
        # safety: ensure sr == self.sample_rate
        assert sr == self.sample_rate, "Resampling invariant violated"

        start = min(seg.start_sample, len(audio))
        end = min(seg.end_sample, len(audio))
        segment = audio[start:end]
        if len(segment) == 0:
            # fallback: tiny silence (will be dropped by training eventually)
            segment = np.zeros(1, dtype=np.float32)

        # crop/pad to fixed target_len
        if len(segment) > self.target_len:
            segment = segment[: self.target_len]
        elif len(segment) < self.target_len:
            pad_len = self.target_len - len(segment)
            segment = np.pad(segment, (0, pad_len), mode="constant")

        wav = torch.from_numpy(segment).unsqueeze(0)  # (1, T)
        return wav

    def __iter__(self):
        """Iterate over segments as they become available"""
        segments_yielded = 0
        
        while True:
            try:
                # Try to get a segment from the queue
                seg = self.segment_queue.get(timeout=1.0)
                
                # Process the segment
                wav = self._extract_segment_waveform(seg)  # (1, T)
                
                with torch.no_grad():
                    mel = self.mel(wav)  # (1, n_mels, T_frames)
                    log_mel = self.to_db(mel + 1e-6)
                # per-sample normalization
                mean = log_mel.mean()
                std = log_mel.std() + 1e-8
                log_mel = (log_mel - mean) / std
                
                segments_yielded += 1
                yield log_mel  # (1, n_mels, T_frames)
                
            except queue.Empty:
                # Check if processing is complete and queue is empty
                if self.processing_complete.is_set() and self.segment_queue.empty():
                    print(f"Finished iterating over all segments. Total yielded: {segments_yielded}")
                    break
                # Otherwise continue waiting for more segments
                continue
                
    def stop(self):
        """Stop the background processing"""
        self.stop_processing.set()
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=5.0)

    def __del__(self):
        """Cleanup when dataset is destroyed"""
        self.stop()


# ============================================================
# VAE Model (same as before, conv encoder/decoder)
# ============================================================

class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, beta=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
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

        # Will be lazily initialized once we know encoder output shape
        self.enc_out_dim = None          # flattened size after conv encoder
        self.enc_C = None                # channels after conv encoder
        self.enc_H = None                # height after conv encoder
        self.enc_W = None                # width after conv encoder

        self.fc_mu = None
        self.fc_logvar = None
        self.decoder_input = None

        # Decoder (ConvTranspose stack mirroring encoder)
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
        """
        x: (B, C, H, W) -> mu, logvar in R^latent_dim
        """
        B = x.size(0)
        h = x
        for block in self.encoder_convs:
            h = block(h)          # (B, C_enc, H_enc, W_enc)

        # Lazily record encoder output shape the first time
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
        """
        z: (B, latent_dim)
        target_shape: shape of x = (B, 1, H, W) so we can crop final output.
        """
        B = z.size(0)
        H, W = target_shape[2], target_shape[3]

        # Map latent back to encoder feature map size
        h = self.decoder_input(z)                    # (B, enc_out_dim)
        h = h.view(B, self.enc_C, self.enc_H, self.enc_W)

        # ConvTranspose2d stack
        for block in self.decoder_convs:
            h = block(h)
        h = self.final_conv(h)

        # Crop spatial dims to exactly match input H, W
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
# Training
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
    # denoise/segment params
    threshold_std: float = 0.25,
    use_mean: bool = True,
    chunk_duration: float = 0.05,
    padding_chunks: int = 5,
    min_segment_duration: float = 0.3,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Global variables for signal handler
    model = None
    optimizer = None
    current_epoch = 0
    
    def get_unique_checkpoint_path(base_name: str) -> str:
        """Generate a unique checkpoint path that doesn't overwrite existing files"""
        if not os.path.exists(base_name):
            return base_name
        
        # Add timestamp to make it unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = base_name.rsplit('.', 1)
        if len(name_parts) == 2:
            unique_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            unique_name = f"{base_name}_{timestamp}"
        
        # If still exists (very unlikely), add a counter
        counter = 1
        while os.path.exists(unique_name):
            if len(name_parts) == 2:
                unique_name = f"{name_parts[0]}_{timestamp}_{counter}.{name_parts[1]}"
            else:
                unique_name = f"{base_name}_{timestamp}_{counter}"
            counter += 1
        
        return unique_name

    def save_checkpoint(epoch, model, optimizer, latent_dim, beta, reason="interrupted"):
        """Save current model state and return the checkpoint path"""
        base_path = f"vae_latent{latent_dim}_epoch{epoch}_{reason}.pt"
        ckpt_path = get_unique_checkpoint_path(base_path)
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
    
    # Flag to handle interruption
    interrupted = False
    
    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) by setting interrupt flag"""
        nonlocal interrupted
        if model is not None and optimizer is not None:
            interrupted = True
        else:
            print("\nTraining interrupted before model initialization!")
            sys.exit(0)
    
    def handle_interrupt_menu(epoch, model, optimizer, latent_dim, beta):
        """Present interactive menu when training is interrupted"""
        print("\n" + "="*50)
        print("TRAINING INTERRUPTED")
        print("="*50)
        print("Choose an option:")
        print("1. Cancel run and save checkpoint")
        print("2. Continue run but archive current checkpoint")
        print("3. Continue run without doing anything")
        print("4. Cancel run without saving")
        print("="*50)
        
        while True:
            try:
                choice = input("Enter your choice (1-4): ").strip()
                
                if choice == "1":
                    ckpt_path = save_checkpoint(epoch, model, optimizer, latent_dim, beta, "cancelled")
                    print(f"Saved checkpoint: {ckpt_path}")
                    print("Exiting...")
                    sys.exit(0)
                    
                elif choice == "2":
                    ckpt_path = save_checkpoint(epoch, model, optimizer, latent_dim, beta, "archived")
                    print(f"Archived checkpoint: {ckpt_path}")
                    print("Continuing training...")
                    return True  # Continue training
                    
                elif choice == "3":
                    print("Continuing training without saving...")
                    return True  # Continue training
                    
                elif choice == "4":
                    print("Exiting without saving...")
                    sys.exit(0)
                    
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\nForced exit...")
                sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    dataset = StreamingSegmentedAudioDataset(
        root_dir=data_dir,
        sample_rate=sample_rate,
        target_duration_sec=duration_sec,
        n_mels=n_mels,
        threshold_std=threshold_std,
        use_mean=use_mean,
        chunk_duration=chunk_duration,
        padding_chunks=padding_chunks,
        min_segment_duration=min_segment_duration,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Can't shuffle streaming dataset
        num_workers=0,   # Must be 0 for IterableDataset with threading
        pin_memory=True,
        drop_last=True,
    )

    # Get example batch to determine shape
    print("Getting example batch to determine input shape...")
    dataloader_iter = iter(dataloader)
    x0 = next(dataloader_iter)
    print(f"Example batch shape (log-mel): {x0.shape}")
    B, C, H, W = x0.shape
    
    # Create a fresh dataloader for training
    dataset = StreamingSegmentedAudioDataset(
        root_dir=data_dir,
        sample_rate=sample_rate,
        target_duration_sec=duration_sec,
        n_mels=n_mels,
        threshold_std=threshold_std,
        use_mean=use_mean,
        chunk_duration=chunk_duration,
        padding_chunks=padding_chunks,
        min_segment_duration=min_segment_duration,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    model = ConvVAE(in_channels=C, latent_dim=latent_dim, beta=beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    try:
        for epoch in range(1, num_epochs + 1):
            current_epoch = epoch
            model.train()
            running_loss = 0.0
            running_recon = 0.0
            running_kl = 0.0
            count = 0

            for batch_idx, x in enumerate(dataloader):
                # Check for interrupt at the start of each batch
                if interrupted:
                    interrupted = False  # Reset flag
                    continue_training = handle_interrupt_menu(epoch, model, optimizer, latent_dim, beta)
                    if not continue_training:
                        return  # This shouldn't happen as menu handles exits
                
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, recon_loss, kl = model.loss_function(recon, x, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_recon += recon_loss.item()
                running_kl += kl.item()
                count += 1
                global_step += 1

                if (batch_idx + 1) % 100 == 0:
                    avg_loss = running_loss / count
                    avg_recon = running_recon / count
                    avg_kl = running_kl / count
                    print(
                        f"Epoch [{epoch}/{num_epochs}] "
                        f"Step [{batch_idx+1}/{len(dataloader)}] "
                        f"Loss: {avg_loss:.4f} Recon: {avg_recon:.4f} KL: {avg_kl:.4f}"
                    )
                    running_loss = running_recon = running_kl = 0.0
                    count = 0

            if count > 0:
                avg_loss = running_loss / count
                print(f"End of Epoch {epoch}: avg loss over last window = {avg_loss:.4f}")

            base_path = f"vae_latent{latent_dim}_epoch{epoch}.pt"
            ckpt_path = get_unique_checkpoint_path(base_path)
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
            
    except KeyboardInterrupt:
        # This shouldn't happen now since we handle interrupts with the flag
        print("\nUnexpected KeyboardInterrupt - exiting...")
        sys.exit(0)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ConvVAE on denoised+segmented audio (log-mel)."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing raw audio files.")
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
                        help="Fixed duration for segments (crop/pad).")
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda', 'cpu', or None to auto-select.")

    # denoise/segment params (mirroring your script defaults for --segment)
    parser.add_argument("--threshold_std", type=float, default=0.25,
                        help="Std multiplier for threshold (std-multiplier).")
    parser.add_argument("--no_mean", action="store_true",
                        help="Use std * std-multiplier from zero (no mean).")
    parser.add_argument("--chunk_duration", type=float, default=0.05,
                        help="Chunk duration in seconds.")
    parser.add_argument("--padding_chunks", type=int, default=5,
                        help="Chunks to keep around loud chunks.")
    parser.add_argument("--min_segment_duration", type=float, default=0.3,
                        help="Minimum segment duration (seconds).")

    return parser.parse_args()


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
        num_workers=args.num_workers,
        device=args.device,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        n_mels=args.n_mels,
        threshold_std=args.threshold_std,
        use_mean=use_mean,
        chunk_duration=args.chunk_duration,
        padding_chunks=args.padding_chunks,
        min_segment_duration=args.min_segment_duration,
    )


if __name__ == "__main__":
    main()
