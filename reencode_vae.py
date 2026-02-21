"""
python reencode_vae.py \
  --checkpoint vae_latent128_epoch10.pt \
  --input "test_segments/*.wav" \
  --sample_rate 16000
"""

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
    beta = ckpt.get("beta", 0.1)

    # Build model and transforms first
    model = ConvVAE(in_channels=1, latent_dim=latent_dim, beta=beta).to(device)
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

    # ---- IMPORTANT: run one forward pass to initialize lazy layers ----
    # Use the first file as a dummy input to set enc_C/enc_H/enc_W and fc_* layers.
    with torch.no_grad():
        dummy_wav = load_mono_resampled(paths[0], sample_rate)
        dummy_wav = crop_pad_10s(dummy_wav, sample_rate).to(device)
        dummy_mel = mel(dummy_wav)
        dummy_log_mel = to_db(dummy_mel + 1e-6)
        d_mean = dummy_log_mel.mean()
        d_std = dummy_log_mel.std() + 1e-8
        dummy_log_mel_norm = (dummy_log_mel - d_mean) / d_std
        dummy_x = dummy_log_mel_norm.unsqueeze(0)  # (1, 1, n_mels, T_frames)
        _ = model(dummy_x)  # initializes fc_mu, fc_logvar, decoder_input
    # -------------------------------------------------------------------

    # Now we can safely load the checkpoint (all keys exist)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for path in paths:
        print(f"Processing {path}")
        wav = load_mono_resampled(path, sample_rate)  # (1, T)
        wav = crop_pad_10s(wav, sample_rate)
        wav = wav.to(device)

        with torch.no_grad():
            mel_spec = mel(wav)                 # (1, n_mels, T_frames)
            log_mel = to_db(mel_spec + 1e-6)    # (1, n_mels, T_frames)

            mean = log_mel.mean()
            std = log_mel.std() + 1e-8
            log_mel_norm = (log_mel - mean) / std
            x = log_mel_norm.unsqueeze(0)       # (1, 1, H, W)

            recon_norm, _, _ = model(x)
            recon_norm = recon_norm.squeeze(0)  # (1, n_mels, T_frames)

            recon_log_mel = recon_norm * std + mean
            recon_wav = logmel_to_waveform(recon_log_mel, inv_mel, griffin)

        recon_wav = recon_wav.cpu()
        target_len = int(10.0 * sample_rate)
        T = recon_wav.size(1)
        if T > target_len:
            recon_wav = recon_wav[:, :target_len]
        elif T < target_len:
            recon_wav = torch.nn.functional.pad(recon_wav, (0, target_len - T))

        out_path = path.with_name(f"{path.stem}-reencoded{path.suffix}")
        torchaudio.save(str(out_path), recon_wav, sample_rate)
        print(f"  Saved {out_path}")
