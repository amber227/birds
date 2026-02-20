#!/usr/bin/env python3
"""
ML-based Audio Denoising Script using DeepFilterNet

Applies deep learning-based denoising to audio files using DeepFilterNet.
Supports multiprocessing and tracks progress for resumability.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Set, Optional, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torchaudio
    import numpy as np
    from df.enhance import enhance, init_df, save_audio
    from df.io import resample
except ImportError:
    print("Error: DeepFilterNet not installed. Please install it with:", file=sys.stderr)
    print("  pip install deepfilternet", file=sys.stderr)
    print("\nOr with uv:", file=sys.stderr)
    print("  uv pip install deepfilternet", file=sys.stderr)
    sys.exit(1)


# Common audio file extensions to process
AUDIO_EXTENSIONS = {
    '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma',
    '.opus', '.wav', '.aiff', '.ape', '.ac3', '.amr'
}


class MLAudioDenoiser:
    def __init__(
        self,
        directory: Path,
        model_name: str = 'DeepFilterNet',
        num_workers: int = 1,  # DeepFilterNet is GPU-intensive, default to 1
        progress_file: str = '.denoise_ml_progress.json',
        output_format: str = 'wav',
        sample_rate: int = None,
        skip_existing: bool = True,
        use_gpu: bool = True,
        post_filter: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
        use_existing_progress: bool = True
    ):
        self.directory = directory
        self.model_name = model_name
        self.num_workers = num_workers
        self.progress_file = directory / progress_file
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.skip_existing = skip_existing
        self.post_filter = post_filter
        self.dry_run = dry_run
        self.verbose = verbose
        self.use_existing_progress = use_existing_progress

        # Set up logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

        # Set device
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            if use_gpu:
                self.logger.warning("GPU requested but not available, using CPU")
            else:
                self.logger.info("Using CPU")

        # Load model (done once, reused across all files)
        if not self.dry_run:
            self.logger.info(f"Loading {model_name} model...")
            self.logger.info(f"Post-filter: {'enabled' if post_filter else 'disabled (faster)'}")
            try:
                self.model, self.df_state, _ = init_df(
                    model_base_dir=model_name,
                    post_filter=post_filter
                )
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise

        # Load or initialize progress tracking
        self.processed_files = self._load_progress()

    def _load_progress(self) -> Set[str]:
        """Load the set of already processed files."""
        if not self.use_existing_progress:
            # Delete existing progress file if starting fresh
            if self.progress_file.exists():
                self.logger.info("Starting fresh - ignoring existing progress")
                self.progress_file.unlink()
            return set()

        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    processed = set(data.get('processed', []))
                    self.logger.info(f"Loaded progress: {len(processed)} files already processed")
                    return processed
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}. Starting fresh.")
                return set()
        return set()

    def _save_progress(self, file_path: str):
        """Save progress after processing a file."""
        self.processed_files.add(file_path)
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'processed': list(self.processed_files),
                    'total_processed': len(self.processed_files)
                }, f)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a unique hash for a file based on its path relative to the base directory."""
        relative_path = file_path.relative_to(self.directory)
        return str(relative_path)

    def find_audio_files(self) -> list[Path]:
        """Recursively find all audio files in the directory."""
        self.logger.info(f"Scanning directory: {self.directory}")
        audio_files = []

        for root, dirs, files in os.walk(self.directory):
            for file in files:
                file_path = Path(root) / file

                # Skip files that already have "-denoised" in the name
                if '-denoised' in file_path.stem:
                    continue

                if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    file_hash = self._get_file_hash(file_path)
                    if file_hash not in self.processed_files:
                        audio_files.append(file_path)

        self.logger.info(f"Found {len(audio_files)} audio files to process")
        return audio_files

    def _process_file(self, file_path: Path) -> Tuple[bool, str, Optional[str]]:
        """
        Process a single audio file with DeepFilterNet.

        Returns:
            Tuple of (success, file_path, error_message)
        """
        file_hash = self._get_file_hash(file_path)

        try:
            # Check if file still exists
            if not file_path.exists():
                return True, file_hash, "File no longer exists"

            # Create output path with -denoised suffix
            output_stem = file_path.stem + '-denoised'
            output_path = file_path.parent / f"{output_stem}.{self.output_format}"

            # Skip if output already exists
            if self.skip_existing and output_path.exists():
                self.logger.debug(f"Skipping {file_path.name}: output already exists")
                return True, file_hash, None

            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would denoise: {file_path.name} -> {output_path.name}")
                return True, file_hash, None

            # Process using DeepFilterNet
            self.logger.info(f"Denoising: {file_path.name}")

            # Load audio
            audio, sr = torchaudio.load(str(file_path))

            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Resample to model's expected sample rate if needed
            model_sr = self.df_state.sr()
            if sr != model_sr:
                audio = resample(audio, sr, model_sr)
                sr = model_sr

            # Apply denoising
            enhanced_audio = enhance(
                self.model,
                self.df_state,
                audio
            )

            # Resample to target sample rate if specified
            if self.sample_rate and sr != self.sample_rate:
                enhanced_audio = resample(enhanced_audio, sr, self.sample_rate)
                sr = self.sample_rate

            # Save denoised audio
            torchaudio.save(
                str(output_path),
                enhanced_audio.cpu(),
                sr,
                format=self.output_format
            )

            self.logger.debug(f"Created: {output_path.name}")

            return True, file_hash, None

        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            self.logger.error(f"Error processing {file_path.name}: {error_msg}")
            return False, file_hash, error_msg

    def _process_file_wrapper(self, file_path: Path) -> Tuple[bool, str, Optional[str]]:
        """Wrapper for processing that saves progress."""
        success, file_hash, error = self._process_file(file_path)
        if success:
            self._save_progress(file_hash)
        return success, file_hash, error

    def process_all(self):
        """Process all audio files."""
        audio_files = self.find_audio_files()

        if not audio_files:
            self.logger.info("No files to process!")
            return

        self.logger.info(f"Starting processing with {self.model_name}")
        if self.num_workers > 1:
            self.logger.warning(
                f"Note: DeepFilterNet is GPU/CPU intensive. "
                f"Using {self.num_workers} workers may be slow. "
                f"Consider using --workers 1 for better performance."
            )

        success_count = 0
        error_count = 0

        # Process files sequentially (DeepFilterNet doesn't parallelize well)
        for i, file_path in enumerate(audio_files, 1):
            if i % 10 == 0 or i == len(audio_files):
                self.logger.info(f"Progress: {i}/{len(audio_files)}")

            success, file_hash, error = self._process_file_wrapper(file_path)
            if success:
                success_count += 1
            else:
                error_count += 1

        self.logger.info(
            f"\nProcessing complete!\n"
            f"  Total processed: {success_count + error_count}\n"
            f"  Successful: {success_count}\n"
            f"  Errors: {error_count}"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Denoise audio files using DeepFilterNet ML model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DeepFilterNet Models:
  - DeepFilterNet3 (default): Latest model, best quality, slowest
  - DeepFilterNet2: Previous version, faster, good quality
  - DeepFilterNet: Original version, fastest, lower quality

Speed vs Quality Tradeoffs:
  For faster processing (sacrifice some quality):
  1. Use older model: --model DeepFilterNet (fastest)
  2. Disable post-filter: --no-post-filter (~30% faster)
  3. Use both: --model DeepFilterNet --no-post-filter (2-3x faster)

Installation:
  pip install deepfilternet
  or
  uv pip install deepfilternet

Examples:
  %(prog)s /path/to/audio
  %(prog)s /path/to/audio --model DeepFilterNet --no-post-filter  # Fast
  %(prog)s /path/to/audio --sample-rate 24000
  %(prog)s /path/to/audio --no-gpu --dry-run
        """
    )

    parser.add_argument(
        'directory',
        type=Path,
        help='Directory containing audio files to process'
    )

    # Model options
    model_group = parser.add_argument_group('model options')

    model_group.add_argument(
        '--model', '-m',
        type=str,
        default='DeepFilterNet3',
        choices=['DeepFilterNet', 'DeepFilterNet2', 'DeepFilterNet3'],
        help='DeepFilterNet model to use (default: DeepFilterNet3)'
    )

    model_group.add_argument(
        '--no-gpu',
        action='store_true',
        help='Force CPU processing (default: use GPU if available)'
    )

    model_group.add_argument(
        '--no-post-filter',
        action='store_true',
        help='Disable post-filter for faster processing (~30%% speed boost, slight quality loss)'
    )

    # Output options
    output_group = parser.add_argument_group('output options')

    output_group.add_argument(
        '--sample-rate', '-r',
        type=int,
        default=None,
        help='Resample to this sample rate in Hz (optional)'
    )

    output_group.add_argument(
        '--format', '-f',
        type=str,
        default='wav',
        choices=['wav', 'flac'],
        help='Output format (default: wav)'
    )

    output_group.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing denoised files (default: skip existing)'
    )

    # Processing options
    process_group = parser.add_argument_group('processing options')

    process_group.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Number of worker processes (default: 1, not recommended to increase)'
    )

    process_group.add_argument(
        '--progress-file', '-p',
        type=str,
        default='.denoise_ml_progress.json',
        help='Progress tracking file name (default: .denoise_ml_progress.json)'
    )

    process_group.add_argument(
        '--fresh-start',
        action='store_true',
        help='Ignore existing progress and reprocess all files'
    )

    process_group.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing progress without prompting'
    )

    process_group.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without actually processing files'
    )

    process_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Validate directory
    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Handle progress file logic
    use_existing_progress = True
    progress_path = args.directory / args.progress_file

    if progress_path.exists() and not args.fresh_start and not args.resume:
        # Progress file exists and user didn't specify what to do - prompt them
        try:
            with open(progress_path, 'r') as f:
                data = json.load(f)
                num_processed = len(data.get('processed', []))

            print(f"\nFound existing progress file with {num_processed} files already processed.")
            print("\nWhat would you like to do?")
            print("  1. Resume from existing progress (skip already processed files)")
            print("  2. Start fresh (reprocess all files and replace results)")

            while True:
                choice = input("\nEnter your choice (1 or 2): ").strip()
                if choice == '1':
                    use_existing_progress = True
                    print("Resuming from existing progress...\n")
                    break
                elif choice == '2':
                    use_existing_progress = False
                    print("Starting fresh - will reprocess all files...\n")
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        except Exception as e:
            print(f"Warning: Could not read progress file: {e}")
            print("Starting fresh...\n")
            use_existing_progress = False
    elif args.fresh_start:
        use_existing_progress = False
    elif args.resume:
        use_existing_progress = True

    # Create and run denoiser
    try:
        denoiser = MLAudioDenoiser(
            directory=args.directory,
            model_name=args.model,
            num_workers=args.workers,
            progress_file=args.progress_file,
            output_format=args.format,
            sample_rate=args.sample_rate,
            skip_existing=not args.overwrite,
            use_gpu=not args.no_gpu,
            post_filter=not args.no_post_filter,
            dry_run=args.dry_run,
            verbose=args.verbose,
            use_existing_progress=use_existing_progress
        )

        denoiser.process_all()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print(f"Run the script again to resume from where it left off.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
