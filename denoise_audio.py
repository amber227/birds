#!/usr/bin/env python3
"""
Audio Denoising Script with Resume Capability

Applies FFT-based denoising and lowpass filtering to audio files.
Supports multiprocessing and tracks progress for resumability.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Set, Optional, Tuple


# Common audio file extensions to process
AUDIO_EXTENSIONS = {
    '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma',
    '.opus', '.wav', '.aiff', '.ape', '.ac3', '.amr'
}


class AudioDenoiser:
    def __init__(
        self,
        directory: Path,
        noise_reduction: float = 16.0,
        noise_floor: float = -60.0,
        lowpass_freq: int = 12000,
        sample_rate: int = None,
        num_workers: int = None,
        progress_file: str = '.denoise_progress.json',
        output_format: str = 'wav',
        skip_existing: bool = True,
        dry_run: bool = False,
        verbose: bool = False,
        use_existing_progress: bool = True
    ):
        self.directory = directory
        self.noise_reduction = noise_reduction
        self.noise_floor = noise_floor
        self.lowpass_freq = lowpass_freq
        self.sample_rate = sample_rate
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.progress_file = directory / progress_file
        self.output_format = output_format
        self.skip_existing = skip_existing
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

    def _build_filter_chain(self) -> str:
        """Build the FFmpeg audio filter chain."""
        filters = []

        # FFT Denoiser
        filters.append(f'afftdn=nr={self.noise_reduction}:nf={self.noise_floor}')

        # Lowpass filter (if specified)
        if self.lowpass_freq:
            filters.append(f'lowpass=f={self.lowpass_freq}')

        return ','.join(filters)

    def _process_file(self, file_path: Path) -> Tuple[bool, str, Optional[str]]:
        """
        Process a single audio file.

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
            temp_path = file_path.parent / f"{output_stem}.{self.output_format}.tmp"

            # Skip if output already exists
            if self.skip_existing and output_path.exists():
                self.logger.debug(f"Skipping {file_path.name}: output already exists")
                return True, file_hash, None

            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would denoise: {file_path.name} -> {output_path.name}")
                return True, file_hash, None

            # Build filter chain
            filter_chain = self._build_filter_chain()

            # Process using FFmpeg
            self.logger.info(f"Denoising: {file_path.name}")

            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-af', filter_chain,
            ]

            # Add sample rate if specified
            if self.sample_rate:
                cmd.extend(['-ar', str(self.sample_rate)])

            # Add output options
            cmd.extend([
                '-f', self.output_format,
                '-y',  # Overwrite output file
                '-loglevel', 'error',
                str(temp_path)
            ])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                error_msg = f"FFmpeg error: {result.stderr}"
                self.logger.error(f"Failed to denoise {file_path.name}: {error_msg}")
                # Clean up temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()
                return False, file_hash, error_msg

            # Move temp file to final location
            if temp_path.exists():
                temp_path.rename(output_path)

            self.logger.debug(f"Created: {output_path.name}")

            return True, file_hash, None

        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            self.logger.error(f"Error processing {file_path.name}: {error_msg}")
            return False, file_hash, error_msg

    def _process_file_wrapper(self, file_path: Path) -> Tuple[bool, str, Optional[str]]:
        """Wrapper for multiprocessing that also saves progress."""
        success, file_hash, error = self._process_file(file_path)
        if success:
            self._save_progress(file_hash)
        return success, file_hash, error

    def process_all(self):
        """Process all audio files using multiprocessing."""
        audio_files = self.find_audio_files()

        if not audio_files:
            self.logger.info("No files to process!")
            return

        self.logger.info(f"Starting processing with {self.num_workers} workers")
        self.logger.info(f"Denoising settings: nr={self.noise_reduction}, nf={self.noise_floor}")
        if self.lowpass_freq:
            self.logger.info(f"Lowpass filter: {self.lowpass_freq} Hz")

        success_count = 0
        error_count = 0

        if self.num_workers == 1:
            # Single-threaded processing
            for i, file_path in enumerate(audio_files, 1):
                self.logger.info(f"Progress: {i}/{len(audio_files)}")
                success, file_hash, error = self._process_file_wrapper(file_path)
                if success:
                    success_count += 1
                else:
                    error_count += 1
        else:
            # Multi-threaded processing
            with Pool(processes=self.num_workers) as pool:
                for i, (success, file_hash, error) in enumerate(
                    pool.imap_unordered(self._process_file_wrapper, audio_files),
                    1
                ):
                    if success:
                        success_count += 1
                    else:
                        error_count += 1

                    if i % 100 == 0 or i == len(audio_files):
                        self.logger.info(
                            f"Progress: {i}/{len(audio_files)} "
                            f"(Success: {success_count}, Errors: {error_count})"
                        )

        self.logger.info(
            f"\nProcessing complete!\n"
            f"  Total processed: {success_count + error_count}\n"
            f"  Successful: {success_count}\n"
            f"  Errors: {error_count}"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Denoise audio files using FFT denoiser and lowpass filter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Denoising Parameters:
  Noise Reduction (nr): Amount of noise reduction in dB (0-97)
    - Lower values (4-8): Gentle, preserves more signal
    - Medium values (8-12): Balanced noise reduction
    - Higher values (12+): Aggressive, may introduce artifacts

  Noise Floor (nf): Noise threshold in dB (-80 to -20)
    - Lower values (e.g., -70): More conservative, preserves more
    - Higher values (e.g., -40): More aggressive reduction

Examples:
  %(prog)s /path/to/audio
  %(prog)s /path/to/audio --nr 6 --nf -65 --lowpass 10000
  %(prog)s /path/to/audio --workers 8 --sample-rate 24000
  %(prog)s /path/to/audio --dry-run --verbose
        """
    )

    parser.add_argument(
        'directory',
        type=Path,
        help='Directory containing audio files to process'
    )

    # Denoising parameters
    denoise_group = parser.add_argument_group('denoising options')

    denoise_group.add_argument(
        '--nr', '--noise-reduction',
        type=float,
        default=8.0,
        dest='noise_reduction',
        help='Noise reduction amount in dB (default: 8.0, range: 0-97)'
    )

    denoise_group.add_argument(
        '--nf', '--noise-floor',
        type=float,
        default=-60.0,
        dest='noise_floor',
        help='Noise floor threshold in dB (default: -60.0, range: -80 to -20)'
    )

    denoise_group.add_argument(
        '--lowpass',
        type=int,
        default=12000,
        help='Lowpass filter frequency in Hz (default: 12000, 0 to disable)'
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
        choices=['wav', 'flac', 'mp3', 'ogg'],
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
        default=None,
        help='Number of worker processes (default: CPU count - 1)'
    )

    process_group.add_argument(
        '--progress-file', '-p',
        type=str,
        default='.denoise_progress.json',
        help='Progress tracking file name (default: .denoise_progress.json)'
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

    # Validate parameters
    if not 0 <= args.noise_reduction <= 97:
        print(f"Error: Noise reduction must be between 0 and 97", file=sys.stderr)
        sys.exit(1)

    if not -80 <= args.noise_floor <= -20:
        print(f"Error: Noise floor must be between -80 and -20", file=sys.stderr)
        sys.exit(1)

    # Check if ffmpeg is available
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg.", file=sys.stderr)
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
    denoiser = AudioDenoiser(
        directory=args.directory,
        noise_reduction=args.noise_reduction,
        noise_floor=args.noise_floor,
        lowpass_freq=args.lowpass if args.lowpass > 0 else None,
        sample_rate=args.sample_rate,
        num_workers=args.workers,
        progress_file=args.progress_file,
        output_format=args.format,
        skip_existing=not args.overwrite,
        dry_run=args.dry_run,
        verbose=args.verbose,
        use_existing_progress=use_existing_progress
    )

    try:
        denoiser.process_all()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print(f"Run the script again to resume from where it left off.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
