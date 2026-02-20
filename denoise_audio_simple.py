#!/usr/bin/env python3
"""
Simple Statistical Audio Denoising Script

Silences sections of audio below a statistical threshold based on amplitude.
Supports multiprocessing and tracks progress for resumability.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Set, Optional, Tuple, List
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("Error: Required packages not installed. Please install them with:", file=sys.stderr)
    print("  pip install numpy soundfile", file=sys.stderr)
    print("\nOr with uv:", file=sys.stderr)
    print("  uv add numpy soundfile", file=sys.stderr)
    sys.exit(1)


# Common audio file extensions to process
AUDIO_EXTENSIONS = {
    '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma',
    '.opus', '.wav', '.aiff', '.ape', '.ac3', '.amr'
}


class SimpleAudioDenoiser:
    def __init__(
        self,
        directory: Path,
        threshold_std: float = 0.0,
        use_mean: bool = True,
        chunk_duration: float = 0.01,
        padding_chunks: int = 0,
        num_workers: int = None,
        progress_file: str = '.denoise_simple_progress.json',
        output_format: str = 'wav',
        sample_rate: int = None,
        skip_existing: bool = True,
        segment_output: bool = False,
        min_segment_duration: float = 0.1,
        dry_run: bool = False,
        verbose: bool = False,
        use_existing_progress: bool = True
    ):
        self.directory = directory
        self.threshold_std = threshold_std
        self.use_mean = use_mean
        self.chunk_duration = chunk_duration
        self.padding_chunks = padding_chunks
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.progress_file = directory / progress_file
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.skip_existing = skip_existing
        self.segment_output = segment_output
        self.min_segment_duration = min_segment_duration
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

    def _calculate_chunk_volumes(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Calculate RMS volume for each chunk of audio.

        Returns:
            Tuple of (chunk_volumes, chunk_size_samples)
        """
        # Calculate chunk size in samples
        chunk_size = int(self.chunk_duration * sr)

        # Pad audio to fit exact chunks
        num_chunks = int(np.ceil(len(audio) / chunk_size))
        padded_length = num_chunks * chunk_size
        padded_audio = np.pad(audio, (0, padded_length - len(audio)), mode='constant')

        # Reshape into chunks
        chunks = padded_audio.reshape(num_chunks, chunk_size)

        # Calculate RMS (root mean square) for each chunk
        chunk_volumes = np.sqrt(np.mean(chunks**2, axis=1))

        return chunk_volumes, chunk_size

    def _calculate_threshold(self, chunk_volumes: np.ndarray) -> float:
        """Calculate the volume threshold based on chunk statistics."""
        # Calculate mean and std of chunk volumes
        mean_vol = np.mean(chunk_volumes)
        std_vol = np.std(chunk_volumes)

        # Calculate threshold
        if self.use_mean:
            threshold = mean_vol + (self.threshold_std * std_vol)
        else:
            # Use just std deviations from zero
            threshold = self.threshold_std * std_vol

        return threshold

    def _expand_mask(self, chunk_mask: np.ndarray) -> np.ndarray:
        """
        Expand the chunk mask to include N chunks on either side of any True value.

        Args:
            chunk_mask: Boolean array indicating which chunks are above threshold

        Returns:
            Expanded boolean array
        """
        if self.padding_chunks == 0:
            return chunk_mask

        expanded_mask = np.copy(chunk_mask)

        # Find indices of chunks above threshold
        above_indices = np.where(chunk_mask)[0]

        # For each chunk above threshold, mark surrounding chunks
        for idx in above_indices:
            start = max(0, idx - self.padding_chunks)
            end = min(len(chunk_mask), idx + self.padding_chunks + 1)
            expanded_mask[start:end] = True

        return expanded_mask

    def _find_segments(self, chunk_mask: np.ndarray, chunk_size: int) -> List[Tuple[int, int]]:
        """
        Find continuous segments of chunks above threshold.

        Args:
            chunk_mask: Boolean array indicating which chunks are above threshold
            chunk_size: Number of samples per chunk

        Returns:
            List of (start_sample, end_sample) tuples
        """
        # Find transitions in chunk mask
        padded = np.pad(chunk_mask, (1, 1), mode='constant', constant_values=0)
        starts = np.where(np.diff(padded.astype(int)) == 1)[0]
        ends = np.where(np.diff(padded.astype(int)) == -1)[0]

        # Convert chunk indices to sample indices
        segments = []
        for start_chunk, end_chunk in zip(starts, ends):
            start_sample = start_chunk * chunk_size
            end_sample = end_chunk * chunk_size

            # Filter by minimum duration
            if (end_sample - start_sample) / chunk_size * self.chunk_duration >= self.min_segment_duration:
                segments.append((start_sample, end_sample))

        return segments

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

            # Skip if output already exists
            if self.skip_existing and output_path.exists():
                self.logger.debug(f"Skipping {file_path.name}: output already exists")
                return True, file_hash, None

            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would denoise: {file_path.name} -> {output_path.name}")
                return True, file_hash, None

            # Process the file
            self.logger.info(f"Processing: {file_path.name}")

            # Load audio
            audio, sr = sf.read(str(file_path), always_2d=True)

            # Convert to mono if stereo
            if audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            else:
                audio = audio[:, 0]

            # Resample if needed (do this before chunking)
            target_sr = self.sample_rate if self.sample_rate else sr
            if target_sr != sr:
                # Simple resampling using scipy if available, otherwise skip
                try:
                    from scipy import signal
                    num_samples = int(len(audio) * target_sr / sr)
                    audio = signal.resample(audio, num_samples)
                    sr = target_sr
                except ImportError:
                    self.logger.warning("scipy not available, skipping resampling")
                    target_sr = sr

            # Calculate chunk volumes
            chunk_volumes, chunk_size = self._calculate_chunk_volumes(audio, sr)

            # Calculate threshold from chunk volumes
            threshold = self._calculate_threshold(chunk_volumes)

            mean_vol = np.mean(chunk_volumes)
            std_vol = np.std(chunk_volumes)
            self.logger.debug(
                f"Chunk stats - Mean: {mean_vol:.6f}, Std: {std_vol:.6f}, Threshold: {threshold:.6f}"
            )
            self.logger.debug(f"Chunk size: {chunk_size} samples ({self.chunk_duration}s)")

            # Create mask of chunks above threshold
            chunk_mask = chunk_volumes >= threshold

            # Expand mask to include surrounding chunks
            chunk_mask = self._expand_mask(chunk_mask)

            if self.padding_chunks > 0:
                num_expanded = np.sum(chunk_mask) - np.sum(chunk_volumes >= threshold)
                self.logger.debug(f"Expanded {num_expanded} additional chunks (padding: {self.padding_chunks})")

            if self.segment_output:
                # Find segments of consecutive chunks above threshold
                segments = self._find_segments(chunk_mask, chunk_size)

                self.logger.info(f"Found {len(segments)} segments above threshold")

                # Save each segment as a separate file
                for i, (start, end) in enumerate(segments):
                    segment_stem = f"{file_path.stem}-segment{i:03d}"
                    segment_path = file_path.parent / f"{segment_stem}.{self.output_format}"

                    # Trim to actual audio length
                    end = min(end, len(audio))
                    segment_audio = audio[start:end]
                    sf.write(str(segment_path), segment_audio, sr)
                    self.logger.debug(f"Saved segment {i}: {segment_path.name}")

                # Also save the full denoised version with chunks zeroed out
                denoised = np.copy(audio)
                num_chunks = len(chunk_mask)
                for i, keep_chunk in enumerate(chunk_mask):
                    if not keep_chunk:
                        start = i * chunk_size
                        end = min((i + 1) * chunk_size, len(audio))
                        denoised[start:end] = 0.0
                sf.write(str(output_path), denoised, sr, format=self.output_format.upper())
            else:
                # Zero out entire chunks below threshold
                denoised = np.copy(audio)
                num_chunks = len(chunk_mask)

                for i, keep_chunk in enumerate(chunk_mask):
                    if not keep_chunk:
                        start = i * chunk_size
                        end = min((i + 1) * chunk_size, len(audio))
                        denoised[start:end] = 0.0

                # Calculate how much was silenced
                num_silenced = np.sum(~chunk_mask)
                silence_ratio = num_silenced / num_chunks
                self.logger.debug(f"Silenced {num_silenced}/{num_chunks} chunks ({silence_ratio*100:.1f}%)")

                # Save denoised audio
                sf.write(str(output_path), denoised, sr, format=self.output_format.upper())

            self.logger.debug(f"Created: {output_path.name}")

            return True, file_hash, None

        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            self.logger.error(f"Error processing {file_path.name}: {error_msg}")
            import traceback
            if self.verbose:
                self.logger.debug(traceback.format_exc())
            return False, file_hash, error_msg

    def _process_file_wrapper(self, file_path: Path) -> Tuple[bool, str, Optional[str]]:
        """Wrapper for processing that saves progress."""
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
        self.logger.info(f"Chunk duration: {self.chunk_duration}s")

        if self.use_mean:
            self.logger.info(f"Threshold: mean + {self.threshold_std} * std")
        else:
            self.logger.info(f"Threshold: {self.threshold_std} * std")

        if self.padding_chunks > 0:
            self.logger.info(f"Padding: {self.padding_chunks} chunks on either side")

        if self.segment_output:
            self.logger.info(f"Segment mode: ON (min duration: {self.min_segment_duration}s)")
        else:
            self.logger.info("Segment mode: OFF")

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
        description='Simple statistical audio denoising by silencing below threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How it works:
  1. Split audio into small chunks (default: 0.01s)
  2. Calculate RMS volume for each chunk
  3. Calculate mean and standard deviation of chunk volumes
  4. Set threshold = mean + (N * std)
  5. Zero out entire chunks below threshold
  6. Optionally segment non-zero sections into separate files

Threshold Modes:
  --use-mean (default): threshold = mean + (std-multiplier * std)
  --no-mean: threshold = std-multiplier * std (from zero)

Examples:
  # Silence chunks below mean (default)
  %(prog)s /path/to/audio

  # More aggressive: mean + 2*std
  %(prog)s /path/to/audio --std-multiplier 2.0

  # Keep 5 chunks on either side of loud chunks (prevents cutting off edges)
  %(prog)s /path/to/audio --padding-chunks 5

  # Smaller chunks (more granular)
  %(prog)s /path/to/audio --chunk-duration 0.005

  # Larger chunks (less granular, faster)
  %(prog)s /path/to/audio --chunk-duration 0.05

  # Just use std from zero (no mean)
  %(prog)s /path/to/audio --std-multiplier 2.0 --no-mean

  # Segment non-silent parts into separate files
  %(prog)s /path/to/audio --segment --min-segment-duration 0.2

  # Conservative with segmentation and padding
  %(prog)s /path/to/audio --std-multiplier 1.0 --segment --padding-chunks 3
        """
    )

    parser.add_argument(
        'directory',
        type=Path,
        help='Directory containing audio files to process'
    )

    # Threshold parameters
    threshold_group = parser.add_argument_group('threshold options')

    threshold_group.add_argument(
        '--std-multiplier', '-s',
        type=float,
        default=0.0,
        dest='threshold_std',
        help='Standard deviation multiplier for threshold (default: 0.0, just mean)'
    )

    threshold_group.add_argument(
        '--no-mean',
        action='store_false',
        dest='use_mean',
        help='Use only std from zero, not mean + std (more aggressive)'
    )

    threshold_group.add_argument(
        '--chunk-duration', '-c',
        type=float,
        default=0.01,
        help='Duration of each chunk in seconds (default: 0.01)'
    )

    threshold_group.add_argument(
        '--padding-chunks', '-p',
        type=int,
        default=0,
        dest='padding_chunks',
        help='Number of chunks to keep on either side of loud chunks (default: 0)'
    )

    # Segmentation options
    segment_group = parser.add_argument_group('segmentation options')

    segment_group.add_argument(
        '--segment',
        action='store_true',
        help='Save non-silent segments as separate files'
    )

    segment_group.add_argument(
        '--min-segment-duration',
        type=float,
        default=0.1,
        help='Minimum segment duration in seconds (default: 0.1)'
    )

    # Output options
    output_group = parser.add_argument_group('output options')

    output_group.add_argument(
        '--sample-rate', '-r',
        type=int,
        default=None,
        help='Resample to this sample rate in Hz (optional, requires scipy)'
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
        default=None,
        help='Number of worker processes (default: CPU count - 1)'
    )

    process_group.add_argument(
        '--progress-file', '-p',
        type=str,
        default='.denoise_simple_progress.json',
        help='Progress tracking file name (default: .denoise_simple_progress.json)'
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
    if args.threshold_std < 0:
        print(f"Error: Standard deviation multiplier must be non-negative", file=sys.stderr)
        sys.exit(1)

    if args.chunk_duration <= 0:
        print(f"Error: Chunk duration must be positive", file=sys.stderr)
        sys.exit(1)

    if args.padding_chunks < 0:
        print(f"Error: Padding chunks must be non-negative", file=sys.stderr)
        sys.exit(1)

    if args.min_segment_duration <= 0:
        print(f"Error: Minimum segment duration must be positive", file=sys.stderr)
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
        denoiser = SimpleAudioDenoiser(
            directory=args.directory,
            threshold_std=args.threshold_std,
            use_mean=args.use_mean,
            chunk_duration=args.chunk_duration,
            padding_chunks=args.padding_chunks,
            num_workers=args.workers,
            progress_file=args.progress_file,
            output_format=args.format,
            sample_rate=args.sample_rate,
            skip_existing=not args.overwrite,
            segment_output=args.segment,
            min_segment_duration=args.min_segment_duration,
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
