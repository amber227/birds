#!/usr/bin/env python3
"""
Audio Re-encoding Script with Resume Capability

Re-encodes audio files to WAV format with specified sample rate using FFmpeg.
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
import hashlib


# Common audio file extensions to process
AUDIO_EXTENSIONS = {
    '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma',
    '.opus', '.wav', '.aiff', '.ape', '.ac3', '.amr'
}


class AudioReencoder:
    def __init__(
        self,
        directory: Path,
        sample_rate: int = 24000,
        num_workers: int = None,
        progress_file: str = '.reencode_progress.json',
        force_reencode_wav: bool = False,
        dry_run: bool = False,
        verbose: bool = False
    ):
        self.directory = directory
        self.sample_rate = sample_rate
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.progress_file = directory / progress_file
        self.force_reencode_wav = force_reencode_wav
        self.dry_run = dry_run
        self.verbose = verbose

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
                if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    file_hash = self._get_file_hash(file_path)
                    if file_hash not in self.processed_files:
                        audio_files.append(file_path)

        self.logger.info(f"Found {len(audio_files)} audio files to process")
        return audio_files

    def _process_file(self, file_path: Path) -> Tuple[bool, str, Optional[str]]:
        """
        Process a single audio file.

        Returns:
            Tuple of (success, file_path, error_message)
        """
        file_hash = self._get_file_hash(file_path)

        try:
            # Check if file still exists (might have been deleted by another process)
            if not file_path.exists():
                return True, file_hash, "File no longer exists"

            # Determine if we need to process this file
            is_wav = file_path.suffix.lower() == '.wav'

            if is_wav and not self.force_reencode_wav:
                # Check if it's already at the target sample rate
                sample_rate = self._get_sample_rate(file_path)
                if sample_rate == self.sample_rate:
                    self.logger.debug(f"Skipping {file_path.name}: already WAV at {self.sample_rate}Hz")
                    return True, file_hash, None

            # Create output path
            output_path = file_path.with_suffix('.wav')
            temp_path = output_path.with_suffix('.wav.tmp')

            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would convert: {file_path.name}")
                return True, file_hash, None

            # Convert using FFmpeg
            self.logger.info(f"Converting: {file_path.name}")

            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-map_metadata', '0',  # Copy all metadata from input
                '-ar', str(self.sample_rate),
                '-ac', '1',  # Mono
                '-f', 'wav',  # Explicitly specify WAV format
                '-y',  # Overwrite output file
                '-loglevel', 'error',
                str(temp_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                error_msg = f"FFmpeg error: {result.stderr}"
                self.logger.error(f"Failed to convert {file_path.name}: {error_msg}")
                # Clean up temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()
                return False, file_hash, error_msg

            # Move temp file to final location
            if temp_path.exists():
                temp_path.rename(output_path)

            # Delete original if it's not already a .wav or if we're forcing reencode
            if file_path != output_path:
                file_path.unlink()
                self.logger.debug(f"Deleted original: {file_path.name}")

            return True, file_hash, None

        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            self.logger.error(f"Error processing {file_path.name}: {error_msg}")
            return False, file_hash, error_msg

    def _get_sample_rate(self, file_path: Path) -> Optional[int]:
        """Get the sample rate of an audio file using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=sample_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())
        except Exception as e:
            self.logger.debug(f"Could not determine sample rate for {file_path.name}: {e}")

        return None

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
        description='Re-encode audio files to WAV format with specified sample rate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/audio
  %(prog)s /path/to/audio --sample-rate 16000 --workers 8
  %(prog)s /path/to/audio --dry-run --verbose
  %(prog)s /path/to/audio --force-reencode-wav
        """
    )

    parser.add_argument(
        'directory',
        type=Path,
        help='Directory containing audio files to process'
    )

    parser.add_argument(
        '--sample-rate', '-r',
        type=int,
        default=24000,
        help='Target sample rate in Hz (default: 24000)'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of worker processes (default: CPU count - 1)'
    )

    parser.add_argument(
        '--progress-file', '-p',
        type=str,
        default='.reencode_progress.json',
        help='Progress tracking file name (default: .reencode_progress.json)'
    )

    parser.add_argument(
        '--force-reencode-wav',
        action='store_true',
        help='Re-encode WAV files even if they are already WAV format'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without actually processing files'
    )

    parser.add_argument(
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

    # Create and run reencoder
    reencoder = AudioReencoder(
        directory=args.directory,
        sample_rate=args.sample_rate,
        num_workers=args.workers,
        progress_file=args.progress_file,
        force_reencode_wav=args.force_reencode_wav,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    try:
        reencoder.process_all()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print(f"Run the script again to resume from where it left off.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
