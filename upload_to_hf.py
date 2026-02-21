#!/usr/bin/env python3
"""
Upload audio dataset to Hugging Face Hub.
"""
import argparse
from pathlib import Path
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import HfApi


# Common audio file extensions
AUDIO_EXTENSIONS = {
    '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma',
    '.opus', '.wav', '.aiff', '.ape', '.ac3', '.amr'
}


def create_dataset_from_audio_dir(audio_dir: Path):
    """
    Create a Hugging Face dataset from a directory of audio files.
    """
    print(f"Scanning directory: {audio_dir}")

    # Find all audio files
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
    audio_files = sorted(audio_files)

    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}. Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}")

    print(f"Found {len(audio_files)} audio files")

    # Show breakdown by file type
    from collections import Counter
    ext_counts = Counter(f.suffix.lower() for f in audio_files)
    print("File types:")
    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext}: {count} files")

    # Extract metadata from filenames (XC IDs)
    data = {
        "audio": [],
        "file_name": [],
        "xc_id": [],
    }

    for audio_file in audio_files:
        data["audio"].append(str(audio_file))
        data["file_name"].append(audio_file.name)

        # Extract XC ID from filename (e.g., "XC1065664.wav" -> "1065664")
        stem = audio_file.stem
        if stem.startswith("XC"):
            xc_id = stem[2:]
        else:
            xc_id = stem
        data["xc_id"].append(xc_id)

    # Define features with Audio type
    features = Features({
        "audio": Audio(sampling_rate=None),  # Will preserve original sample rate
        "file_name": Value("string"),
        "xc_id": Value("string"),
    })

    # Create dataset
    dataset = Dataset.from_dict(data, features=features)

    print(f"\nDataset created with {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    print(f"\nExample entry:")
    print(f"  file_name: {dataset[0]['file_name']}")
    print(f"  xc_id: {dataset[0]['xc_id']}")

    return dataset


def upload_dataset(dataset, repo_name: str, private: bool = False, token: str = None):
    """
    Upload dataset to Hugging Face Hub.
    """
    print(f"\nUploading to Hugging Face Hub: {repo_name}")
    print(f"Private: {private}")

    # Push to hub
    dataset.push_to_hub(
        repo_name,
        private=private,
        token=token,
    )

    print(f"\n✓ Dataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload audio files to Hugging Face Hub as a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload test_AB directory
  python upload_to_hf.py test_AB --repo my-username/xeno-canto-test

  # Upload as private dataset
  python upload_to_hf.py test_AB --repo my-username/xeno-canto-test --private

  # Specify token explicitly
  python upload_to_hf.py test_AB --repo my-username/xeno-canto-test --token hf_xxxxx

Note:
  You need to be logged in to Hugging Face:
    huggingface-cli login

  Or provide a token with --token
"""
    )

    parser.add_argument(
        "audio_dir",
        type=str,
        help="Directory containing audio files to upload (supports: .wav, .mp3, .flac, .ogg, etc.)"
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face repository name (format: username/dataset-name)"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (optional if logged in via huggingface-cli)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create dataset but don't upload"
    )

    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)

    if not audio_dir.exists():
        print(f"Error: Directory not found: {audio_dir}")
        return

    if not audio_dir.is_dir():
        print(f"Error: Not a directory: {audio_dir}")
        return

    # Create dataset
    try:
        dataset = create_dataset_from_audio_dir(audio_dir)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    if args.dry_run:
        print("\n[DRY RUN] Dataset created but not uploaded")
        return

    # Upload dataset
    try:
        upload_dataset(dataset, args.repo, args.private, args.token)
    except Exception as e:
        print(f"\nError uploading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're logged in: huggingface-cli login")
        print("  2. Check your token has write permissions")
        print("  3. Verify the repo name format: username/dataset-name")
        return


if __name__ == "__main__":
    main()
