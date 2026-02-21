#!/usr/bin/env python3
"""
Fix a corrupted VAE checkpoint by resetting NaN BatchNorm running statistics.
"""
import argparse
from pathlib import Path
import torch


def fix_checkpoint(input_path: Path, output_path: Path):
    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(str(input_path), map_location='cpu')

    if 'model_state_dict' not in ckpt:
        print("Error: Checkpoint does not contain 'model_state_dict'")
        return

    state = ckpt['model_state_dict']

    print("\nFixing NaN values in BatchNorm running statistics...")

    fixed_count = 0

    # Fix running_var (initialize to 1.0)
    for name, param in state.items():
        if 'running_var' in name:
            if torch.isnan(param).any():
                print(f"  Fixing {name}: resetting to ones")
                state[name] = torch.ones_like(param)
                fixed_count += 1

    # Fix running_mean (initialize to 0.0)
    for name, param in state.items():
        if 'running_mean' in name:
            if torch.isnan(param).any():
                print(f"  Fixing {name}: resetting to zeros")
                state[name] = torch.zeros_like(param)
                fixed_count += 1

    # Fix num_batches_tracked (initialize to 0)
    for name, param in state.items():
        if 'num_batches_tracked' in name:
            if torch.isnan(param).any():
                print(f"  Fixing {name}: resetting to zero")
                state[name] = torch.zeros_like(param)
                fixed_count += 1

    print(f"\nFixed {fixed_count} corrupted buffers")

    # Save fixed checkpoint
    print(f"\nSaving fixed checkpoint to: {output_path}")
    torch.save(ckpt, str(output_path))
    print("✓ Done!")

    print("\n" + "=" * 70)
    print("IMPORTANT NOTE")
    print("=" * 70)
    print("The decoder's BatchNorm statistics have been reset to default values.")
    print("This means the decoder will use default normalization during inference.")
    print("The model should work now, but quality might be affected.")
    print()
    print("For best results, you should:")
    print("1. Investigate why the decoder had NaN during training")
    print("2. Retrain the model with a fix")
    print()
    print("Possible causes of NaN during training:")
    print("  - Gradient explosion in the decoder")
    print("  - Numerical instability in ConvTranspose2d")
    print("  - Bug in the lazy layer initialization")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Fix corrupted BatchNorm stats in VAE checkpoint"
    )
    parser.add_argument("input", type=str, help="Input checkpoint .pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output checkpoint file (default: INPUT_fixed.pt)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input checkpoint not found: {input_path}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_fixed")

    if output_path.exists():
        response = input(f"Output file {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    fix_checkpoint(input_path, output_path)


if __name__ == "__main__":
    main()
